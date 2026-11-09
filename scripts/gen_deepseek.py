"""
用 DeepSeek API（OpenAI 兼容接口）批量造可校验的训练题。官方开放平台协议 4.2(3) 明确允许用输出训练其他模型（蒸馏）。
两种题：
  code：函数签名 + docstring（含 >>> 示例）+ 参考实现 + 5 条 assert 测试
  math：小学到高中难度应用题 + 文字推理 + 一段 Python 验算代码 + 最终答案（数值）
种子 = 知识点 × 难度 × 场景，随机组合保证多样性；每次请求让模型一次出 --per 道题（JSON 数组），省输入 token。
安全：API Key 只从环境变量 DEEPSEEK_API_KEY 读；带花费上限 --budget（元），用响应里的 usage 实时累计，超了就停。
输出：data/gen/<kind>_raw.jsonl（原始），校验见 check_gen.py。
用法（先试跑）：python scripts/gen_deepseek.py --kind math --requests 50 --per 5 --budget 3 --model <模型名>
  模型名、价格以 DeepSeek 开放平台文档为准（--price-in / --price-out，元/百万 token）。低谷时段更便宜。
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gen"
API = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"

CODE_TOPICS = [
    "string parsing", "string formatting", "palindromes and reversals", "counting characters", "list filtering",
    "list deduplication preserving order", "sorting with custom keys", "two pointers", "sliding window",
    "prefix sums", "binary search", "dictionary counting", "grouping items", "nested lists / matrices",
    "matrix rotation and transpose", "recursion", "simple dynamic programming", "greedy choice", "stack usage",
    "queue / BFS on a grid", "interval merging", "date and time arithmetic without libraries", "number digits",
    "prime numbers", "gcd / lcm", "base conversion", "bit manipulation", "run-length encoding", "simple ciphers",
    "validating input formats", "tuple and set operations", "generators and iteration", "edge cases with empty input",
    "floating point rounding", "simulation of a small process", "text statistics", "frequency ranking",
]
MATH_TOPICS = [
    "arithmetic word problem", "percentages and discounts", "ratios and proportions", "rates, speed and time",
    "unit conversion", "fractions", "averages", "money and change", "work and rates", "simple interest",
    "area and perimeter", "volume", "angles", "linear equations", "systems of two linear equations",
    "quadratic equations", "inequalities", "sequences and series", "counting and combinations", "probability",
    "divisibility and remainders", "primes and factors", "gcd and lcm", "digits of numbers", "exponents",
    "functions and evaluation", "coordinate geometry", "Pythagorean theorem", "age problems", "mixtures",
]
LEVELS = ["very easy (grade 3-4)", "easy (grade 5-6)", "medium (grade 7-8)", "harder (grade 9-10)", "hard (grade 11-12)"]
SCENES = ["a school", "a bakery", "a farm", "a sports team", "a library", "a train trip", "a small shop", "a garden",
          "a game", "a science lab", "a factory", "a city park", "a birthday party", "a music band", "a hospital",
          "a bank account", "a recipe", "a road trip", "a class election", "an aquarium"]
# 放量时加的多样性维度（每次请求随机挑一个，写进提示词的 Extra 行）
MATH_EXTRA = ["needs exactly 2 steps", "needs 3-4 steps", "needs 5 or more steps", "the answer is a fraction",
              "the answer is a decimal", "the answer is a whole number", "includes one irrelevant number",
              "works backwards from a final amount", "compares two options", "uses a table of values described in words"]
CODE_EXTRA = ["input is a list of integers", "input is a string", "input is a dict", "input is a list of tuples",
              "input is a 2D grid (list of lists)", "takes two arguments", "takes an optional keyword argument",
              "returns a tuple", "must run in linear time", "must handle negative numbers", "must handle unicode text"]

PROMPT_CODE = """Create {per} different, self-contained Python programming exercises.
Topic: {topic}. Difficulty: {level}. Theme for names and data: {scene}. Extra: {extra}.
Rules:
- Each exercise is ONE Python function, standard library only, no input(), no printing, no files, no randomness.
- Give a clear docstring describing behavior, inputs, outputs and edge cases, with 2 doctest examples (>>>).
- Give a correct reference implementation and exactly 5 assert statements that test it (include an edge case).
Return ONLY a JSON array; each item: {{"signature": "def name(args) -> type:", "docstring": "...", "solution": "full function code", "tests": ["assert ...", ...]}}"""

PROMPT_MATH = """Create {per} different math word problems with full solutions.
Topic: {topic}. Difficulty: {level}. Setting: {scene}. Extra: {extra}.
Rules:
- Each problem has a single numerical final answer (integer, or a decimal/fraction that can be written exactly).
- "solution": clear step-by-step reasoning in English, ending with the sentence "The answer is X."
- "code": a short Python program that computes the answer from scratch and stores it in a variable named answer.
- Do not copy well-known textbook or competition problems; invent new numbers and situations.
Return ONLY a JSON array; each item: {{"problem": "...", "solution": "...", "code": "...", "answer": "X"}}"""


class Budget:
    def __init__(self, yuan, p_in, p_out):
        self.yuan, self.p_in, self.p_out = yuan, p_in, p_out
        self.spent, self.tok_in, self.tok_out, self.lock = 0.0, 0, 0, threading.Lock()

    def add(self, usage):
        with self.lock:
            ti, to = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            k = 2.0 if is_peak() else 1.0  # 高峰时段价格翻倍（节假日按高峰算，偏保守）
            self.tok_in += ti
            self.tok_out += to
            self.spent += k * (ti / 1e6 * self.p_in + to / 1e6 * self.p_out)

    def over(self):
        return self.spent >= self.yuan


def is_peak():
    """DeepSeek 高峰：北京时间周一至周五 9-12、14-18 点（其余为空闲价，即高峰价的一半）。"""
    t = datetime.now(timezone(timedelta(hours=8)))
    return t.weekday() < 5 and (9 <= t.hour < 12 or 14 <= t.hour < 18)


def call(key, model, prompt, thinking="off", timeout=600):
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0}
    if thinking == "off":
        req.update(thinking={"type": "disabled"}, max_tokens=8000)
    else:
        req.update(thinking={"type": "enabled"}, reasoning_effort=thinking, max_tokens=32000)
    body = json.dumps(req).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_items(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    i, j = t.find("["), t.rfind("]")
    return json.loads(t[i:j + 1]) if i >= 0 and j > i else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["code", "math"], required=True)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--budget", type=float, required=True, help="本次最多花多少元，超了就停")
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", ""))
    ap.add_argument("--price-in", type=float, default=1.0, help="元/百万输入 token（按官方价格填）")
    ap.add_argument("--price-out", type=float, default=4.0, help="元/百万输出 token（按官方价格填）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--levels", default="0,1,2,3,4", help="难度档下标，逗号分隔（0 最易 ~ 4 最难）")
    ap.add_argument("--thinking", default="off", choices=["off", "low", "high"], help="思考模式：off 最省；low/high 更准但思考 token 按输出计费")
    ap.add_argument("--out", default="", help="输出文件名（默认 <kind>_raw.jsonl）")
    ap.add_argument("--no-peak", action="store_true", help="到高峰时段就停（只用空闲价）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    assert key, "请先设置环境变量 DEEPSEEK_API_KEY（不要写进代码或发到聊天里）"
    assert args.model, "请用 --model 或环境变量 DEEPSEEK_MODEL 指定模型名（以官方文档为准）"
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / (args.out or f"{args.kind}_raw.jsonl")
    rng = random.Random(args.seed + int(time.time()))
    topics, tmpl = (CODE_TOPICS, PROMPT_CODE) if args.kind == "code" else (MATH_TOPICS, PROMPT_MATH)
    extra = CODE_EXTRA if args.kind == "code" else MATH_EXTRA
    levels = [LEVELS[int(i)] for i in args.levels.split(",")]
    bud = Budget(args.budget, args.price_in, args.price_out)
    lock, stats = threading.Lock(), {"ok": 0, "items": 0, "fail": 0}

    def one(i):
        if bud.over() or (args.no_peak and is_peak()):
            return
        seed = {"topic": rng.choice(topics), "level": rng.choice(levels), "extra": rng.choice(extra), "scene": rng.choice(SCENES), "per": args.per}
        for attempt in range(3):
            try:
                r = call(key, args.model, tmpl.format(**seed), args.thinking)
                bud.add(r.get("usage", {}))
                items = parse_items(r["choices"][0]["message"]["content"])
                with lock:
                    with open(outp, "a", encoding="utf-8") as f:
                        for it in items:
                            if isinstance(it, dict):
                                f.write(json.dumps(dict(it, _seed=seed, _model=args.model, _thinking=args.thinking), ensure_ascii=False) + "\n")
                    stats["ok"] += 1
                    stats["items"] += len(items)
                return
            except Exception as e:  # noqa: BLE001
                err = str(e)[:120]
                time.sleep(2 + 3 * attempt)
        with lock:
            stats["fail"] += 1
            print("  失败:", err, flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(one, range(args.requests))):
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{args.requests} 请求，{stats['items']} 道题，花费约 ¥{bud.spent:.3f}"
                      f"（输入 {bud.tok_in:,} / 输出 {bud.tok_out:,} token），{time.time() - t0:.0f}s", flush=True)
    print(f"[完成] 成功 {stats['ok']} 失败 {stats['fail']}，共 {stats['items']} 道题 -> {outp}；"
          f"花费约 ¥{bud.spent:.3f}（按 {args.price_in}/{args.price_out} 元每百万 token 估算，以账单为准）")


if __name__ == "__main__":
    main()
