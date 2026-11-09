"""
数学评测：GSM8K test（1319 题）+ MATH-500。提示词 = 题目 + "\n\n"（和精调格式一致），批量贪心生成，遇 <|endoftext|> 停。
取答案：最后一个 \\boxed{...}；没有就取 "The answer is ..." / "#### ..."；GSM8K 再退到最后一个数字。
判分：GSM8K 按数值相等；MATH 先做常见 LaTeX 规范化再比字符串，能转成数的按数值比。
用法：python scripts/eval_math.py --ckpt runs/v2/sft_final.pt --tokenizer data/packed_v2/tokenizer.json --out-dir runs/v2/eval_math
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import nanocode_lib as L  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1] if depth == 0 else None


def extract(s, kind):
    b = last_boxed(s)
    if b is not None:
        return b
    for pat in (r"The answer is:?\s*(.+)", r"####\s*(.+)"):
        m = re.findall(pat, s)
        if m:
            return m[-1].strip().rstrip(".")
    if kind == "gsm8k":
        m = NUM.findall(s)
        return m[-1] if m else ""
    return ""


def to_num(x):
    x = x.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    m = re.fullmatch(r"-?\d+(\.\d+)?", x)
    if m:
        return float(x)
    m = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", x)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    return None


def norm_latex(x):
    x = x.strip().strip("$").strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\!", ""),
                 ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\text{ }", ""),
                 ("\\$", ""), (" ", "")):
        x = x.replace(a, b)
    x = re.sub(r"\\text\{(.*?)\}", r"\1", x)
    x = re.sub(r"\\mbox\{(.*?)\}", r"\1", x)
    x = re.sub(r"\\sqrt([0-9a-zA-Z])", r"\\sqrt{\1}", x)  # \sqrt2 == \sqrt{2}
    x = re.sub(r"\\frac([0-9a-zA-Z])([0-9a-zA-Z])", r"\\frac{\1}{\2}", x)  # \frac12 == \frac{1}{2}
    x = re.sub(r"\\frac\{([^{}]+)\}([0-9a-zA-Z])", r"\\frac{\1}{\2}", x)
    x = re.sub(r"\\frac([0-9a-zA-Z])\{", r"\\frac{\1}{", x)
    return x.rstrip(".")


def correct(pred, gold, kind):
    if pred is None or pred == "":
        return False
    a, b = to_num(norm_latex(pred)), to_num(norm_latex(gold))
    if a is not None and b is not None:
        return abs(a - b) <= 1e-6 * max(1, abs(b))
    return kind == "math" and norm_latex(pred) == norm_latex(gold)


def load_problems(limit):
    raw = ROOT / "data" / "raw"
    out = []
    t = pq.read_table(next((raw / "openai/gsm8k/main").glob("test-*.parquet"))).to_pydict()
    for q, a in zip(t["question"], t["answer"]):
        out.append(("gsm8k", q, a.rpartition("####")[2].strip()))
    for l in open(raw / "HuggingFaceH4/MATH-500/test.jsonl", encoding="utf-8"):
        r = json.loads(l)
        out.append(("math500", r["problem"], r["answer"]))
    if limit:
        out = [x for x in out if x[0] == "gsm8k"][:limit] + [x for x in out if x[0] == "math500"][:limit]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    dev = torch.device(args.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    cfgp = Path(args.ckpt).parent / "model_cfg.json"
    m = L.ScratchNanoCode10K(max_len=1024, **(json.load(open(cfgp)) if cfgp.exists() else {}))
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model", sd))
    m = m.to(device=dev, dtype=dtype).eval()
    probs = load_problems(args.limit)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    order = sorted(range(len(probs)), key=lambda i: len(probs[i][1]))  # 按长度分批，少填充
    for k in range(0, len(order), args.bsz):
        part = [probs[i] for i in order[k:k + args.bsz]]
        ps = [[L.EOT, L.EOT] + tok.encode(q + "\n\n").ids for _, q, _ in part]
        mx = min(args.max_new, m.max_len - max(len(p) for p in ps) - 1)
        with torch.no_grad():
            gens = L.greedy_generate(m, ps, mx)
        for (kind, q, gold), g in zip(part, gens):
            txt = tok.decode(g)
            pred = extract(txt, "gsm8k" if kind == "gsm8k" else "math")
            recs.append({"set": kind, "problem": q, "gold": gold, "pred": pred,
                         "ok": correct(pred, gold, "gsm8k" if kind == "gsm8k" else "math"), "output": txt})
        print(f"  {len(recs)}/{len(probs)}，{time.time() - t0:.0f}s", flush=True)
    summ = {}
    for kind in ("gsm8k", "math500"):
        r = [x for x in recs if x["set"] == kind]
        if r:
            summ[kind] = {"n": len(r), "acc": sum(x["ok"] for x in r) / len(r)}
    with open(out / "math_samples.jsonl", "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(summ, open(out / "math_summary.json", "w"), indent=1)
    print("[数学评测]", json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
