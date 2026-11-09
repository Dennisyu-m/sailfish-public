"""
校验 gen_deepseek.py 造的题，只留"自己跑得通"的：
  math：在子进程里跑 code，算出的 answer 必须和题目给的 answer、解答末尾 "The answer is X" 三者数值一致
  code：参考实现 + 5 条 assert 全部通过（子进程，超时 5 秒）
再做去污染（数学对 GSM8K/MATH 测试题；编程对 HumanEval/MBPP 的函数名+参数与 13 词片段）和题目去重。
输出（精调格式）：
  data/gen/math_ok.jsonl：每题两条，CoT（题目 -> 文字推理）和 PoT（题目 -> ```python 代码``` + The answer is X）
  data/gen/code_ok.jsonl：HumanEval 式（prompt = 签名 + docstring，completion = 函数体）
用法：python scripts/check_gen.py --kind math
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
GEN = ROOT / "data" / "gen"


def run_py(code, timeout=5):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return -9, ""
    finally:
        Path(path).unlink(missing_ok=True)


def num(x):
    x = str(x).replace(",", "").replace("$", "").strip().rstrip(".")
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", x)
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(x)
    except ValueError:
        return None


def check_math(it):
    p, s, c, a = (str(it.get(k, "")) for k in ("problem", "solution", "code", "answer"))
    if len(p) < 20 or "The answer is" not in s or "answer" not in c:
        return None, "format"
    # 出题模型写到一半改题（"Wait… let me adjust the profit to $1,985"）：题面没改、答案按改后的算，三者照样一致 → 剔
    if re.search(r"\b(Wait|wait|recheck|Re-check|Let me|let me|Corrected|corrected|adjust|instead of|not an integer|"
                 r"not a whole|isn't a whole|doesn't divide evenly|Hmm)\b", s + "\n" + c):
        return None, "self_fix"
    # 代码里可能自己 print 过；用标记行取答案，Fraction/Decimal/int 统一转 float
    rc, out = run_py(c + "\ntry:\n    print('@@ANS@@', float(answer))\nexcept Exception:\n    print('@@ANS@@', answer)\n")
    if rc != 0:
        return None, "code_error"
    out = out.rsplit("@@ANS@@", 1)[-1] if "@@ANS@@" in out else ""
    got, stated = num(out.strip().strip("'\"")), num(a)
    said = num(s.rsplit("The answer is", 1)[1].strip().split()[0] if s.rsplit("The answer is", 1)[1].strip() else "")
    if got is None or stated is None or said is None:
        return None, "not_numeric"
    if not (abs(got - stated) <= 1e-6 * max(1, abs(stated)) and abs(said - stated) <= 1e-6 * max(1, abs(stated))):
        return None, "mismatch"
    return it, "ok"


def check_code(it):
    sol, tests = str(it.get("solution", "")), it.get("tests", [])
    if not isinstance(tests, list) or len(tests) < 3 or "def " not in sol:
        return None, "format"
    if any(w in sol for w in ("input(", "open(", "import os", "subprocess", "random")):
        return None, "unsafe"
    rc, _ = run_py(sol + "\n\n" + "\n".join(tests) + "\nprint('PASS')\n")
    return (it, "ok") if rc == 0 else (None, "test_fail")


def code_to_sft(it):
    tree = ast.parse(it["solution"])
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return None
    lines = it["solution"].splitlines()
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                           and isinstance(fn.body[0].value.value, str)) else fn.body
    if not body:
        return None
    pre = "\n".join(lines[:fn.lineno - 1]).strip()  # 函数前的 import / 辅助代码
    header = "\n".join(lines[fn.lineno - 1:fn.body[0].lineno - 1]) if fn.body[0].lineno > fn.lineno else lines[fn.lineno - 1]
    header = header.split("\n")[0] if '"""' in header else header
    doc = it.get("docstring", "").strip().replace('"""', "'''")
    doc = "\n".join(x if k == 0 else ("    " + x if x.strip() else "") for k, x in enumerate(doc.splitlines()))  # 续行缩进 4 格
    prompt = (pre + "\n\n" if pre else "") + header.rstrip() + '\n    """' + doc + '\n    """\n'
    completion = "\n".join(lines[body[0].lineno - 1:fn.end_lineno]) + "\n"
    return {"prompt": prompt, "completion": completion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["code", "math"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--inp", default="", help="默认 <kind>_raw.jsonl")
    ap.add_argument("--out", default="", help="默认 <kind>_ok.jsonl")
    args = ap.parse_args()
    raw = [json.loads(l) for l in open(GEN / (args.inp or f"{args.kind}_raw.jsonl"), encoding="utf-8")]
    from collections import Counter
    fn = check_math if args.kind == "math" else check_code
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(fn, raw))
    why = Counter(r[1] for r in res)
    good = [r[0] for r in res if r[0] is not None]
    # 去污染 + 去重
    from prep_math import MathDecontam, norm
    seen, out, drop = set(), [], Counter()
    if args.kind == "math":
        dec = MathDecontam()
        for it in good:
            k = norm(it["problem"])
            if k in seen:
                drop["dup"] += 1
                continue
            if dec.hit(it["problem"], it["problem"] + "\n\n" + it["solution"]):
                drop["contam"] += 1
                continue
            seen.add(k)
            ans = str(it["answer"]).strip()
            out.append({"src": "ds_math_cot", "prompt": it["problem"] + "\n\n", "completion": it["solution"], "answer": ans})
            out.append({"src": "ds_math_pot", "prompt": it["problem"] + "\n\n",
                        "completion": "```python\n" + it["code"].strip() + "\n```\nThe answer is " + ans + ".", "answer": ans})
    else:
        from datautil import Decontam, word_ids
        dec = Decontam(ROOT / "data/eval/HumanEvalPlus.jsonl", ROOT / "data/eval/MbppPlus.jsonl")
        for it in good:
            k = it.get("signature", "")
            if k in seen:
                drop["dup"] += 1
                continue
            names = re.findall(r"def\s+(\w+)\s*\(", it["solution"])  # 造的题更严：和评测题重名就丢，不看参数
            if dec.name_hit(it["solution"]) or any(n in dec.names for n in names) or dec.ngram_hit(word_ids(it.get("docstring", "") + "\n" + it["solution"])):
                drop["contam"] += 1
                continue
            seen.add(k)
            try:
                r = code_to_sft(it)
            except SyntaxError:
                r = None
            if r:
                out.append(dict(r, src="ds_code"))
    outp = GEN / (args.out or f"{args.kind}_ok.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{args.kind}] 原始 {len(raw)} 道：校验 {dict(why)}；通过 {len(good)}，去重/去污染 {dict(drop)}；"
          f"写出 {len(out)} 条 -> {outp}")


if __name__ == "__main__":
    main()
