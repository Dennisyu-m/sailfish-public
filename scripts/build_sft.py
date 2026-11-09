"""
用 dedup.py 选出的 4 万道题造 Stage 2 / Stage 3 数据（§4.3）：
  格式 A（指令式，和预训练里的 OCI 一致）：prompt = 题目 + "\n\n"，completion = "```python\n代码\n```"
  格式 B（HumanEval 式补全）：prompt = import + 前置辅助函数 + 函数签名 + docstring（没有就用题目首段生成），
                              completion = 函数体（之后接 <|endoftext|>）
另做 §4.4 第 3 步的简化版：每道题和 HumanEval+/MBPP+ 每道题算词集合 Jaccard 相似度，
把最高的 30 对写进 sft_similarity_top.txt 供人工抽查，>= --drop-sim 的直接剔除。
输出 data/packed/sft.jsonl：{"id", "fmt", "prompt", "completion"}
"""
import argparse
import ast
import json
import os
import re
import textwrap
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CLEAN = Path(os.environ.get("NANO_CLEAN", ROOT / "data" / "clean"))
OUT = Path(os.environ.get("NANO_PACKED", ROOT / "data" / "packed"))


def first_paragraph(text, limit=400):
    para = text.strip().split("\n\n")[0].strip()
    para = re.sub(r"\s+", " ", para)
    return para[:limit].rsplit(" ", 1)[0] if len(para) > limit else para


def to_completion_format(prompt, code):
    """返回 (prompt_b, completion_b) 或 None。"""
    try:
        tree = ast.parse(code)
    except Exception:  # noqa: BLE001
        return None
    lines = code.split("\n")
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None
    names = re.findall(r"`(\w+)\(", prompt)
    main = next((f for f in funcs if f.name in names), funcs[-1] if len(funcs) > 1 else funcs[0])
    if not main.body:
        return None
    head_end = main.body[0].lineno - 1          # def 头（含多行签名、装饰器）结束行（1-based 包含）
    start = (main.decorator_list[0].lineno if main.decorator_list else main.lineno) - 1
    pre = []
    for n in tree.body:
        if n is main:
            break
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign)):
            s0 = n.decorator_list[0].lineno if getattr(n, "decorator_list", None) else n.lineno
            pre.append("\n".join(lines[s0 - 1:n.end_lineno]))
    header = "\n".join(lines[start:head_end])
    body_nodes = main.body
    doc = ast.get_docstring(main, clean=False)
    if doc is not None and isinstance(body_nodes[0], ast.Expr):
        doc_src = "\n".join(lines[body_nodes[0].lineno - 1:body_nodes[0].end_lineno])
        rest = body_nodes[1:]
    else:
        indent = re.match(r"\s*", lines[main.body[0].lineno - 1]).group(0) or "    "
        wrapped = textwrap.fill(first_paragraph(prompt), 88 - len(indent))
        doc_src = indent + '"""\n' + textwrap.indent(wrapped, indent) + "\n" + indent + '"""'
        rest = body_nodes
    if not rest:
        return None
    body = "\n".join(lines[rest[0].lineno - 1:main.end_lineno])
    p = ("\n\n".join(pre) + "\n\n\n" if pre else "") + header + "\n" + doc_src + "\n"
    return p, body.rstrip() + "\n"


def words(s):
    return set(re.findall(r"[a-z_]+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-sim", type=float, default=0.6)
    args = ap.parse_args()
    idx = json.load(open(CLEAN / "_sft_index.json"))
    evals = []
    for path in (ROOT / "data/eval/HumanEvalPlus.jsonl", ROOT / "data/eval/MbppPlus.jsonl"):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            evals.append((r["task_id"], words(r["prompt"]), r["prompt"][:200]))
    out, sims, nA, nB, dropped = [], [], 0, 0, 0
    for item in idx:
        t = pq.read_table(item["file"], columns=["prompt", "code"]).to_pydict()
        for r in item["rows"]:
            prompt, code = t["prompt"][r], t["code"][r]
            w = words(prompt)
            best = max(((len(w & e) / max(1, len(w | e)), tid, ep) for tid, e, ep in evals), key=lambda x: x[0])
            sims.append((best[0], best[1], prompt[:200], best[2]))
            if best[0] >= args.drop_sim:
                dropped += 1
                continue
            uid = f"{Path(item['file']).stem}:{r}"
            out.append({"id": uid, "fmt": "A", "prompt": prompt.strip() + "\n\n",
                        "completion": "```python\n" + code.strip() + "\n```"})
            nA += 1
            b = to_completion_format(prompt, code)
            if b:
                out.append({"id": uid, "fmt": "B", "prompt": b[0], "completion": b[1]})
                nB += 1
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sft.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    sims.sort(key=lambda x: -x[0])
    with open(OUT / "sft_similarity_top.txt", "w", encoding="utf-8") as f:
        for s, tid, p, ep in sims[:30]:
            f.write(f"{s:.3f}  {tid}\n  训练题: {p!r}\n  评测题: {ep!r}\n\n")
    print(f"题目 {len(sims)}，相似度>={args.drop_sim} 剔除 {dropped}；格式A {nA} 条，格式B {nB} 条 -> {OUT / 'sft.jsonl'}")


if __name__ == "__main__":
    main()
