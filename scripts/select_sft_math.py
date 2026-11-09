"""
挑数学精调 / 强化学习题，并把这些题从预训练里扣掉（写 data/clean2/_sft_math_h.npy，pack_v2 读取）。
  - gsm8k_train、math_train：人写解答，全要
  - omi2：随机 --omi2 道（只挑有 expected_answer 的，可自动判分）
  - numina：随机 --numina 道（只挑能取出 \\boxed{} 答案的）
输出 data/sft_math.jsonl：{"src","prompt","completion","answer"}，prompt = 题目 + "\n\n"，completion = 解题过程
用法：venv/Scripts/python scripts/select_sft_math.py
"""
import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CLEAN2 = ROOT / "data" / "clean2"


def rows(src):
    for f in sorted(glob.glob(str(CLEAN2 / src / "*.parquet"))):
        keep = np.load(CLEAN2 / "_keep" / src / (Path(f).stem + ".npy"))
        t = pq.read_table(f, columns=["problem", "solution", "answer", "h"]).to_pydict()
        for i in range(len(keep)):
            if keep[i]:
                yield {k: t[k][i] for k in t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi2", type=int, default=40000)
    ap.add_argument("--numina", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    picked = []
    for src in ("gsm8k_train", "math_train"):
        picked += [dict(r, src=src) for r in rows(src)]
    for src, n in (("omi2", args.omi2), ("numina", args.numina)):
        pool = [r for r in rows(src) if r["answer"].strip()]
        seen, cand = set(), []
        for r in pool:  # omi2 一题多解：一道题只挑一份
            if r["h"] not in seen:
                seen.add(r["h"])
                cand.append(r)
        picked += [dict(r, src=src) for r in rng.sample(cand, min(n, len(cand)))]
    hs = np.array(sorted({r["h"] for r in picked}), dtype=np.uint64)
    np.save(CLEAN2 / "_sft_math_h.npy", hs)
    with open(ROOT / "data" / "sft_math.jsonl", "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({"src": r["src"], "prompt": r["problem"] + "\n\n", "completion": r["solution"],
                                "answer": r["answer"]}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"数学精调题 {len(picked)} 道（不同题 {hs.size}）：{dict(Counter(r['src'] for r in picked))}")
    print(f"-> data/sft_math.jsonl；预训练排除列表 data/clean2/_sft_math_h.npy")


if __name__ == "__main__":
    main()
