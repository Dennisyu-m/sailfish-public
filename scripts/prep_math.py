"""
v2 数学 / 英文数据清洗：data/raw/... -> data/clean2/<src>/<stem>.parquet（+ .stats.json），再做全局去重 -> data/clean2/_keep/<src>/<stem>.npy
  统一格式：题目 + "\n\n" + 解题过程（和编程精调格式 A 一致）。
  列：text（预训练用全文）、problem、solution、answer（可空）、h（题目规范化哈希，用于去重）
去污染（对 GSM8K test 1319 题、MATH test 5000 题（含 MATH-500）、HumanEval/MBPP）：
  (1) 题目规范化后完全相同；(2) 题目开头 60 个规范化字符相同；(3) 任意 13 词片段重合（题目+解答）。
全局去重：同一道题（h 相同）最多保留 2 份解答（OpenMathInstruct-2 一题多解）；网页类（finemath/fwedu）按全文哈希去重。
用法：venv/Scripts/python scripts/prep_math.py --workers 6
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from datautil import ngram_hashes, word_ids  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = Path(os.environ.get("NANO_CLEAN2", ROOT / "data" / "clean2"))
EVAL = ROOT / "data" / "eval"

# 源 -> (glob, 读取方式)
SOURCES = {
    "finemath": ("HuggingFaceTB/finemath/finemath-4plus/*.parquet", "web"),
    "fwedu": ("HuggingFaceTB/smollm-corpus/fineweb-edu-dedup/*.parquet", "web"),
    "omi2": ("nvidia/OpenMathInstruct-2/data/train_5M-*.parquet", "omi2"),
    "numina": ("project-numina/NuminaMath-CoT/data/train-*.parquet", "numina"),
    "metamath": ("meta-math/MetaMathQA/*.json", "metamath"),
    "orca": ("microsoft/orca-math-word-problems-200k/data/*.parquet", "orca"),
    "gsm8k_train": ("openai/gsm8k/main/train-*.parquet", "gsm8k"),
    "math_train": ("EleutherAI/hendrycks_math/*/train-*.parquet", "math"),
}
CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{")
DEC = None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest(), "little")


def last_boxed(s):
    """取最后一个 \\boxed{...} 的内容（处理嵌套括号）。"""
    i = s.rfind("\\boxed{")
    if i < 0:
        return ""
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1] if depth == 0 else ""


class MathDecontam:
    def __init__(self):
        probs, texts = [], []
        for f in glob.glob(str(RAW / "openai/gsm8k/main/test-*.parquet")):
            t = pq.read_table(f).to_pydict()
            probs += t["question"]
            texts += list(t["question"])
        for f in glob.glob(str(RAW / "EleutherAI/hendrycks_math/*/test-*.parquet")):
            t = pq.read_table(f).to_pydict()
            probs += t["problem"]
            texts += list(t["problem"])
        m500 = RAW / "HuggingFaceH4/MATH-500/test.jsonl"
        if m500.exists():
            for l in open(m500, encoding="utf-8"):
                r = json.loads(l)
                probs.append(r["problem"])
                texts.append(r["problem"])
        for name in ("HumanEvalPlus.jsonl", "MbppPlus.jsonl"):
            p = EVAL / name
            if p.exists():
                for l in open(p, encoding="utf-8"):
                    r = json.loads(l)
                    texts.append(r.get("prompt", "") + "\n" + r.get("canonical_solution", ""))
        self.n_probs = len(probs)
        self.exact = {norm(p) for p in probs if len(norm(p)) >= 20}
        self.prefix = {norm(p)[:60] for p in probs if len(norm(p)) >= 60}
        self.grams = np.unique(np.concatenate([ngram_hashes(word_ids(t), 13) for t in texts]))

    def hit(self, problem, full):
        n = norm(problem)
        if n in self.exact or (len(n) >= 60 and n[:60] in self.prefix):
            return "exact"
        g = ngram_hashes(word_ids(full), 13)
        if g.size:
            idx = np.searchsorted(self.grams, g)
            idx[idx >= self.grams.size] = 0
            if (self.grams[idx] == g).any():
                return "ngram"
        return None


def init():
    global DEC
    import warnings
    warnings.filterwarnings("ignore")
    DEC = MathDecontam()


COLS = {"web": ["text"], "omi2": ["problem", "generated_solution", "expected_answer"],
        "numina": ["problem", "solution"], "orca": ["question", "answer"], "gsm8k": ["question", "answer"],
        "math": ["problem", "solution"]}


def read_rows(path, kind):
    """逐批产出 list[(problem, solution, answer)]（流式读，避免大文件一次进内存）；网页类 problem 为空。"""
    if kind == "metamath":
        rows = json.load(open(path, encoding="utf-8"))
        for k in range(0, len(rows), 20000):
            yield [(r["query"], r["response"], r["response"].rsplit("The answer is:", 1)[-1].strip()
                    if "The answer is:" in r["response"] else "") for r in rows[k:k + 20000]]
        return
    for b in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=COLS[kind]):
        d = b.to_pydict()
        if kind == "web":
            yield [("", x, "") for x in d["text"]]
        elif kind == "omi2":
            yield list(zip(d["problem"], d["generated_solution"], d["expected_answer"]))
        elif kind in ("numina", "math"):
            yield [(p, s, last_boxed(s)) for p, s in zip(d["problem"], d["solution"])]
        elif kind == "orca":
            yield [(q, a, "") for q, a in zip(d["question"], d["answer"])]
        elif kind == "gsm8k":
            out = []
            for q, a in zip(d["question"], d["answer"]):
                sol, _, ans = CALC.sub("", a).rpartition("####")
                out.append((q, sol.strip() + "\nThe answer is " + ans.strip() + ".", ans.strip()))
            yield out


def file_stem(path):
    """同一源下文件名可能重复（hendrycks_math 每科都叫 train-00000-of-00001），带上上级目录名。"""
    p = Path(path)
    return f"{p.parent.name}__{p.stem}"


def work(job):
    src, path, kind = job
    stem = file_stem(path)
    outp = OUT / src / (stem + ".parquet")
    if outp.exists() and (OUT / src / (stem + ".stats.json")).exists():
        return src, json.load(open(OUT / src / (stem + ".stats.json"), encoding="utf-8"))
    st = defaultdict(int)
    (OUT / src).mkdir(parents=True, exist_ok=True)
    schema = pa.schema([(k, pa.uint64() if k == "h" else pa.string()) for k in ("text", "problem", "solution", "answer", "h")])
    tmp = str(outp) + ".tmp"
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    for batch in read_rows(path, kind):
        cols = {"text": [], "problem": [], "solution": [], "answer": [], "h": []}
        for prob, sol, ans in batch:
            st["in"] += 1
            prob, sol = (prob or "").strip(), (sol or "").strip()
            full = (prob + "\n\n" + sol) if prob else sol
            if kind == "web":
                if not (300 <= len(full) <= 60_000):
                    st["drop_len"] += 1
                    continue
            elif len(prob) < 10 or len(sol) < 10 or len(full) > 12_000:
                st["drop_len"] += 1
                continue
            why = DEC.hit(prob or full[:400], full)
            if why:
                st["drop_" + why] += 1
                continue
            cols["text"].append(full)
            cols["problem"].append(prob)
            cols["solution"].append(sol)
            cols["answer"].append(str(ans or ""))
            cols["h"].append(h64(norm(prob)) if prob else h64(full))
            st["kept"] += 1
            st["chars"] += len(full)
        writer.write_table(pa.table(cols, schema=schema))
    writer.close()
    os.replace(tmp, outp)
    json.dump(dict(st), open(OUT / src / (stem + ".stats.json"), "w", encoding="utf-8"))
    return src, dict(st)


def global_dedup(per_problem=2):
    """跨文件：同一题最多保留 per_problem 份；网页按全文哈希只留 1 份。源的先后顺序决定优先级。"""
    seen = defaultdict(int)
    stats = {}
    order = ["gsm8k_train", "math_train", "numina", "orca", "metamath", "omi2", "finemath", "fwedu"]
    for src in order:
        files = sorted(glob.glob(str(OUT / src / "*.parquet")))
        cap = 1 if SOURCES[src][1] == "web" else per_problem
        kept = tot = 0
        for f in files:
            h = pq.read_table(f, columns=["h"])["h"].to_numpy()
            keep = np.zeros(len(h), dtype=bool)
            for i, x in enumerate(h.tolist()):
                if seen[x] < cap:
                    seen[x] += 1
                    keep[i] = True
            (OUT / "_keep" / src).mkdir(parents=True, exist_ok=True)
            np.save(OUT / "_keep" / src / (Path(f).stem + ".npy"), keep)  # f 已是 file_stem 命名
            kept += int(keep.sum())
            tot += len(h)
        stats[src] = {"清洗后": tot, "去重后": kept, "重复剔除": tot - kept}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for src, (g, kind) in SOURCES.items():
        if args.only and src not in args.only.split(","):
            continue
        for p in sorted(glob.glob(str(RAW / g))):
            jobs.append((src, p, kind))
    dec = MathDecontam()
    print(f"去污染参照：{dec.n_probs} 道评测题，{dec.grams.size:,} 个 13-gram", flush=True)
    del dec
    tot = defaultdict(lambda: defaultdict(int))
    with Pool(args.workers, initializer=init) as pool:
        for src, st in pool.imap_unordered(work, jobs):
            for k, v in st.items():
                tot[src][k] += v
            print(f"  {src}: 累计 {tot[src]['kept']:,} 条，{tot[src]['chars'] / 1e6:,.0f}M 字符", flush=True)
    dd = global_dedup()
    report = {}
    for src in tot:
        s = dict(tot[src])
        s.update(dd.get(src, {}))
        s["估计token(÷3.6)"] = int(s.get("chars", 0) / 3.6 * s.get("去重后", 0) / max(1, s.get("清洗后", 1)))
        report[src] = s
    json.dump(report, open(OUT / "_math_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
