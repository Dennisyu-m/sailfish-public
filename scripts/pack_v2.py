"""
v2 打包：用 tokenizer_v2（16384 词表、数字逐位切）把编程 + 数学 + 英文文档打成 uint16 .bin（每篇前加 <|endoftext|>）。
  编程：data/clean/<src>（第一轮清洗）+ data/clean/_keep；cosmo 只取 CS 类 + 少量其它
  数学/英文：data/clean2/<src> + data/clean2/_keep；若有 data/clean2/_sft_math_h.npy，这些题（精调/RL 题）不进预训练
输出 data/packed_v2/：<src>.bin、val_<src>.bin（每 200 篇抽 1 篇）、bigram_top256.npy、meta.json
用法：venv/Scripts/python scripts/pack_v2.py --workers 6
"""
import argparse
import glob
import json
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
CLEAN2 = ROOT / "data" / "clean2"
OUT = Path(os.environ.get("NANO_PACKED", ROOT / "data" / "packed_v2"))
CODE_SRCS = ["oci", "ann", "cosmo", "sft2", "mago"]
MATH_SRCS = ["finemath", "omi2", "numina", "metamath", "orca", "gsm8k_train", "math_train", "fwedu"]
COSMO_OTHER_FRAC = float(os.environ.get("NANO_COSMO_OTHER_FRAC", "0.05"))
FWEDU_FRAC = float(os.environ.get("NANO_FWEDU_FRAC", "1.0"))
VOCAB = 16384
TOK = None
EXCL = None


def init(tok_path):
    global TOK, EXCL
    TOK = Tokenizer.from_file(tok_path)
    p = CLEAN2 / "_sft_math_h.npy"
    EXCL = set(np.load(p).tolist()) if p.exists() else set()


def work(job):
    """流式：按 5000 行一批读 parquet、分词、直接追加写盘；不把整个文件留在内存里（fwedu 一个文件约 10 亿 token）。
    窗口在训练时随机抽，所以文件内不需要打乱。做完写 .stats.json，重跑时跳过已完成的文件。"""
    src, f = job
    stem = Path(f).stem
    tmp = OUT / "_parts" / src
    tmp.mkdir(parents=True, exist_ok=True)
    done = tmp / (stem + ".stats.json")
    if done.exists():
        st = json.load(open(done))
        return src, st["docs"], st["train"], st["val"], st["chars"], np.load(tmp / (stem + ".freq.npy"))
    base = CLEAN if src in CODE_SRCS else CLEAN2
    keep = np.load(base / "_keep" / src / (stem + ".npy"))
    cols = ["text"] + (["cs"] if src == "cosmo" else []) + (["h"] if src not in CODE_SRCS else [])
    r = random.Random(src + stem)
    freq = np.zeros(VOCAB, dtype=np.int64)
    nd = nt = nv = chars = 0
    row = 0
    with open(tmp / (stem + ".train.tmp"), "wb") as ft, open(tmp / (stem + ".val.tmp"), "wb") as fv:
        for b in pq.ParquetFile(f).iter_batches(batch_size=5000, columns=cols):
            d = b.to_pydict()
            texts = []
            for j, text in enumerate(d["text"]):
                i = row + j
                if not keep[i]:
                    continue
                if src == "cosmo" and not (d["cs"][j] or r.random() < COSMO_OTHER_FRAC):
                    continue
                if "h" in d and int(d["h"][j]) in EXCL:
                    continue
                if src == "fwedu" and FWEDU_FRAC < 1 and r.random() >= FWEDU_FRAC:
                    continue
                texts.append((i, text))
            row += len(d["text"])
            if not texts:
                continue
            tr, va = [], []
            for (i, text), e in zip(texts, TOK.encode_batch([t for _, t in texts])):
                arr = np.empty(len(e.ids) + 1, dtype=np.uint16)
                arr[0] = 0
                arr[1:] = e.ids
                (va if i % 200 == 0 else tr).append(arr)
                chars += len(text)
            nd += len(texts)
            if tr:
                t = np.concatenate(tr)
                t.tofile(ft)
                nt += t.size
                freq += np.bincount(t, minlength=VOCAB)
            if va:
                v = np.concatenate(va)
                v.tofile(fv)
                nv += v.size
    os.replace(tmp / (stem + ".train.tmp"), tmp / (stem + ".train"))
    os.replace(tmp / (stem + ".val.tmp"), tmp / (stem + ".val"))
    np.save(tmp / (stem + ".freq.npy"), freq)
    json.dump({"docs": nd, "train": int(nt), "val": int(nv), "chars": chars}, open(done, "w"))
    return src, nd, int(nt), int(nv), chars, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer_v2.json"))
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in CODE_SRCS:
        jobs += [(s, f) for f in sorted(glob.glob(str(CLEAN / s / "*.parquet")))]
    for s in MATH_SRCS:
        jobs += [(s, f) for f in sorted(glob.glob(str(CLEAN2 / s / "*.parquet")))]
    meta, freq = {}, np.zeros(VOCAB, dtype=np.int64)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with Pool(args.workers, initializer=init, initargs=(args.tokenizer,)) as pool:
        for src, nd, nt, nv, ch, cnt in pool.imap_unordered(work, jobs):
            m = meta.setdefault(src, {"docs": 0, "train_tokens": 0, "val_tokens": 0, "chars": 0})
            m["docs"] += nd
            m["train_tokens"] += nt
            m["val_tokens"] += nv
            m["chars"] += ch
            freq += cnt
            print(f"  {src}: 累计 {m['train_tokens'] / 1e6:.1f}M token", flush=True)
    for src in meta:
        for kind, ext in (("", ".train"), ("val_", ".val")):
            with open(OUT / f"{kind}{src}.bin", "wb") as fo:
                for p in sorted(glob.glob(str(OUT / "_parts" / src / f"*{ext}"))):
                    with open(p, "rb") as fi:
                        shutil.copyfileobj(fi, fo, 64 << 20)
        meta[src]["chars_per_token"] = meta[src]["chars"] / max(1, meta[src]["train_tokens"] + meta[src]["val_tokens"])
        meta[src]["domain"] = "code" if src in CODE_SRCS else ("english" if src == "fwedu" else "math")
    if not args.keep_parts:
        shutil.rmtree(OUT / "_parts", ignore_errors=True)
    freq[0] = -1
    np.save(OUT / "bigram_top256.npy", np.argsort(-freq)[:256])
    srcs = [k for k in meta if not k.startswith("_")]
    meta["_total_train_tokens"] = sum(meta[k]["train_tokens"] for k in srcs)
    for d in ("code", "math", "english"):
        meta[f"_{d}_train_tokens"] = sum(meta[k]["train_tokens"] for k in srcs if meta[k]["domain"] == d)
    meta["_vocab"] = VOCAB
    json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
