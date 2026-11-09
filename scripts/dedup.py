"""
第 2 遍（全局）：只读 data/clean 的元数据列（不读正文）
  1. 防污染：函数名+参数命中 -> 剔除；13-gram 命中 -> 剔除
  2. 精确去重（空白归一后的哈希）
  3. MinHash 近似去重（8 带 x 8 行，标签传播求连通块，每块留优先级最高的一条）
  4. 从 OpenCodeInstruct 挑 4 万道精调题（§4.3，按 domain 分层抽样），从预训练数据里扣掉
输出：data/clean/_keep/<src>/<文件>.npy（bool：进预训练）、data/clean/_sft_index.json、data/clean/_dedup_stats.json
保留优先级：OpenCodeInstruct（测试分高的优先）> sft2 > Magicoder > annealing > cosmopedia。
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CLEAN = Path(os.environ.get("NANO_CLEAN", ROOT / "data" / "clean"))
SRC_RANK = {"oci": 0, "sft2": 1, "mago": 2, "mage": 3, "ann": 4, "cosmo": 5}
NB = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sft", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(str(CLEAN / "*" / "*.parquet")) if "_keep" not in f)
    metas, offs = [], [0]
    exact, bands, nh, gh, src, score, sft_ok, domain = [], [], [], [], [], [], [], []
    for f in files:
        s = Path(f).parent.name
        cols = ["exact", "name_hit", "ngram_hit"] + [f"b{i}" for i in range(NB)]
        has = pq.ParquetFile(f).schema_arrow.names
        extra = [c for c in ("score", "sft_ok", "domain") if c in has]
        t = pq.read_table(f, columns=cols + extra)
        n = t.num_rows
        exact.append(t["exact"].to_numpy())
        bands.append(np.stack([t[f"b{i}"].to_numpy() for i in range(NB)], axis=1))
        nh.append(t["name_hit"].to_numpy(zero_copy_only=False))
        gh.append(t["ngram_hit"].to_numpy(zero_copy_only=False))
        src.append(np.full(n, SRC_RANK[s], dtype=np.int8))
        score.append(t["score"].to_numpy() if "score" in extra else np.zeros(n))
        sft_ok.append(t["sft_ok"].to_numpy(zero_copy_only=False) if "sft_ok" in extra else np.zeros(n, bool))
        domain.append(np.array(t["domain"].to_pylist()) if "domain" in extra else np.full(n, ""))
        metas.append((f, s, n))
        offs.append(offs[-1] + n)
    exact, bands = np.concatenate(exact), np.concatenate(bands)
    nh, gh = np.concatenate(nh).astype(bool), np.concatenate(gh).astype(bool)
    src, score = np.concatenate(src), np.concatenate(score)
    sft_ok, domain = np.concatenate(sft_ok).astype(bool), np.concatenate(domain)
    N = len(exact)
    print(f"文档总数 {N:,}", flush=True)

    # 优先级排序：rank[i] 越小越优先
    gidx = np.arange(N)
    order = np.lexsort((gidx, -score, src))
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)

    stats = {"by_src": {}}
    names = {v: k for k, v in SRC_RANK.items()}

    def count(mask):
        return {names[s]: int(((src == s) & mask).sum()) for s in np.unique(src)}

    stats["by_src"]["0_清洗后"] = count(np.ones(N, bool))
    alive = np.ones(N, bool)
    stats["by_src"]["1_函数名参数命中剔除"] = count(alive & nh)
    alive &= ~nh
    stats["by_src"]["2_13gram命中剔除"] = count(alive & gh)
    alive &= ~gh

    # 精确去重：同哈希留 rank 最小的
    idx = np.where(alive)[0]
    o = idx[np.lexsort((rank[idx], exact[idx]))]
    first = np.ones(len(o), bool)
    first[1:] = exact[o[1:]] != exact[o[:-1]]
    dup = np.zeros(N, bool)
    dup[o[~first]] = True
    stats["by_src"]["3_精确重复剔除"] = count(dup)
    alive &= ~dup

    # 近似去重：标签 = 连通块里最小 rank
    idx = np.where(alive)[0]
    lab = rank[idx].copy()
    for it in range(20):
        changed = 0
        for b in range(NB):
            v = bands[idx, b]
            o = np.argsort(v, kind="stable")
            sv = v[o]
            starts = np.r_[0, np.where(sv[1:] != sv[:-1])[0] + 1]
            mins = np.minimum.reduceat(lab[o], starts)
            lens = np.diff(np.r_[starts, len(o)])
            new = np.repeat(mins, lens)
            ch = new < lab[o]
            changed += int(ch.sum())
            lab[o] = np.minimum(lab[o], new)
        print(f"  近似去重第 {it + 1} 轮，更新 {changed:,}", flush=True)
        if changed == 0:
            break
    near = np.zeros(N, bool)
    near[idx[lab != rank[idx]]] = True
    stats["by_src"]["4_近似重复剔除"] = count(near)
    alive &= ~near
    stats["by_src"]["5_去重后"] = count(alive)

    # 精调题：OCI、sft_ok、存活；按 domain 分层
    rng = np.random.default_rng(args.seed)
    cand = np.where(alive & sft_ok & (src == SRC_RANK["oci"]))[0]
    doms = np.unique(domain[cand])
    per = args.n_sft // len(doms)
    chosen = []
    for k, d in enumerate(doms):
        c = cand[domain[cand] == d]
        want = per if k < len(doms) - 1 else args.n_sft - len(chosen)
        chosen.extend(rng.choice(c, size=min(want, len(c)), replace=False).tolist())
    chosen = np.array(sorted(chosen))
    stats["sft"] = {"候选": int(len(cand)), "选中": int(len(chosen)),
                    "按domain": {d: int((domain[chosen] == d).sum()) for d in doms}}
    alive[chosen] = False
    stats["by_src"]["6_扣掉精调题后进预训练"] = count(alive)

    keep_root = CLEAN / "_keep"
    sft_index = []
    for k, (f, s, n) in enumerate(metas):
        a, b = offs[k], offs[k + 1]
        d = keep_root / s
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / (Path(f).stem + ".npy"), alive[a:b])
        rows = chosen[(chosen >= a) & (chosen < b)] - a
        if len(rows):
            sft_index.append({"file": f, "rows": rows.tolist()})
    json.dump(sft_index, open(CLEAN / "_sft_index.json", "w"))
    json.dump(stats, open(CLEAN / "_dedup_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
