"""
第 1 遍（逐文件并行）：读原始数据 -> 筛 Python / 语法检查 -> 统一成文本 -> 算防污染标记和去重指纹
-> data/clean/<src>/<文件名>.parquet，每个文件一份统计 data/clean/<src>/<文件名>.stats.json。
已处理过的文件跳过（可反复运行，边下载边处理）；只处理大小已与 filelist 一致的文件。

用法：venv/Scripts/python scripts/prep_data.py [--workers 7] [--limit-files N] [--max-rows N]
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datautil as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = Path(os.environ.get("NANO_CLEAN", ROOT / "data" / "clean"))
DEC = None

CS_KW = re.compile(r"\b(program|programming|python|computer|computing|algorithm|software|data structure|coding|code|"
                   r"machine learning|database|compiler|operating system|computation|data science)", re.I)
TITLE = re.compile(r'textbook on "([^"]+)"')


def expected_sizes():
    sizes = {}
    for fl in (RAW / "filelist.tsv", RAW / "filelist_cosmo.txt"):
        if fl.exists():
            for line in open(fl, encoding="utf-8"):
                parts = line.strip().replace("\t", " ").split(" ")
                if len(parts) == 3:
                    sizes[str(RAW / parts[0] / parts[1])] = int(parts[2])
    return sizes


def src_of(path: str) -> str:
    p = path.replace("\\", "/")
    if "OpenCodeInstruct" in p:
        return "oci"
    if "opc-annealing" in p:
        return "ann"
    if "opc-sft-stage2" in p:
        return "sft2"
    if "Magicoder-OSS" in p:
        return "mago"
    if "Magicoder-Evol" in p:
        return "mage"
    if "cosmopedia" in p:
        return "cosmo"
    raise ValueError(p)


# ---------------------------------------------------------------- 各数据源读取：产出 (text, extra) 或 (None, 丢弃原因)
def iter_rows(path):
    if path.endswith(".parquet"):
        f = pq.ParquetFile(path)
        for i in range(f.num_row_groups):
            for r in f.read_row_group(i).to_pylist():
                yield r
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except Exception:  # noqa: BLE001
                    yield {"__bad__": True}


def first(r, keys):
    for k in keys:
        if r.get(k):
            return r[k]
    return None


def oci_sft_check(code: str):
    """§4.3 的 2、3 条：写一个 Python 函数（顶层有 def、没有 class、不读标准输入），代码 5~60 行。"""
    lines = [l for l in code.strip().split("\n") if l.strip()]
    if not (5 <= len(lines) <= 60):
        return False
    if "input(" in code or "sys.stdin" in code:
        return False
    tree = ast.parse(code)
    has_def = any(isinstance(n, ast.FunctionDef) for n in tree.body)
    has_cls = any(isinstance(n, ast.ClassDef) for n in tree.body)
    return has_def and not has_cls


def convert(src, r):
    if "__bad__" in r:
        return None, "bad_json", {}
    if src == "oci":
        inp, out = (r.get("input") or "").strip(), (r.get("output") or "").strip()
        ok, blocks = D.markdown_python_ok(out)
        if not ok:
            return None, "not_python_or_syntax", {}
        score = float(r.get("average_test_score") or 0.0)
        extra = {"score": score, "domain": r.get("domain") or ""}
        sft = score == 1.0 and len(blocks) == 1 and oci_sft_check(blocks[0])
        extra["sft_ok"] = sft
        extra["prompt"] = inp if sft else ""
        extra["code"] = blocks[0] if sft else ""
        return inp + "\n\n" + out, None, extra
    if src == "ann":
        lang = (r.get("lang") or r.get("file_ext") or "").lower()
        if lang not in ("python", "py"):
            return None, "not_python", {}
        text = r.get("text") or ""
        if "```" in text:
            ok, _ = D.markdown_python_ok(text)
        else:
            ok = D.parses(text)
        if not ok:
            return None, "syntax", {}
        return text, None, {"domain": r.get("type") or ""}
    if src in ("sft2", "mago", "mage"):
        q = first(r, ["instruction", "problem", "prompt", "question"])
        a = first(r, ["output", "response", "solution", "answer", "code"])
        if not q or not a:
            return None, "missing_field", {}
        lang = (r.get("lang") or r.get("language") or "python").lower()
        if lang not in ("python", "py"):
            return None, "not_python", {}
        if "```" in a:
            ok, _ = D.markdown_python_ok(a)
            if not ok:
                return None, "not_python_or_syntax", {}
            text = q.strip() + "\n\n" + a.strip()
        else:
            if not D.parses(a):
                return None, "not_python_or_syntax", {}
            text = q.strip() + "\n\n```python\n" + a.strip() + "\n```"
        return text, None, {}
    if src == "cosmo":
        m = TITLE.search(r.get("prompt") or "")
        title = m.group(1) if m else (r.get("prompt") or "")[:200]
        return (r.get("text") or "").strip(), None, {"cs": bool(CS_KW.search(title))}
    raise ValueError(src)


def init_worker():
    global DEC
    import warnings
    warnings.filterwarnings("ignore")
    DEC = D.Decontam(ROOT / "data/eval/HumanEvalPlus.jsonl", ROOT / "data/eval/MbppPlus.jsonl")


def process(args):
    path, max_rows = args
    src = src_of(path)
    rel = os.path.relpath(path, RAW).replace("\\", "/")
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", rel.split("/", 2)[-1])
    out_dir = CLEAN / src
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (stem + ".parquet")
    if out.exists():
        return None
    stats = {"file": rel, "src": src, "read": 0, "kept": 0, "drop": {}, "name_hit": 0, "ngram_hit": 0, "chars": 0}
    cols = {k: [] for k in ["text", "exact", "name_hit", "ngram_hit"] + [f"b{i}" for i in range(D.N_BANDS)]}
    extras = {}
    for r in iter_rows(path):
        if max_rows and stats["read"] >= max_rows:
            break
        stats["read"] += 1
        text, why, extra = convert(src, r)
        if text is None:
            stats["drop"][why] = stats["drop"].get(why, 0) + 1
            continue
        if not text.strip() or len(text) > 200000:
            stats["drop"]["empty_or_huge"] = stats["drop"].get("empty_or_huge", 0) + 1
            continue
        w = D.word_ids(text)
        nh, gh = DEC.name_hit(text), DEC.ngram_hit(w)
        bands = D.minhash_bands(w)
        cols["text"].append(text)
        cols["exact"].append(D.exact_hash(text))
        cols["name_hit"].append(nh)
        cols["ngram_hit"].append(gh)
        for i in range(D.N_BANDS):
            cols[f"b{i}"].append(int(bands[i]))
        for k, v in extra.items():
            extras.setdefault(k, []).append(v)
        stats["kept"] += 1
        stats["name_hit"] += int(nh)
        stats["ngram_hit"] += int(gh)
        stats["chars"] += len(text)
    cols.update(extras)
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    os.replace(tmp, out)
    json.dump(stats, open(out.with_suffix(".stats.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--only", default="", help="只处理某个数据源，如 oci")
    args = ap.parse_args()
    sizes = expected_sizes()
    files = sorted(glob.glob(str(RAW / "**" / "*.parquet"), recursive=True) +
                   glob.glob(str(RAW / "**" / "*.jsonl"), recursive=True))
    ready = []
    for f in files:
        f = str(Path(f))
        if sizes.get(f) is not None and os.path.getsize(f) == sizes[f]:
            if not args.only or src_of(f) == args.only:
                ready.append(f)
    if args.limit_files:
        ready = ready[:args.limit_files]
    print(f"可处理文件 {len(ready)} / 已下载 {len(files)}", flush=True)
    with Pool(args.workers, initializer=init_worker) as pool:
        for s in pool.imap_unordered(process, [(f, args.max_rows) for f in ready]):
            if s:
                print(f"{s['src']:5s} {s['file'][-48:]:48s} 读 {s['read']:>8} 留 {s['kept']:>8} "
                      f"丢 {s['drop']} 名称命中 {s['name_hit']} 13gram命中 {s['ngram_hit']}", flush=True)


if __name__ == "__main__":
    main()
