"""
RFT 采样：约 2 万题各采样 8 次，只留做对的，写成精调数据 <run>/rft.jsonl。
  数学：sft_math.jsonl + ds_sft.jsonl 里带 answer 的文字解题，用 eval_math 的规则比答案。
  编程：data/gen/code_*_ok.jsonl，用原始 code_*.jsonl 里的 tests 在子进程里跑（超时 5 秒）。
多卡并行采样，每张卡写 <run>/rft_parts/part<rank>.jsonl，中断后接着采；判分最后统一做。
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
BAD_CODE = ("input(", "open(", "import os", "subprocess", "random", "sys.", "shutil", "socket")


def fn_name(s):
    m = re.search(r"def\s+(\w+)\s*\(", s)
    return m.group(1) if m else None


def load_problems(data_dir, n_math, n_code, seed):
    data_dir = Path(data_dir)
    rng = random.Random(seed)
    math = []
    for fn in ("sft_math.jsonl", "ds_sft.jsonl"):
        for l in open(data_dir / fn, encoding="utf-8"):
            r = json.loads(l)
            if r.get("answer") and "```" not in r["completion"] and "print(" not in r["completion"]:
                math.append({"kind": "math", "prompt": r["prompt"], "answer": str(r["answer"]), "src": r.get("src", "")})
    tests = {}
    for fn in ("code_off.jsonl", "code_low.jsonl"):
        for l in open(ROOT / "data" / "gen" / fn, encoding="utf-8"):
            r = json.loads(l)
            t = r.get("tests")
            if isinstance(t, str):
                try:
                    t = json.loads(t)
                except json.JSONDecodeError:
                    continue
            name = fn_name(str(r.get("signature", "")))
            if name and isinstance(t, list) and len(t) >= 3:
                tests[name] = t
    code = []
    for fn in ("code_off_ok.jsonl", "code_low_ok.jsonl"):
        for l in open(ROOT / "data" / "gen" / fn, encoding="utf-8"):
            r = json.loads(l)
            name = fn_name(r["prompt"])
            if name in tests:
                code.append({"kind": "code", "prompt": r["prompt"], "tests": tests[name], "src": r.get("src", "")})
    rng.shuffle(math)
    rng.shuffle(code)
    probs = math[:n_math] + code[:n_code]
    for i, p in enumerate(probs):
        p["id"] = i
    return probs


def worker(rank, a, probs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import nanocode_lib as L
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.data_dir) / "tokenizer.json"))
    run = Path(a.run)
    cfg = json.load(open(run / "model_cfg.json"))
    dev = torch.device("cuda:0")
    m = L.ScratchNanoCode10K(max_len=1024, **cfg)
    sd = torch.load(a.init, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model", sd))
    m = m.to(device=dev, dtype=torch.bfloat16).eval()
    torch.manual_seed(a.seed * 100 + rank + int.from_bytes(os.urandom(4), "little"))  # 续跑不能和上次采出一样的
    path = run / "rft_parts" / f"part{rank}.jsonl"
    done = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = done.get(r["id"], 0) + 1
    mine = [p for p in probs if p["id"] % a.gpus == rank]
    todo = []
    for p in mine:
        ids = [L.EOT, L.EOT] + tok.encode(p["prompt"]).ids
        if len(ids) + a.max_new <= 1024:
            todo += [(p, ids)] * max(0, a.samples - done.get(p["id"], 0))
    todo.sort(key=lambda t: len(t[1]))
    print(f"[卡 {rank}] {len(mine)} 题，本次 {len(todo)} 次采样", flush=True)
    with open(path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.bsz):
            part = todo[i:i + a.bsz]
            gens = L.greedy_generate(m, [x[1] for x in part], a.max_new, temperature=a.temperature)
            for (p, _), g in zip(part, gens):
                f.write(json.dumps({"id": p["id"], "text": tok.decode(g)}, ensure_ascii=False) + "\n")
            f.flush()
            if rank == 0 and (i // a.bsz) % 20 == 0:
                print(f"  卡 0：{i + len(part)}/{len(todo)}", flush=True)


def check_code(args):
    from check_gen import run_py
    p, text = args
    if any(w in text for w in BAD_CODE) or not text.strip():
        return False
    rc, _ = run_py(p["prompt"] + text + "\n\n" + "\n".join(p["tests"]) + "\nprint('PASS')\n")
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-math", type=int, default=12000)
    ap.add_argument("--n-code", type=int, default=8000)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    (run / "rft_parts").mkdir(parents=True, exist_ok=True)
    probs = load_problems(a.data_dir, a.n_math, a.n_code, a.seed)
    print(f"[RFT] 题目 {len(probs)}（数学 {sum(p['kind'] == 'math' for p in probs)}，编程 {sum(p['kind'] == 'code' for p in probs)}）", flush=True)
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(r, a, probs)) for r in range(a.gpus)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    if any(p.exitcode for p in ps):
        sys.exit(1)
    from eval_math import correct, extract
    by_id = {p["id"]: p for p in probs}
    samples = []
    for r in range(a.gpus):
        for l in open(run / "rft_parts" / f"part{r}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            samples.append((by_id[x["id"]], x["text"]))
    ok = [correct(extract(t, "math"), p["answer"], "math") if p["kind"] == "math" else None for p, t in samples]
    idx = [i for i, (p, _) in enumerate(samples) if p["kind"] == "code"]
    with ThreadPoolExecutor(32) as ex:
        for i, r in zip(idx, ex.map(check_code, [samples[i] for i in idx])):
            ok[i] = r
    seen, n_out = set(), 0
    with open(run / "rft.jsonl", "w", encoding="utf-8") as f:
        for (p, t), good in zip(samples, ok):
            key = (p["id"], t.strip())
            if good and key not in seen:
                seen.add(key)
                f.write(json.dumps({"src": "rft_" + p["kind"], "prompt": p["prompt"], "completion": t}, ensure_ascii=False) + "\n")
                n_out += 1
    stat = {}
    for kind in ("math", "code"):
        rows = [g for (p, _), g in zip(samples, ok) if p["kind"] == kind]
        solved = {p["id"] for (p, _), g in zip(samples, ok) if p["kind"] == kind and g}
        stat[kind] = {"samples": len(rows), "correct": sum(rows), "acc": sum(rows) / max(1, len(rows)), "problems_solved": len(solved)}
    stat["kept_unique"] = n_out
    json.dump(stat, open(run / "rft_stats.json", "w"), indent=1)
    print("[RFT]", json.dumps(stat, ensure_ascii=False), flush=True)
    if n_out == 0:
        sys.exit("没有任何做对的样本")


if __name__ == "__main__":
    main()
