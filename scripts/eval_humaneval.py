"""
HumanEval+ 评测 + 端到端测速。
  1. 每题：提示词前加 <|endoftext|>，用投机解码（prefill + forward_cycle，推理路径原样）贪心生成，
     遇到 <|endoftext|> 或停止串就截断；记录 生成词数 / 轮数（=实测 τ）/ 字符数。
  2. 正确性交叉校验：同一批题用批量逐词贪心（greedy_generate）再生成一遍，统计两边完全一致的题数
     （BF16 下不同形状的矩阵乘可能有极小数值差导致个别题分叉，所以报一致率而不是硬断言）。
  3. 写 samples.jsonl，调用 evalplus 算 HumanEval / HumanEval+ pass@1。
  4. 端到端测速（--speed）：CUDA Graph（可选 torch.compile），每 4 轮回放同步一次 CPU 检查停止条件，
     逐题计时：报 TPS、字符/秒、每题耗时（只计停止前的有效词/字符，回放多跑的轮次时间照算）。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nanocode_lib as L  # noqa: E402
from benchmark_nanocode_10k import capture_graph  # noqa: E402

STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert ", "\n```"]


def stop_start(txt, after_def):
    """停止串的搜索起点。MBPP 提示词没有函数头，模型要自己写 def，所以从第一个 def 之后才开始找。"""
    if not after_def:
        return 0
    i = txt.find("def ")
    return len(txt) if i == -1 else i + 4


def make_stop(tok, after_def=False):
    def stop_fn(ids):
        if len(ids) % 4:
            return False
        txt = tok.decode(ids)
        k = stop_start(txt, after_def)
        return any(txt.find(s, k) != -1 for s in STOPS)
    return stop_fn


def truncate(txt, after_def=False):
    cut = len(txt)
    k = stop_start(txt, after_def)
    for s in STOPS:
        i = txt.find(s, k)
        if i != -1:
            cut = min(cut, i)
    return txt[:cut]


def load_model(args, dev, dtype):
    cfgp = Path(args.ckpt).parent / "model_cfg.json"  # v2 训练会写这个；v1 权重旁没有，走原来的默认
    if cfgp.exists():
        m = L.ScratchNanoCode10K(max_len=args.max_len, **json.load(open(cfgp)))
    elif args.small:
        m = L.ScratchNanoCode10K(d=128, d_ff=256, vocab_size=8192, max_len=args.max_len, n_q=4, n_kv=2)
    else:
        m = L.ScratchNanoCode10K(max_len=args.max_len)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model", sd))
    return m.to(device=dev, dtype=dtype).eval()


@torch.no_grad()
def speed_e2e(m, prompts, max_new, K, use_compile, dev):
    """CUDA Graph：K 轮一张图；每次回放后把 K 轮的候选与接受长度拷回 CPU 检查停止。"""
    B = m.block_size
    st = L.DecodeState(m, dev, m.embed.weight.dtype)
    m.prefill(st, torch.tensor([prompts[0][1]], device=dev))
    cycle = m.forward_cycle
    if use_compile:
        cycle = torch.compile(m.forward_cycle, mode="max-autotune-no-cudagraphs", dynamic=False)
        for _ in range(3):
            cycle(st)
    dev_buf = torch.zeros(K, B + 1, dtype=torch.long, device=dev)
    host_buf = torch.empty(K, B + 1, dtype=torch.long, pin_memory=True)

    def k_rounds():
        for k in range(K):
            dev_buf[k, :B].copy_(st.cand[0])
            n = cycle(st)
            dev_buf[k, B].copy_(n[0])

    snap = st.snapshot()
    g = capture_graph(k_rounds)
    st.restore(snap)
    res = []
    limit = m.max_len - (K + 1) * B
    for tid, p, stop_fn, tok in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        st.conv.zero_()
        m.prefill(st, torch.tensor([p], device=dev))
        out, cycles, done = [], 0, False
        while not done:
            g.replay()
            host_buf.copy_(dev_buf)
            hb = host_buf.tolist()
            for k in range(K):
                cycles += 1
                cand, n = hb[k][:B], hb[k][B]
                for t in cand[:n]:
                    if t == L.EOT:
                        done = True
                        break
                    out.append(t)
                if done:
                    break
            txt = tok.decode(out)
            if any(s in txt for s in STOPS) or len(out) >= max_new or int(st.pos.item()) > limit:
                done = True
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        txt = truncate(tok.decode(out))
        n_tok = len(tok.encode(txt).ids) if txt else 0
        res.append({"task_id": tid, "sec": dt, "tokens": n_tok, "chars": len(txt), "cycles": cycles})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--problems", required=True, help="HumanEvalPlus.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="跳过批量贪心交叉校验")
    ap.add_argument("--no-evalplus", action="store_true")
    ap.add_argument("--def-hint", action="store_true", help="诊断用，非标准评测：MBPP 提示词末尾补 def 函数名(")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--replay", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    dev = torch.device(args.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    if dev.type == "cpu":
        torch.set_num_threads(1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = load_model(args, dev, dtype)
    probs = [json.loads(l) for l in open(args.problems, encoding="utf-8")]
    if args.limit:
        probs = probs[:args.limit]
    after_def = args.dataset == "mbpp" and not args.def_hint
    if args.def_hint:
        for r in probs:
            r["prompt"] += f"def {r['entry_point']}("
    stop_fn = make_stop(tok, after_def)
    prompts = []
    for r in probs:
        p = [L.EOT] + tok.encode(r["prompt"]).ids
        mx = min(args.max_new, args.max_len - len(p) - 3 * m.block_size)
        prompts.append((r, p, mx))

    # 1. 投机解码生成（eager，逐题）
    samples, stats = [], []
    st = L.DecodeState(m, dev, dtype)
    t_all = time.time()
    for r, p, mx in prompts:
        t0 = time.time()
        ids, cyc = L.spec_generate(m, p, mx, stop_fn=stop_fn, st=st)
        dt = time.time() - t0
        comp = truncate(tok.decode(ids), after_def)
        samples.append({"task_id": r["task_id"], "solution": r["prompt"] + comp})
        stats.append({"task_id": r["task_id"], "gen_tokens": len(ids), "cycles": cyc, "sec_eager": dt,
                      "completion": comp})
    tau = sum(s["gen_tokens"] for s in stats) / max(1, sum(s["cycles"] for s in stats))
    print(f"[生成] {len(stats)} 题，用时 {time.time() - t_all:.1f}s（eager），实测 τ = {tau:.3f}（生成词数/轮数）")
    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    with open(out / "gen_stats.jsonl", "w", encoding="utf-8") as f:
        for s in stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = {"tau_measured": tau, "n": len(stats)}

    # 2. 交叉校验
    if not args.no_check:
        same = 0
        bs = 16
        for i in range(0, len(prompts), bs):
            part = prompts[i:i + bs]
            mx = min(x[2] for x in part)
            outs = L.greedy_generate(m, [x[1] for x in part], mx)
            for (r, p, _), o, s in zip(part, outs, stats[i:i + bs]):
                a = truncate(tok.decode(o), after_def)
                b = s["completion"]
                n = min(len(a), len(b))
                same += int(a[:n] == b[:n])
        summary["greedy_consistency"] = f"{same}/{len(prompts)}"
        print(f"[校验] 投机解码与逐词贪心一致：{same}/{len(prompts)} 题（按两者较短长度比较前缀）")

    # 3. evalplus
    if not args.no_evalplus:
        prob_path = Path(args.problems).resolve()
        if args.limit:  # evalplus 要求样本覆盖题库全部题目
            prob_path = out / "problems_limited.jsonl"
            with open(prob_path, "w", encoding="utf-8") as f:
                for r in probs:
                    f.write(json.dumps(r) + "\n")
        env = dict(os.environ, **{f"{args.dataset.upper()}_OVERRIDE_PATH": str(prob_path)})
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "evalplus_win.py"), "--dataset", args.dataset,
               "--samples", str(out / "samples.jsonl"), "--parallel", "8", "--i-just-wanna-run"]
        print("[evalplus] " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (out / "evalplus.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit(f"evalplus 失败，退出码 {r.returncode}")

    # 4. 端到端测速
    if args.speed:
        sp_prompts = [(r["task_id"], p, stop_fn, tok) for r, p, mx in prompts]
        for tag, comp in (("CUDA Graph", False), ("torch.compile + CUDA Graph", True)):
            if comp and not args.compile:
                continue
            res = speed_e2e(m, sp_prompts[:3], args.max_new, args.replay, comp, dev)  # 预热
            res = speed_e2e(m, sp_prompts, args.max_new, args.replay, comp, dev)
            T = sum(x["sec"] for x in res)
            tk = sum(x["tokens"] for x in res)
            ch = sum(x["chars"] for x in res)
            cy = sum(x["cycles"] for x in res)
            summary[tag] = {"TPS": tk / T, "chars_per_s": ch / T, "ms_per_problem": T / len(res) * 1000,
                            "ms_per_cycle": T / cy * 1000, "tokens": tk, "chars": ch}
            print(f"[测速 {tag}，{torch.cuda.get_device_name()}] {len(res)} 题：{tk / T:,.0f} TPS，"
                  f"{ch / T:,.0f} 字符/秒，每题 {T / len(res) * 1000:.1f} ms，每轮（含 prefill 均摊）{T / cy * 1000:.3f} ms")
            with open(out / f"speed_{'compile' if comp else 'graph'}.jsonl", "w") as f:
                for x in res:
                    f.write(json.dumps(x) + "\n")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
