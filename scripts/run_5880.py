"""
Sailfish-v3 全流程（Windows）：
  pre -> sft -> rft_sample -> rft -> eval_math -> eval_sft -> gen -> draft -> eval -> eval_mbpp
每阶段成功后写 <run>/<阶段>.done，重跑跳过已完成阶段，未完成的从断点续；任一阶段失败立即停（退出码非 0）。

正式：python scripts/run_5880.py v3 --tokens 5.5e9
走通：python scripts/run_5880.py smoke --tokens 2e6 --small --sft-limit 400 --gen-limit 64 --eval-limit 8 --rft-n 40
可用 --only pre,sft 只跑部分阶段，--stop-after sft 跑到某阶段为止。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "scripts"
D = ROOT / "data" / "packed_v2"
SFT_FILES = "sft.jsonl,sft_math.jsonl,ds_sft.jsonl"
STAGES = ["pre", "sft", "rft_sample", "rft", "eval_math", "eval_sft", "gen", "draft", "eval", "eval_mbpp"]


def say(run, msg):
    line = f"=== {time.strftime('%F %T')} {msg} ==="
    print(line, flush=True)
    with open(run / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(run, name, cmd):
    """跑一个子进程，输出同时进控制台和 <run>/<阶段>.log；返回退出码。"""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(run / f"{name}.log", "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        return p.wait()


def limited_sft(data_dir, n):
    """走通用：每个题库只取前 n 行，放到临时目录（训练脚本读 data-dir 下的文件）。"""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in os.listdir(D):
        dst = out / name
        if name.endswith(".jsonl"):
            with open(D / name, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
                for i, l in enumerate(fi):
                    if i >= n:
                        break
                    fo.write(l)
        elif not dst.exists():
            os.link(D / name, dst)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--tokens", required=True, help="预训练 token 数，如 3e9")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--sft-gpus", type=int, default=0, help="精调用几张卡，0 = 同 --gpus")
    ap.add_argument("--small", action="store_true", help="走通用的小模型")
    ap.add_argument("--sft-limit", type=int, default=0, help="走通用：每个精调题库只取前 N 行")
    ap.add_argument("--gen-limit", type=int, default=60000)
    ap.add_argument("--eval-limit", type=int, default=0, help="走通用：评测只做前 N 题")
    ap.add_argument("--rft-n", type=int, default=0, help="走通用：RFT 数学/编程各取 N 题；0 = 12000/8000")
    ap.add_argument("--only", default="", help="逗号分隔，只跑这些阶段")
    ap.add_argument("--stop-after", default="", choices=[""] + STAGES)
    a = ap.parse_args()

    run = ROOT / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    data = limited_sft(run / "_data", a.sft_limit) if a.sft_limit else D
    py = sys.executable
    train = [py, str(S / "train.py"), "--mix", "v2", "--data-dir", str(data), "--out-dir", str(run), "--resume"]
    # n_kv=6：定稿写的 18/4 除不尽（GQA 要求 n_q 被 n_kv 整除），保持 d 和头维不变取 6（每组 3 个 query，同 v1 的 12/4）
    size = ["--small"] if a.small else ["--d", "1152", "--d-ff", "3072", "--n-q", "18", "--n-kv", "6"]
    train += size + ["--arch", "v3"]
    final = str(run / "rft_final.pt")
    rft_n = (["--n-math", str(a.rft_n), "--n-code", str(a.rft_n)] if a.rft_n else [])
    sft_gpus = str(a.sft_gpus or a.gpus)
    tok = str(D / "tokenizer.json")
    he = str(ROOT / "data" / "eval" / "HumanEvalPlus.jsonl")
    lim = ["--limit", str(a.eval_limit)] if a.eval_limit else []
    plan = {
        "pre": train + ["--stage", "pre", "--tokens", a.tokens, "--micro-bsz", "16", "--accum", "16", "--compile",
                        "--ckpt-min", "20", "--gpus", str(a.gpus)],
        "sft": train + ["--stage", "sft", "--init", str(run / "pre_final.pt"), "--sft-files", SFT_FILES, "--epochs", "2",
                        "--micro-bsz", "16", "--accum", "2", "--lr-muon", "0.005", "--lr-adam", "5e-4", "--warmup", "50",
                        "--decay-frac", "0.5", "--ckpt-min", "10", "--gpus", sft_gpus],
        "rft_sample": [py, str(S / "rft_sample.py"), "--run", str(run), "--init", str(run / "sft_final.pt"),
                       "--data-dir", str(data), "--gpus", str(a.gpus)] + rft_n,
        "rft": train + ["--stage", "rft", "--init", str(run / "sft_final.pt"), "--sft-files", str(run / "rft.jsonl"),
                        "--epochs", "1", "--micro-bsz", "16", "--accum", "2", "--lr-muon", "0.002", "--lr-adam", "2e-4",
                        "--warmup", "20", "--decay-frac", "0.5", "--ckpt-min", "10", "--gpus", sft_gpus],
        "eval_math": [py, str(S / "eval_math.py"), "--ckpt", final, "--tokenizer", tok,
                      "--out-dir", str(run / "eval_math")] + lim,
        "eval_sft": [py, str(S / "eval_humaneval.py"), "--ckpt", final, "--tokenizer", tok,
                     "--problems", he, "--out-dir", str(run / "eval_sft")] + lim + (["--small"] if a.small else []),
        "gen": train + ["--stage", "gen", "--init", final, "--sft-files", SFT_FILES, "--gen-len", "256",
                        "--gen-bsz", "128", "--gen-limit", str(a.gen_limit)],
        "draft": train + ["--stage", "draft", "--init", final, "--epochs", "3", "--micro-bsz", "32",
                          "--anchors", "32", "--lr-muon", "0.02", "--lr-adam", "3e-3", "--warmup", "100",
                          "--decay-frac", "0.3", "--ckpt-min", "10"],
        "eval": [py, str(S / "eval_humaneval.py"), "--ckpt", str(run / "draft_final.pt"), "--tokenizer", tok,
                 "--problems", he, "--out-dir", str(run / "eval"), "--speed", "--compile"] + lim
                + (["--small"] if a.small else []),
        "eval_mbpp": [py, str(S / "eval_humaneval.py"), "--ckpt", str(run / "draft_final.pt"), "--tokenizer", tok,
                      "--problems", str(ROOT / "data" / "eval" / "MbppPlus.jsonl"), "--dataset", "mbpp",
                      "--out-dir", str(run / "eval_mbpp")] + lim + (["--small"] if a.small else []),
    }
    only = set(x for x in a.only.split(",") if x)
    if only - set(STAGES):
        sys.exit(f"未知阶段：{only - set(STAGES)}")
    json.dump(vars(a), open(run / "pipeline_args.json", "w"), indent=1)
    for name in STAGES:
        if only and name not in only:
            continue
        done = run / f"{name}.done"
        if done.exists():
            say(run, f"跳过 {name}（已完成）")
        else:
            t0 = time.time()
            say(run, f"开始 {name}")
            code = run_cmd(run, name, plan[name])
            if code != 0:
                say(run, f"失败 {name} 退出码 {code}，流程停止")
                sys.exit(1)
            done.write_text(time.strftime("%F %T"), encoding="utf-8")
            (run / f"{name}_ckpt.pt").unlink(missing_ok=True)  # 阶段完成后断点没用了（每个约 3.4GB，D 盘只有 17GB）
            say(run, f"完成 {name}，用时 {(time.time() - t0) / 60:.1f} 分钟")
        if name == a.stop_after:
            say(run, f"--stop-after {name}：到此为止")
            return
    say(run, "流程跑完")


if __name__ == "__main__":
    main()
