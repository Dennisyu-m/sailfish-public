"""
和模型对话（有显卡时用显卡）。注意：这不是聊天模型，它只会"按题目写 Python 函数"，而且只学过英文题目。
两种输入：
  1. 英文题目（一句或几句），模型回一段 ```python 代码```（精调格式 A）
  2. 以 def 开头的函数头 + 文档串（可多行，空行结束），模型补全函数体（HumanEval 格式）
命令：/t 0.8 设采样温度（0 = 贪心）；/q 退出
用法：python scripts/chat.py [--ckpt weights/v1/model.pt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import nanocode_lib as L  # noqa: E402
from eval_humaneval import STOPS, truncate  # noqa: E402


@torch.no_grad()
def generate(m, prompt, max_new, temp, stop_fn):
    return L.greedy_generate(m, [prompt], min(max_new, m.max_len - len(prompt) - 1), stop_fn=stop_fn, temperature=temp)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "weights" / "v3" / "model.pt"))
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tok = Tokenizer.from_file(args.tokenizer or str(Path(args.ckpt).parent / "tokenizer.json"))
    cfgp = Path(args.ckpt).parent / "model_cfg.json"
    m = L.ScratchNanoCode10K(max_len=1024, **(json.load(open(cfgp)) if cfgp.exists() else {}))
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model", sd))
    dev = torch.device(args.device)
    m = m.to(device=dev, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
    temp = 0.0
    print("模型已加载。输入英文题目，或以 def 开头的函数头（空行结束）。/t 0.8 改温度，/q 退出。")
    while True:
        try:
            line = input("\n你> ")
        except EOFError:
            break
        if line.strip() == "/q":
            break
        if line.startswith("/t"):
            temp = float(line.split()[1]) if len(line.split()) > 1 else 0.0
            print(f"温度 = {temp}")
            continue
        if not line.strip():
            continue
        if line.lstrip().startswith(("def ", "from ", "import ")):
            lines = [line]
            while True:
                l2 = input("... ")
                if not l2.strip():
                    break
                lines.append(l2)
            text = "\n".join(lines) + "\n"
            stop = lambda ids: any(s in tok.decode(ids) for s in STOPS)  # noqa: E731
            mode = "B"
        else:
            text = line.strip() + "\n\n"
            stop = lambda ids: tok.decode(ids).count("```") >= 2  # noqa: E731
            mode = "A"
        prompt = [L.EOT, L.EOT] + tok.encode(text).ids
        t0 = time.time()
        out = generate(m, prompt, args.max_new, temp, stop)
        dt = time.time() - t0
        if not out:
            print("\n模型>（它直接输出了结束符，相当于不回答。它只学过 Python 和规整的英文编程题，"
                  "试试 'Write a function that ...' 这种写法。）")
            continue
        reply = tok.decode(out)
        if mode == "B":
            reply = text + truncate(reply)
        print("\n模型>\n" + reply.rstrip())
        print(f"（{len(out)} 词，{dt:.1f} 秒，{len(out) / max(dt, 1e-6):.0f} 词/秒，{args.device}）")


if __name__ == "__main__":
    main()
