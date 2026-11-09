"""
CPU 正确性测试：投机解码（验证 + 回滚 + 草稿）输出必须与 "逐词贪心、整段重算" 完全一致。

用小尺寸随机模型（FP32）：
  - 先用 reference_logits 整段重算得到贪心参考序列；
  - 再走 prefill + forward_cycle，草稿交替使用
      (a) 参考序列 + 在随机槽位故意改错一个词（覆盖 n = 1..8 的所有接受长度）；
      (b) 模型自己的草稿头输出（检查草稿头不会弄脏主干 KV / 卷积状态）。
  - 每轮检查：接受长度 n 与预期一致、新锚点与参考一致、pos 正确。

运行：python test_nanocode_correctness.py
"""
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_nanocode_10k import DecodeState, ScratchNanoCode10K  # noqa: E402


def make_model(seed=0, arch="v1"):
    torch.manual_seed(seed)
    m = ScratchNanoCode10K(d=128, d_ff=256, vocab_size=512, block_size=8, max_len=256, n_q=4, n_kv=2, arch=arch).eval()
    with torch.no_grad():  # 让默认为 0 的支路也参与计算，否则测不到
        m.bigram_scale.fill_(1.0)
        m.bigram_table.normal_(0, 0.5)
        if arch == "v1":
            m.unet_lambdas.normal_(0, 0.5)
        else:  # 让大多数 token 都算数字，数字槽位的各种前后文都测到
            m.digit2.weight.normal_(0, 0.5)
            m.digit3.weight.normal_(0, 0.5)
            m.digit_val[:] = torch.randint(-1, 10, (512,))
        m.mask_emb.normal_(0, 1.0)
        m.prefix_delta_up.weight.normal_(0, 0.05)
        m.draft_gate_up.weight.normal_(0, 0.05)
    m.set_bigram_vocab(torch.randperm(512)[:256])
    return m


def run(seed=0, P=24, N=160, arch="v1"):
    m = make_model(seed, arch)
    B, V = m.block_size, m.vocab_size
    prompt = torch.randint(0, V, (1, P))

    # 参考：逐词贪心，每步整段重算
    seq = prompt.clone()
    for _ in range(N + B + 1):
        nxt = m.reference_logits(seq)[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    ref = seq[0, P:].tolist()

    st = DecodeState(m, "cpu", torch.float32)
    m.debug = True
    m.prefill(st, prompt)
    rng = random.Random(seed)
    out, ns, rounds, max_err = [], Counter(), 0, 0.0
    while len(out) < N:
        i = len(out)
        assert st.cand[0, 0].item() == ref[i], f"第 {rounds} 轮锚点不一致"
        assert st.pos.item() == P + i, "pos 不一致"
        if rounds % 3 == 2:   # (b) 用模型自己的草稿
            drafts = st.cand[0, 1:].tolist()
            exp = 1
            while exp < B and drafts[exp - 1] == ref[i + exp]:
                exp += 1
        else:                 # (a) 参考草稿，随机改错一个槽位（或不改）
            drafts = ref[i + 1:i + B]
            c = rng.choice(list(range(1, B)) + [None])
            if c is not None:
                drafts[c - 1] = (drafts[c - 1] + 1) % V
            exp = c if c is not None else B
            st.cand[0, 1:] = torch.tensor(drafts)
        cand = st.cand.clone()
        n = m.forward_cycle(st).item()
        assert n == exp, f"第 {rounds} 轮接受长度 {n}，预期 {exp}"
        # 数值检查：这一轮 8 个位置的验证 logits 必须与整段重算一致（能抓出卷积回滚、KV 位置等细小错误）
        full = torch.cat([prompt, torch.tensor([ref[:i]], dtype=torch.long), cand], dim=1)
        ref_logits = m.reference_logits(full)[:, -B:]
        err = (m.last_logits - ref_logits).abs().max().item()
        assert err < 1e-3, f"第 {rounds} 轮 logits 偏差 {err:.2e}"
        max_err = max(max_err, err)
        out.extend(ref[i:i + n])
        ns[n] += 1
        rounds += 1
    return rounds, ns, max_err


if __name__ == "__main__":
    for arch in ("v1", "v3"):
        for seed in (0, 1, 2):
            rounds, ns, max_err = run(seed, arch=arch)
            print(f"{arch} seed={seed}: {rounds} rounds match greedy; max logits err {max_err:.1e}; "
                  f"accept-length histogram {dict(sorted(ns.items()))}")
    print("PASS")
