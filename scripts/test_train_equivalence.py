"""
CPU 测试：训练用实现与推理路径（benchmark_nanocode_10k 原样）逐项一致（FP32）。
  1. backbone_train 的 logits == reference_logits（整段重算）
  2. draft_train 的草稿 logits == model.draft()（在真实 DecodeState 上）
  3. greedy_generate（左填充批量）== spec_generate（prefill + forward_cycle）
  4. 训练几步 loss 下降（Muon + AdamW），无 NaN
运行：venv/Scripts/python scripts/test_train_equivalence.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nanocode_lib as L  # noqa: E402
from test_nanocode_correctness import make_model  # noqa: E402


def check(name, a, b, tol=1e-4):
    err = (a - b).abs().max().item()
    print(f"  {name}: 最大误差 {err:.2e}")
    assert err < tol, f"{name} 不一致"


def main(arch):
    print(f"=== arch={arch} ===")
    torch.manual_seed(0)
    m = make_model(0, arch)
    V = m.vocab_size

    # 1. 训练前向 == 参考前向
    ids = torch.randint(1, V, (3, 40))
    prev = torch.cat([torch.zeros(3, 1, dtype=torch.long), ids[:, :-1]], dim=1)
    h = L.backbone_train(m, ids, prev)
    for b in range(3):
        check(f"主干 logits 第{b}条", m.lm_head(h[b:b + 1]), m.reference_logits(ids[b:b + 1]))

    # 2. 草稿头批量训练前向 == model.draft
    h, kv = L.backbone_train(m, ids, prev, return_kv=True)
    anchors = torch.tensor([[5, 17, 30], [1, 9, 32], [12, 20, 25]])
    d_rej = torch.randint(0, V, (3, 3))
    use_delta = torch.tensor([[True, False, True], [False, True, True], [True, True, False]])
    logits = L.draft_train(m, h, kv, ids, anchors, d_rej, use_delta)
    for b in range(3):
        for j in range(3):
            p = anchors[b, j].item()
            st = L.DecodeState(m, "cpu", torch.float32)
            m.backbone(ids[b:b + 1, :p], prev[b:b + 1, :p], torch.arange(p), st)  # 写入主干 KV
            st.pos.fill_(p)
            v = ids[b:b + 1, p:p + 1]
            feat = h[b:b + 1, p - 1:p]
            delta = (m.embed(v) - m.embed(d_rej[b:b + 1, j:j + 1])) * float(use_delta[b, j])
            # model.draft 只返回 argmax，这里重放它的计算拿 logits
            ref_tok = m.draft(st, v, feat, delta)
            check(f"草稿 argmax b{b} p{p}", logits[b, j].argmax(-1).float(), ref_tok[0].float(), tol=0.5)
    # logits 精确比对：临时替换 argmax 抓取 lm_head 输出
    captured = {}
    orig = m.lm_head.forward

    def hook(x):
        out = orig(x)
        captured["lg"] = out
        return out

    b, j = 1, 1
    p = anchors[b, j].item()
    st = L.DecodeState(m, "cpu", torch.float32)
    m.backbone(ids[b:b + 1, :p], prev[b:b + 1, :p], torch.arange(p), st)
    st.pos.fill_(p)
    m.lm_head.forward = hook
    m.draft(st, ids[b:b + 1, p:p + 1], h[b:b + 1, p - 1:p],
            (m.embed(ids[b:b + 1, p:p + 1]) - m.embed(d_rej[b:b + 1, j:j + 1])) * float(use_delta[b, j]))
    m.lm_head.forward = orig
    check("草稿 logits", logits[b, j], captured["lg"][0])

    # 3. 批量贪心生成 == 投机解码
    prompts = [[0] + torch.randint(1, V, (n,)).tolist() for n in (7, 23, 15, 31)]
    outs = L.greedy_generate(m, prompts, max_new=60)
    for i, p in enumerate(prompts):
        so, cyc = L.spec_generate(m, p, max_new=60)
        assert so == outs[i][:len(so)] and len(so) == len(outs[i]), f"第 {i} 条生成不一致\n{so}\n{outs[i]}"
        print(f"  生成一致 第{i}条：{len(so)} 词 / {cyc} 轮")

    # 4. 小训练：loss 下降
    m2 = make_model(1, arch)
    L.init_for_training(m2)
    m2.train()
    muon_p, adam_p = L.param_groups(m2)
    opt1 = L.Muon(muon_p, lr=0.02)
    opt2 = torch.optim.AdamW(adam_p, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    data = torch.randint(1, 50, (8, 66))
    data[:, 1::2] = (data[:, 0::2] * 7) % 50  # 可学习的规律
    losses = []
    for step in range(30):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = L.lm_loss(m2, data)
        loss.backward()
        opt1.step(); opt2.step()
        opt1.zero_grad(set_to_none=True); opt2.zero_grad(set_to_none=True)
        losses.append(loss.item())
    print(f"  训练 loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0] - 1.0 and all(l == l for l in losses)
    print("全部通过")


if __name__ == "__main__":
    main("v1")
    main("v3")
