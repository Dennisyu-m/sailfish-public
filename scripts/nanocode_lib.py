"""
训练用工具：复用 benchmark_nanocode_10k.ScratchNanoCode10K 的层，补上
  1. 支持 batch 的训练前向（不用 KV 缓存）            backbone_train / lm_loss
  2. Stage 3 草稿头的批量训练前向                      draft_train
  3. 批量贪心生成（左填充 + KV 缓存，用于 Stage 3 造数据和交叉校验）  greedy_generate
  4. Muon 优化器 + 参数分组
推理路径（prefill / forward_cycle / draft）一律不改；本文件的实现与它的等价性由
test_train_equivalence.py 在 CPU 上逐项验证。
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_nanocode_10k import DecodeState, ScratchNanoCode10K, apply_rope  # noqa: E402,F401

EOT = 0  # <|endoftext|>：文档分隔，同时当 BOS（推理时 bos_id=0）


# ---------------------------------------------------------------- 初始化
def init_for_training(m: ScratchNanoCode10K):
    """原始默认初始化不适合训练：Embedding N(0,1) 与 lm_head 绑定后 logits 过大；
    bigram_table 与 bigram_scale 同为 0 时两者梯度都恒为 0（永远学不动）。"""
    with torch.no_grad():
        nn.init.normal_(m.embed.weight, std=0.02)
        m.bigram_table.zero_()
        m.bigram_scale.fill_(1.0)
        for blk in list(m.conv_layers) + list(m.attn_layers) + list(m.draft_layers):
            for name in ("out_proj", "o_proj", "down"):
                if hasattr(blk, name):
                    nn.init.zeros_(getattr(blk, name).weight)
        m.mask_emb.normal_(0, 0.02)


# ---------------------------------------------------------------- 训练前向
def _rope_b(m, positions, dtype):
    """positions: [T] 或 [B,T] -> cos/sin 可与 [B,H,T,hd] 广播。"""
    cos, sin = m.rope_cos[positions].to(dtype), m.rope_sin[positions].to(dtype)
    if positions.dim() == 2:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return cos, sin


def _conv_u(blk, x):
    u, gate = blk.in_proj(blk.norm1(x)).chunk(2, dim=-1)
    return u * F.silu(gate)


def _conv_rest(blk, x, U):
    """U: [B, 2+T, d]（前 2 个是历史）。"""
    c = F.conv1d(U.transpose(1, 2), blk.dw_conv.weight.to(U.dtype), groups=U.size(-1)).transpose(1, 2)
    x = x + blk.out_proj(F.silu(c))
    g, up = blk.gate_up(blk.norm2(x)).chunk(2, dim=-1)
    return x + blk.down(F.silu(g) * up)


def conv_train(blk, x):
    u = _conv_u(blk, x)
    U = F.pad(u, (0, 0, 2, 0))  # 序列开头的卷积历史 = 0，与 DecodeState.conv 初值一致
    return _conv_rest(blk, x, U)


def _qkv(blk, x, cos, sin):
    B, T, _ = x.shape
    q, k, v = torch.split(blk.qkv(blk.norm1(x)),
                          [blk.n_q * blk.hd, blk.n_kv * blk.hd, blk.n_kv * blk.hd], dim=-1)
    q = q.view(B, T, blk.n_q, blk.hd).transpose(1, 2)
    k = k.view(B, T, blk.n_kv, blk.hd).transpose(1, 2)
    v = v.view(B, T, blk.n_kv, blk.hd).transpose(1, 2)
    q = apply_rope(blk.q_norm(q), cos, sin)
    k = apply_rope(blk.k_norm(k), cos, sin)
    return q, k, v


def _attn_out(blk, x, o):
    B, _, T, _ = o.shape
    x = x + blk.o_proj(o.transpose(1, 2).reshape(B, T, blk.d))
    g, up = blk.gate_up(blk.norm2(x)).chunk(2, dim=-1)
    return x + blk.down(F.silu(g) * up)


def _rep(blk, t):
    return t.repeat_interleave(blk.rep, dim=1)  # 第 h 个 query 头对应 KV 头 h // rep，与推理折叠一致


def attn_train(blk, x, cos, sin):
    q, k, v = _qkv(blk, x, cos, sin)
    o = F.scaled_dot_product_attention(q, _rep(blk, k), _rep(blk, v), is_causal=True)
    return _attn_out(blk, x, o), k, v


def backbone_train(m, ids, prev_ids, return_kv=False):
    """ids/prev_ids: [B,T]，位置从 0 开始。返回 final_norm 后的 h [B,T,d]（以及第 8/10 层 KV）。"""
    T = ids.size(1)
    x = m.input_emb(ids, prev_ids)
    cos, sin = _rope_b(m, torch.arange(T, device=ids.device), x.dtype)
    C, A = m.conv_layers, m.attn_layers
    kv, skips = {}, []

    def attn(i, x):
        x, k, v = attn_train(A[i], x, cos, sin)
        if return_kv and i >= 2:
            kv[i] = (k, v)
        return x

    if m.arch == "v3":
        h = m.final_norm(m.run_v3_layers(x, lambda i, y: conv_train(C[i], y), attn))
        return (h, kv) if return_kv else h
    lam = m.unet_lambdas
    x = conv_train(C[0], x); skips.append(x)          # L1
    x = attn(0, x); skips.append(x)                    # L2
    x = conv_train(C[1], x); skips.append(x)          # L3
    x = conv_train(C[2], x); skips.append(x)          # L4
    x = attn(1, x)                                     # L5
    x = conv_train(C[3], x)                            # L6
    x = conv_train(C[4], x + lam[0] * skips[3])       # L7
    x = attn(2, x + lam[1] * skips[2])                 # L8
    x = conv_train(C[5], x + lam[2] * skips[1])       # L9
    x = attn(3, x + lam[3] * skips[0])                 # L10
    h = m.final_norm(x)
    return (h, kv) if return_kv else h


def chunked_ce(m, h, targets, mask=None, chunk=4096):
    """分块算 lm_head + 交叉熵，省显存。targets: [B,T]；mask: [B,T] 1=计入损失。返回 (loss_sum, n)。"""
    h = h.reshape(-1, h.size(-1))
    t = targets.reshape(-1)
    w = mask.reshape(-1).float() if mask is not None else None
    tot = h.new_zeros((), dtype=torch.float32)
    n = 0.0
    for i in range(0, h.size(0), chunk):
        lg = m.lm_head(h[i:i + chunk]).float()
        l = F.cross_entropy(lg, t[i:i + chunk], reduction="none")
        if w is not None:
            tot = tot + (l * w[i:i + chunk]).sum()
        else:
            tot = tot + l.sum()
    n = w.sum() if w is not None else torch.tensor(float(t.numel()), device=h.device)
    return tot, n


def lm_loss(m, x, mask=None):
    """x: [B, T+2] 连续 token 窗口。prev = x[:, :T], ids = x[:, 1:T+1], 目标 = x[:, 2:]。"""
    prev, ids, tgt = x[:, :-2], x[:, 1:-1], x[:, 2:]
    h = backbone_train(m, ids, prev)
    tot, n = chunked_ce(m, h, tgt, None if mask is None else mask[:, 2:])
    return tot / n.clamp(min=1)


# ---------------------------------------------------------------- Stage 3 草稿头
def draft_train(m, h, kv, ids, anchors, d_rej, use_delta):
    """批量版 ScratchNanoCode10K.draft。
    h: [B,T,d] 主干 final_norm 输出；kv: {2:(k,v),3:(k,v)} 主干第 8/10 层 KV [B,n_kv,T,hd]；
    ids: [B,T]；anchors: [B,K] 锚点位置 p（>=1）；d_rej: [B,K] 被拒草稿词；use_delta: [B,K] bool。
    对每个锚点：输入 [ids[p], MASK x7]，位置 p..p+7，看得见主干 KV 的 [0, p) 和本块 8 个位置。
    返回草稿 logits [B,K,7,V]（预测 ids[p+1..p+7]）。"""
    B, K = anchors.shape
    Bk, d, T = m.block_size, m.d, ids.size(1)
    dev = ids.device
    v = ids.gather(1, anchors)                                            # [B,K]
    feat = h.gather(1, (anchors - 1).unsqueeze(-1).expand(B, K, d))      # [B,K,d]
    delta = (m.embed(v) - m.embed(d_rej)) * use_delta.unsqueeze(-1).to(h.dtype)
    ctx = feat + m.prefix_delta_up(F.silu(m.prefix_delta_down(torch.cat([feat, delta], dim=-1))))
    tok = torch.cat([m.embed(v).unsqueeze(2), m.mask_emb.view(1, 1, 1, d).expand(B, K, Bk - 1, d)], dim=2)
    z = (tok + ctx.unsqueeze(2)).reshape(B, K * Bk, d)
    pos = (anchors.unsqueeze(-1) + m.blk_ar.view(1, 1, Bk)).reshape(B, K * Bk)
    cos, sin = _rope_b(m, pos, z.dtype)
    # 掩码 [B,1,K*8, T+K*8]
    blk_id = torch.arange(K * Bk, device=dev) // Bk
    ctx_vis = torch.arange(T, device=dev).view(1, 1, 1, T) < anchors.view(B, K, 1, 1)
    ctx_vis = ctx_vis.expand(B, K, Bk, T).reshape(B, K * Bk, T)
    own = (blk_id.view(-1, 1) == blk_id.view(1, -1)).unsqueeze(0).expand(B, -1, -1)
    mask = torch.cat([ctx_vis, own], dim=-1).unsqueeze(1)
    for blk, layer_i in zip(m.draft_layers, m.draft_kv):
        q, k, vv = _qkv(blk, z, cos, sin)
        ck, cv = kv[layer_i]
        kk = torch.cat([ck.to(k.dtype), k], dim=2)
        vk = torch.cat([cv.to(vv.dtype), vv], dim=2)
        o = F.scaled_dot_product_attention(q, _rep(blk, kk), _rep(blk, vk), attn_mask=mask)
        z = _attn_out(blk, z, o)
    zb = z.reshape(B * K, Bk, d)
    z_conv = m.draft_dw_conv(zb.transpose(1, 2)).transpose(1, 2)
    z_gate = torch.sigmoid(m.draft_gate_up(F.silu(m.draft_gate_down(zb))))
    zb = m.draft_norm(zb + z_conv * z_gate)
    return m.lm_head(zb[:, 1:]).view(B, K, Bk - 1, -1)


# ---------------------------------------------------------------- 批量贪心生成（左填充 + KV 缓存）
class BatchCache:
    def __init__(self, m, B, L, offsets, device, dtype):
        self.k = torch.zeros(m.n_attn, B, m.n_kv, L, m.hd, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.conv = torch.zeros(m.n_conv, B, 2, m.d, device=device, dtype=dtype)
        self.off = offsets  # [B] 每条序列真实内容开始的槽号
        self.L = L


def forward_cached(m, ids, prev_ids, s0, cache, prev2_first=None):
    """处理槽号 s0..s0+T-1 的 T 个新词（全 batch 共享槽号）。返回 h [B,T,d]。"""
    B, T = ids.shape
    dev = ids.device
    slots = torch.arange(s0, s0 + T, device=dev)
    real = slots.view(1, T) >= cache.off.view(B, 1)                      # [B,T]
    pos = (slots.view(1, T) - cache.off.view(B, 1)).clamp(min=0)
    x = m.input_emb(ids, prev_ids, prev2_first)
    cos, sin = _rope_b(m, pos, x.dtype)
    S = s0 + T
    key_sl = torch.arange(S, device=dev)
    vis = (key_sl.view(1, 1, S) >= cache.off.view(B, 1, 1)) & (key_sl.view(1, 1, S) <= slots.view(1, T, 1))
    vis = vis | (key_sl.view(1, 1, S) == slots.view(1, T, 1))            # 填充位只看自己，避免全遮挡出 NaN
    mask = vis.unsqueeze(1)
    C, A = m.conv_layers, m.attn_layers
    skips = []

    def conv(i, x):
        u = _conv_u(C[i], x) * real.unsqueeze(-1).to(x.dtype)            # 填充位的卷积输入置 0
        U = torch.cat([cache.conv[i], u], dim=1)
        cache.conv[i].copy_(U[:, -2:])
        return _conv_rest(C[i], x, U)

    def attn(i, x):
        blk = A[i]
        q, k, v = _qkv(blk, x, cos, sin)
        cache.k[i][:, :, s0:S] = k
        cache.v[i][:, :, s0:S] = v
        o = F.scaled_dot_product_attention(q, _rep(blk, cache.k[i][:, :, :S]), _rep(blk, cache.v[i][:, :, :S]),
                                           attn_mask=mask)
        return _attn_out(blk, x, o)

    if m.arch == "v3":
        return m.final_norm(m.run_v3_layers(x, conv, attn))
    lam = m.unet_lambdas
    x = conv(0, x); skips.append(x)
    x = attn(0, x); skips.append(x)
    x = conv(1, x); skips.append(x)
    x = conv(2, x); skips.append(x)
    x = attn(1, x)
    x = conv(3, x)
    x = conv(4, x + lam[0] * skips[3])
    x = attn(2, x + lam[1] * skips[2])
    x = conv(5, x + lam[2] * skips[1])
    x = attn(3, x + lam[3] * skips[0])
    return m.final_norm(x)


@torch.no_grad()
def greedy_generate(m, prompts, max_new, stop_fn=None, eot=EOT, temperature=0.0):
    """prompts: list[list[int]]（已含开头的 EOT）。返回 list[list[int]] 生成部分（不含 EOT）。
    stop_fn(tokens)->bool 可选，用于停止串。temperature>0 时按温度采样（RFT 用）。"""
    def pick(lg):
        if temperature <= 0:
            return lg.argmax(-1)
        return torch.multinomial(F.softmax(lg.float() / temperature, dim=-1), 1).squeeze(-1)

    dev = m.embed.weight.device
    dtype = m.embed.weight.dtype
    B = len(prompts)
    P = max(len(p) for p in prompts)
    L = P + max_new
    assert L <= m.max_len, f"提示词 {P} + 生成 {max_new} 超过 max_len {m.max_len}"
    ids = torch.full((B, P), eot, dtype=torch.long, device=dev)
    for b, p in enumerate(prompts):
        ids[b, P - len(p):] = torch.tensor(p, device=dev)
    off = torch.tensor([P - len(p) for p in prompts], device=dev)
    prev = torch.cat([torch.full((B, 1), eot, dtype=torch.long, device=dev), ids[:, :-1]], dim=1)
    prev = torch.where(torch.arange(P, device=dev).view(1, P) == off.view(B, 1), torch.full_like(prev, eot), prev)
    cache = BatchCache(m, B, L, off, dev, dtype)
    h = forward_cached(m, ids, prev, 0, cache)
    nxt = pick(m.lm_head(h[:, -1]))
    last = ids[:, -1]
    last2 = ids[:, -2] if P >= 2 else torch.full_like(last, eot)  # 左填充位本来就是 EOT
    outs = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(max_new):
        nl = nxt.tolist()
        dl = done.tolist()
        for b in range(B):
            if not dl[b]:
                if nl[b] == eot:
                    done[b] = True
                else:
                    outs[b].append(nl[b])
                    if stop_fn is not None and stop_fn(outs[b]):
                        done[b] = True
        if bool(done.all()) or t == max_new - 1:
            break
        h = forward_cached(m, nxt.view(B, 1), last.view(B, 1), P + t, cache, last2.view(B, 1))
        last2, last = last, nxt
        nxt = pick(m.lm_head(h[:, -1]))
    return outs


# ---------------------------------------------------------------- 投机解码生成（batch=1，推理路径原样）
@torch.no_grad()
def spec_generate(m, prompt, max_new, stop_fn=None, eot=EOT, st=None):
    """用 prefill + forward_cycle 生成（与逐词贪心等价）。返回 (tokens, n_cycles)。"""
    dev = m.embed.weight.device
    if st is None:
        st = DecodeState(m, dev, m.embed.weight.dtype)
    else:
        st.conv.zero_()
    m.prefill(st, torch.tensor([prompt], device=dev))
    out, cycles = [], 0
    limit = m.max_len - m.block_size
    while len(out) < max_new and st.pos.item() + m.block_size <= limit:
        cand = st.cand[0].tolist()
        n = int(m.forward_cycle(st)[0])
        cycles += 1
        stop = False
        for tkn in cand[:n]:
            if tkn == eot:
                stop = True
                break
            out.append(tkn)
            if (stop_fn is not None and stop_fn(out)) or len(out) >= max_new:
                stop = True
                break
        if stop:
            break
    return out, cycles


# ---------------------------------------------------------------- Muon
def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """单卡 Muon（Keller Jordan 版）：动量 + Nesterov + Newton-Schulz 正交化。只用于二维矩阵。"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(grad)
                buf = st["buf"]
                buf.lerp_(grad, 1 - g["momentum"])
                upd = grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, g["ns_steps"]).to(p.dtype)
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd, alpha=-g["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)


def param_groups(m, trainable_prefixes=None):
    """Muon：层内二维 Linear 权重；AdamW：Embedding(=lm_head)、Bigram 表、Norm、标量、卷积核。"""
    muon, adam = [], []
    seen = set()
    for name, p in m.named_parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        if trainable_prefixes is not None and not any(name.startswith(t) for t in trainable_prefixes):
            continue
        if p.dim() == 2 and name != "embed.weight" and not name.startswith("digit"):
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


DRAFT_PREFIXES = ("mask_emb", "prefix_delta_", "draft_layers", "draft_dw_conv", "draft_gate_", "draft_norm")
