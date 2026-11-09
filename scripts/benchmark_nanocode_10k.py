"""
Scratch-NanoCode-10K  第 1 步：随机权重单轮耗时测速（v7.2-fix）

相对 v7.2 原脚本的修正（原文件备份在 benchmark_nanocode_10k.v72_backup.py）：
  1. 短卷积状态回滚：原来取 U[n-1 : n+1]（错一位），改为 U[n : n+2]。
  2. Bigram 表：原来 clamp(id, 255) —— 字节级 BPE 的 0..255 是原始字节，最不常用；
     改为 "token id -> 高频排名" 映射表（bigram_rank），不在前 256 的位置贡献置 0。
  3. 注意力：
     - KV 缓存开到真实长度（默认 1024），写入位置跟随 pos（GPU 上的张量，图内推进）；
     - 掩码按真实位置做因果遮挡（被拒草稿留下的旧 KV 自动被遮住，下一轮覆盖）；
     - 补上 RoPE 与 QK-Norm；
     - GQA 不再 repeat_interleave 复制 KV：把同组 3 个 query 头折叠进序列维，零拷贝。
  4. 草稿头：
     - 对主干第 8、10 层 KV 只读不写；草稿块自己的 K/V 写在缓存末尾的 16 个暂存槽；
     - 输入改为 [新锚点 v, MASK x 7]，位置与下一轮候选对齐（pos_new .. pos_new+7）；
     - 全部接受（n=8）时 Δe 置 0；prev_last_id 每轮更新为最后一个被接受的词。
  5. 新增 --compile：torch.compile(max-autotune-no-cudagraphs) 后再手动录 CUDA Graph，
     与不编译版本对比；新增 --profile：统计一轮的 CUDA kernel 个数。

说明：
  - 掩码式 SDPA 每次扫描整个缓存（max_len），耗时对应的是 "上下文 = max_len" 的情况，
    偏保守；正式版可换 flash_attn_with_kvcache(cache_seqlens=...) 按真实长度计算。
  - 随机权重只能测单轮耗时，测不出 τ；下面打印的 TPS 均为 "按假设 τ 折算"。
  - 正确性（投机解码输出 == 逐词贪心）由同目录 test_nanocode_correctness.py 在 CPU 上验证。
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRATCH = 16  # 缓存末尾给草稿块用的暂存槽数（>= block_size，且保持 16 对齐）


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


def build_rope(n_pos: int, hd: int, base: float = 10000.0):
    inv = 1.0 / base ** (torch.arange(0, hd, 2).float() / hd)
    f = torch.outer(torch.arange(n_pos).float(), inv)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x: [1, H, T, hd]; cos/sin: [T, hd]
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class GatedShortConvBlock(nn.Module):
    """LFM2 式门控 k=3 因果深度卷积 + SwiGLU FFN。"""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.dw_conv = nn.Conv1d(d, d, kernel_size=3, padding=0, groups=d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.gate_up = nn.Linear(d, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x, conv_state):
        """x: [1, T, d]; conv_state: [1, 2, d]（前文最后 2 个词的卷积输入）。
        返回 (x, U)，U = [conv_state; u] 形状 [1, 2+T, d]，供回滚使用。"""
        u, gate = self.in_proj(self.norm1(x)).chunk(2, dim=-1)
        u = u * F.silu(gate)
        U = torch.cat([conv_state, u], dim=1)
        c = self.dw_conv(U.transpose(1, 2)).transpose(1, 2)  # [1, T, d]
        x = x + self.out_proj(F.silu(c))
        g, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * up), U


class GQAAttentionBlock(nn.Module):
    """GQA + QK-Norm + RoPE + SwiGLU FFN。KV 写入位置与掩码都由调用方给出（图内张量）。"""

    def __init__(self, d: int, d_ff: int, n_q: int, n_kv: int):
        super().__init__()
        self.d, self.n_q, self.n_kv = d, n_q, n_kv
        self.hd = d // n_q
        self.rep = n_q // n_kv
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, (n_q + 2 * n_kv) * self.hd, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.gate_up = nn.Linear(d, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x, cos, sin, k_cache, v_cache, write_idx, mask):
        """x: [1,T,d]; k_cache/v_cache: [1,n_kv,L,hd]; write_idx: [T] 缓存槽号;
        mask: [1,1,rep*T,L] bool（True = 可见）。"""
        T = x.size(1)
        q, k, v = torch.split(self.qkv(self.norm1(x)),
                              [self.n_q * self.hd, self.n_kv * self.hd, self.n_kv * self.hd], dim=-1)
        q = q.view(1, T, self.n_q, self.hd).transpose(1, 2)
        k = k.view(1, T, self.n_kv, self.hd).transpose(1, 2)
        v = v.view(1, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        k_cache.index_copy_(2, write_idx, k)
        v_cache.index_copy_(2, write_idx, v)
        # GQA 零拷贝：同一 KV 头下的 rep 个 query 头折叠进序列维 -> [1, n_kv, rep*T, hd]
        qf = q.reshape(1, self.n_kv, self.rep * T, self.hd)
        o = F.scaled_dot_product_attention(qf, k_cache, v_cache, attn_mask=mask)
        o = o.reshape(1, self.n_q, T, self.hd).transpose(1, 2).reshape(1, T, self.d)
        x = x + self.o_proj(o)
        g, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * up)


class DecodeState:
    """推理状态，全部是固定地址的 GPU 张量（CUDA Graph 要求）。"""

    def __init__(self, model, device, dtype):
        L = model.max_len + SCRATCH
        B, d = model.block_size, model.d
        self.cand = torch.zeros(1, B, dtype=torch.long, device=device)       # 本轮候选 [锚点, 7 个草稿]
        self.prev_last = torch.zeros(1, 1, dtype=torch.long, device=device)  # 候选第 0 位之前的那个词
        self.prev_last2 = torch.zeros(1, 1, dtype=torch.long, device=device)  # 再往前一个词（v3 数字槽位用）
        self.conv = torch.zeros(model.n_conv, 1, 2, d, dtype=dtype, device=device)
        self.k = torch.zeros(model.n_attn, 1, model.n_kv, L, model.hd, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)           # 候选第 0 位的序列位置

    def tensors(self):
        return [self.cand, self.prev_last, self.prev_last2, self.conv, self.k, self.v, self.pos]

    def snapshot(self):
        return [t.clone() for t in self.tensors()]

    def restore(self, snap):
        for t, s in zip(self.tensors(), snap):
            t.copy_(s)


class ScratchNanoCode10K(nn.Module):
    def __init__(self, d=768, d_ff=2048, vocab_size=8192, block_size=8, max_len=1024, n_q=12, n_kv=4, arch="v1"):
        super().__init__()
        assert max_len % 16 == 0
        assert arch in ("v1", "v3")
        self.arch = arch
        # v1：10 层 C A C C A C C A C A；v3：16 层 4 x [C C C A]，没有跨层残差
        self.n_conv, self.n_attn = (6, 4) if arch == "v1" else (12, 4)
        self.draft_kv = (2, 3) if arch == "v1" else (2, 3, 3)  # 每个草稿层读主干第几个注意力层的 KV
        self.d, self.vocab_size, self.block_size, self.max_len = d, vocab_size, block_size, max_len
        self.n_q, self.n_kv, self.hd, self.rep = n_q, n_kv, d // n_q, n_q // n_kv

        # 1. 共享词表（主干验证与草稿共用）
        self.embed = nn.Embedding(vocab_size, d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # 2. Top-256 Bigram 表 + "token id -> 高频排名" 映射（-1 = 不在前 256）
        self.bigram_table = nn.Parameter(torch.zeros(256, 256, d))
        self.bigram_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("bigram_rank", torch.full((vocab_size,), -1, dtype=torch.long))

        # 3. 10 层混合主干：C A C C A C C A C A
        self.conv_layers = nn.ModuleList([GatedShortConvBlock(d, d_ff) for _ in range(self.n_conv)])
        self.attn_layers = nn.ModuleList([GQAAttentionBlock(d, d_ff, n_q, n_kv) for _ in range(self.n_attn)])
        if arch == "v1":
            self.unet_lambdas = nn.Parameter(torch.zeros(4))
        else:
            # 数字槽位：连续 2 位 / 3 位数字，在最后一位处查表加到输入
            self.digit2 = nn.Embedding(100, d)
            self.digit3 = nn.Embedding(1000, d)
            nn.init.zeros_(self.digit2.weight)
            nn.init.zeros_(self.digit3.weight)
            self.register_buffer("digit_val", torch.full((vocab_size,), -1, dtype=torch.long))
        self.final_norm = RMSNorm(d)

        # 4. 2 层块扩散草稿头 + 秩-64 前缀差值修正 + 秩-64 门控块内短卷积
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.prefix_delta_down = nn.Linear(2 * d, 64, bias=False)
        self.prefix_delta_up = nn.Linear(64, d, bias=False)
        nn.init.zeros_(self.prefix_delta_up.weight)
        self.draft_layers = nn.ModuleList([GQAAttentionBlock(d, d_ff, n_q, n_kv) for _ in self.draft_kv])
        self.draft_dw_conv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.draft_gate_down = nn.Linear(d, 64, bias=False)
        self.draft_gate_up = nn.Linear(64, d, bias=False)
        nn.init.zeros_(self.draft_gate_up.weight)
        self.draft_norm = RMSNorm(d)

        cos, sin = build_rope(max_len + SCRATCH, self.hd)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("slot_ar", torch.arange(max_len + SCRATCH), persistent=False)
        self.register_buffer("blk_ar", torch.arange(block_size), persistent=False)
        self.register_buffer("two_ar", torch.arange(2), persistent=False)
        self.bench_wrap = None  # 仅测速用：(limit, reset)，pos 超过 limit 时绕回，防止越界
        self.debug = False      # 测试用：保存每轮验证 logits

    # ---------- 工具 ----------
    def set_bigram_vocab(self, top_ids: torch.Tensor):
        """top_ids: 语料里最常见的 256 个 token id（按频率排序）。"""
        self.bigram_rank.fill_(-1)
        self.bigram_rank[top_ids.to(self.bigram_rank.device)] = torch.arange(
            len(top_ids), device=self.bigram_rank.device)

    def bigram(self, prev_ids, cur_ids):
        rp, rc = self.bigram_rank[prev_ids], self.bigram_rank[cur_ids]
        valid = ((rp >= 0) & (rc >= 0)).unsqueeze(-1)
        e = self.bigram_table[rp.clamp(min=0), rc.clamp(min=0)]
        return self.bigram_scale * e * valid.to(e.dtype)

    def set_digit_vocab(self, digit_ids):
        """digit_ids[i] = 数字 i 对应的 token id。"""
        self.digit_val.fill_(-1)
        self.digit_val[torch.as_tensor(digit_ids, device=self.digit_val.device)] = torch.arange(10, device=self.digit_val.device)

    def digit_slots(self, prev2_ids, prev_ids, cur_ids):
        c, p, pp = self.digit_val[cur_ids], self.digit_val[prev_ids], self.digit_val[prev2_ids]
        v2 = (c >= 0) & (p >= 0)
        v3 = v2 & (pp >= 0)
        e2 = self.digit2((p * 10 + c).clamp(0, 99)) * v2.unsqueeze(-1)
        e3 = self.digit3((pp * 100 + p * 10 + c).clamp(0, 999)) * v3.unsqueeze(-1)
        return e2 + e3

    def input_emb(self, ids, prev_ids, prev2_first=None):
        """prev2_first: 第一个位置的“前前词”，[B,1]；None 表示序列开头（EOT）。"""
        x = self.embed(ids) + self.bigram(prev_ids, ids)
        if self.arch == "v3":
            if prev2_first is None:
                prev2_first = torch.zeros_like(prev_ids[:, :1])
            x = x + self.digit_slots(torch.cat([prev2_first, prev_ids[:, :-1]], dim=1), prev_ids, ids).to(x.dtype)
        return x

    def run_v3_layers(self, x, conv, attn):
        for g in range(4):
            for j in range(3):
                x = conv(3 * g + j, x)
            x = attn(g, x)
        return x

    def rope(self, positions, dtype):
        return self.rope_cos[positions].to(dtype), self.rope_sin[positions].to(dtype)

    def causal_mask(self, positions):
        # 槽号 = 序列位置；暂存槽号 >= max_len > 任何 query 位置，自动不可见
        T = positions.numel()
        m = self.slot_ar.view(1, -1) <= positions.view(-1, 1)  # [T, L]
        return m.repeat(self.rep, 1).view(1, 1, self.rep * T, -1)

    def rollback(self, Us, n, st):
        """接受 n 个候选后，下一轮需要的卷积历史 = U[n : n+2]（候选第 n-2、n-1 位）。"""
        idx = (n.view(1, 1, 1) + self.two_ar.view(1, 2, 1)).expand(1, 2, self.d)
        for i in range(self.n_conv):
            st.conv[i].copy_(torch.gather(Us[i], 1, idx))

    # ---------- 主干 ----------
    def backbone(self, ids, prev_ids, positions, st, prev2_first=None):
        x = self.input_emb(ids, prev_ids, prev2_first)
        cos, sin = self.rope(positions, x.dtype)
        mask = self.causal_mask(positions)
        C, A = self.conv_layers, self.attn_layers
        Us, skips = [], []

        def conv(i, x):
            x, U = C[i](x, st.conv[i])
            Us.append(U)
            return x

        def attn(i, x):
            return A[i](x, cos, sin, st.k[i], st.v[i], positions, mask)

        if self.arch == "v3":
            return self.final_norm(self.run_v3_layers(x, conv, attn)), Us
        lam = self.unet_lambdas
        x = conv(0, x); skips.append(x)          # L1
        x = attn(0, x); skips.append(x)          # L2
        x = conv(1, x); skips.append(x)          # L3
        x = conv(2, x); skips.append(x)          # L4
        x = attn(1, x)                           # L5
        x = conv(3, x)                           # L6
        x = conv(4, x + lam[0] * skips[3])       # L7  <- L4
        x = attn(2, x + lam[1] * skips[2])       # L8  <- L3
        x = conv(5, x + lam[2] * skips[1])       # L9  <- L2
        x = attn(3, x + lam[3] * skips[0])       # L10 <- L1
        return self.final_norm(x), Us

    # ---------- 草稿头 ----------
    def draft(self, st, v, feat, delta):
        """v: [1,1] 新锚点；feat: [1,1,d] 最后一个被接受位置的主干特征；delta: [1,1,d]。
        st.pos 已推进到新锚点的位置。返回下一轮 7 个草稿词 [1,7]。"""
        B, L = self.block_size, self.max_len + SCRATCH
        tok = torch.cat([self.embed(v), self.mask_emb.view(1, 1, -1).expand(1, B - 1, -1)], dim=1)
        ctx = feat + self.prefix_delta_up(F.silu(self.prefix_delta_down(torch.cat([feat, delta], dim=-1))))
        z = tok + ctx
        positions = st.pos + self.blk_ar
        cos, sin = self.rope(positions, z.dtype)
        write_idx = self.max_len + self.blk_ar  # 只写暂存槽，不碰主干已确认的 KV
        vis = (self.slot_ar < st.pos) | ((self.slot_ar >= self.max_len) & (self.slot_ar < self.max_len + B))
        mask = vis.view(1, 1, 1, L).expand(1, 1, self.rep * B, L)
        for blk, i in zip(self.draft_layers, self.draft_kv):  # v1 复用主干 L8/L10 的 KV，v3 复用 L12/L16/L16
            z = blk(z, cos, sin, st.k[i], st.v[i], write_idx, mask)
        z_conv = self.draft_dw_conv(z.transpose(1, 2)).transpose(1, 2)
        z_gate = torch.sigmoid(self.draft_gate_up(F.silu(self.draft_gate_down(z))))
        z = self.draft_norm(z + z_conv * z_gate)
        return torch.argmax(self.lm_head(z[:, 1:]), dim=-1)

    # ---------- 推理入口 ----------
    @torch.no_grad()
    def prefill(self, st, prompt, bos_id: int = 0):
        P = prompt.size(1)
        dev = prompt.device
        prev = torch.cat([torch.full((1, 1), bos_id, dtype=torch.long, device=dev), prompt[:, :-1]], dim=1)
        h, Us = self.backbone(prompt, prev, torch.arange(P, device=dev), st)
        self.rollback(Us, torch.tensor([P], device=dev), st)
        v = torch.argmax(self.lm_head(h[:, -1:]), dim=-1)  # [1,1]
        st.pos.fill_(P)
        st.prev_last.copy_(prompt[:, -1:])
        st.prev_last2.copy_(prompt[:, -2:-1] if P >= 2 else torch.full((1, 1), bos_id, dtype=torch.long, device=dev))
        drafts = self.draft(st, v, h[:, -1:], torch.zeros_like(h[:, -1:]))
        st.cand[:, :1].copy_(v)
        st.cand[:, 1:].copy_(drafts)

    @torch.no_grad()
    def forward_cycle(self, st):
        """一轮：验证 8 个候选 -> 图内比对 -> 回滚 -> 前缀差值修正 -> 起草下一轮。
        全程无 .item()，返回 n_accepted（GPU 张量，1..8）。"""
        B, d = self.block_size, self.d
        cand = st.cand
        positions = st.pos + self.blk_ar
        prev_ids = torch.cat([st.prev_last, cand[:, :-1]], dim=1)
        h, Us = self.backbone(cand, prev_ids, positions, st, st.prev_last2)
        logits = self.lm_head(h)
        if self.debug:
            self.last_logits = logits.detach().clone()
        pred = torch.argmax(logits, dim=-1)                               # [1,8]

        matches = (cand[:, 1:] == pred[:, :-1]).long()
        n = 1 + torch.cumprod(matches, dim=-1).sum(-1)                    # [1]
        last = (n - 1).view(1, 1)
        v = pred.gather(1, last)                                          # 新锚点
        d_rej = cand.gather(1, n.clamp(max=B - 1).view(1, 1))             # 被拒的草稿词
        not_full = (n < B).to(h.dtype).view(1, 1, 1)                      # 全部接受时 Δe = 0
        feat = h.gather(1, last.view(1, 1, 1).expand(1, 1, d))
        delta = (self.embed(v) - self.embed(d_rej)) * not_full

        self.rollback(Us, n, st)
        st.prev_last2.copy_(prev_ids.gather(1, last))
        st.prev_last.copy_(cand.gather(1, last))
        st.pos.add_(n)
        if self.bench_wrap is not None:
            limit, reset = self.bench_wrap
            st.pos.copy_(torch.where(st.pos > limit, reset, st.pos))

        drafts = self.draft(st, v, feat, delta)
        st.cand[:, :1].copy_(v)
        st.cand[:, 1:].copy_(drafts)
        return n

    @torch.no_grad()
    def reference_logits(self, ids, bos_id: int = 0):
        """不投机、整段重算的参考前向（测试用）。"""
        st = DecodeState(self, ids.device, self.embed.weight.dtype)
        prev = torch.cat([torch.full((1, 1), bos_id, dtype=torch.long, device=ids.device), ids[:, :-1]], dim=1)
        h, _ = self.backbone(ids, prev, torch.arange(ids.size(1), device=ids.device), st)
        return self.lm_head(h)


# ======================= 测速 =======================
def capture_graph(fn, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def time_graph_cycles(g, st, snap, n_iters, rounds_per_replay=1, host_check=None):
    st.restore(snap)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n_iters // rounds_per_replay):
        g.replay()
        if host_check is not None:
            host_check()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=1024, help="KV 缓存长度（SDPA 每轮扫描整段）")
    ap.add_argument("--ctx", type=int, default=512, help="测速时的提示词长度")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--replay", type=int, default=4, help="每回放几轮同步一次 CPU 检查停止条件")
    ap.add_argument("--compile", action="store_true", help="额外测 torch.compile + CUDA Graph 版本")
    ap.add_argument("--profile", action="store_true", help="统计一轮的 CUDA kernel 个数")
    ap.add_argument("--chars-per-token", type=float, default=3.2, help="假设值，分词器训好后换成实测值")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "测速需要 CUDA；CPU 上请运行 test_nanocode_correctness.py"
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    model = ScratchNanoCode10K(max_len=args.max_len).to(device=dev, dtype=dtype).eval()
    model.set_bigram_vocab(torch.randperm(model.vocab_size)[:256])  # 正式版用语料词频前 256
    B = model.block_size
    model.bench_wrap = (torch.tensor([args.max_len - 2 * B], device=dev), torch.tensor([args.ctx], device=dev))

    active = sum(p.numel() for n, p in model.named_parameters() if "bigram_table" not in n)
    total = sum(p.numel() for p in model.parameters())
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"每轮参与计算的参数（不含 Bigram 表）: {active / 1e6:.2f}M ({active * 2 / 1e6:.1f} MB BF16)")
    print(f"总参数（含 Bigram 表）:              {total / 1e6:.2f}M ({total * 2 / 1e6:.1f} MB BF16)")
    print(f"KV 缓存长度 {args.max_len}，提示词长度 {args.ctx}，每 {args.replay} 轮同步一次 CPU")

    st = DecodeState(model, dev, dtype)
    model.prefill(st, torch.randint(0, model.vocab_size, (1, args.ctx), device=dev))
    snap = st.snapshot()

    # 0. 不录图（eager）作参照
    st.restore(snap)
    for _ in range(10):
        model.forward_cycle(st)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n_eager = 200
    a.record()
    for _ in range(n_eager):
        model.forward_cycle(st)
    b.record()
    torch.cuda.synchronize()
    eager_ms = a.elapsed_time(b) / n_eager

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        st.restore(snap)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model.forward_cycle(st)
            torch.cuda.synchronize()
        n_k = sum(e.count for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA)
        print(f"一轮 CUDA kernel 数（不编译）: {n_k}")

    results = {}

    def bench(tag, cycle_fn):
        # (a) 单轮图，纯回放
        st.restore(snap)
        g1 = capture_graph(lambda: cycle_fn(st))
        pure = time_graph_cycles(g1, st, snap, args.iters)
        # (b) K 轮一张图 + 每次回放后 D2H 拷回、CPU 检查停止条件
        K = args.replay
        dev_buf = torch.zeros(K, B + 1, dtype=torch.long, device=dev)
        host_buf = torch.empty(K, B + 1, dtype=torch.long, pin_memory=True)

        def k_rounds():
            for k in range(K):
                dev_buf[k, :B].copy_(st.cand[0])
                n = cycle_fn(st)
                dev_buf[k, B].copy_(n[0])

        def host_check():
            host_buf.copy_(dev_buf)        # 同步拷贝 = 真实的 CPU 等待
            _ = int(host_buf[0, B])        # 这里正式版检查 EOS / 停止串

        st.restore(snap)
        gK = capture_graph(k_rounds)
        real = time_graph_cycles(gK, st, snap, args.iters, rounds_per_replay=K, host_check=host_check)
        results[tag] = (pure, real)

    bench("不编译 + CUDA Graph", model.forward_cycle)
    if args.compile:
        try:
            compiled = torch.compile(model.forward_cycle, mode="max-autotune-no-cudagraphs", dynamic=False)
            st.restore(snap)
            for _ in range(3):
                compiled(st)
            torch.cuda.synchronize()
            bench("torch.compile + CUDA Graph", compiled)
        except Exception as e:  # noqa: BLE001
            print(f"[compile 版本失败] {type(e).__name__}: {e}")

    print(f"\n=== 随机权重单轮耗时（{args.iters} 轮）===")
    print(f"  eager（不录图）: {eager_ms:.3f} ms/轮")
    for tag, (pure, real) in results.items():
        print(f"  {tag}: 纯回放 {pure:.3f} ms/轮 | 含每 {args.replay} 轮 CPU 同步 {real:.3f} ms/轮")
    print("\n[注意] 随机权重测不出 τ。以下 TPS 按假设 τ 折算；字符数按假设每 token "
          f"{args.chars_per_token} 字符折算（分词器训好后用实测值）。")
    for tag, (_, real) in results.items():
        print(f"  {tag}:")
        for tau in (4.0, 4.5, 5.0, 5.5):
            tps = tau / (real / 1000.0)
            print(f"    假设 τ={tau:.1f} -> {tps:,.0f} TPS（约 {tps * args.chars_per_token:,.0f} 字符/秒）")


if __name__ == "__main__":
    main()
