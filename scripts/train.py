"""
Scratch-NanoCode-10K 训练脚本（默认单卡；pre/sft 可用 --gpus 2 双卡，两卡各算各的梯度，经共享内存求平均）。
  --stage pre    预训练：按数据源比例抽窗口，Muon(二维矩阵) + AdamW(Embedding/Bigram/Norm/标量/卷积核)，BF16 autocast
  --stage sft    Stage 2：4 万道题（格式 A + B），只在答案部分算损失
  --stage gen    Stage 3 前置：冻结主干在题目开头上贪心生成，存 gen.jsonl
  --stage draft  Stage 3：冻结主干/词表/Bigram，只训 2 层草稿头（+ 前缀差值修正 + 块内短卷积）
断点续训：每 --ckpt-min 分钟把 模型+优化器+步数+数据计数器 原子写入 <out>/<stage>_ckpt.pt；--resume 自动接上。
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nanocode_lib as L  # noqa: E402

MIX = {"oci": 0.60, "ann": 0.25, "cosmo": 0.10, "sft2": 0.02, "mago": 0.015, "mage": 0.015}
# v2：数学 + 编程各约一半（编程 45%、数学 45%、英文 10%）；gsm8k_train/math_train 整体是精调题，不进预训练
MIX_V2 = {"oci": 0.26, "ann": 0.14, "cosmo": 0.03, "sft2": 0.01, "mago": 0.01,
          "finemath": 0.25, "omi2": 0.13, "numina": 0.05, "metamath": 0.015, "orca": 0.005,
          "fwedu": 0.10}
VAL_SRCS = ["oci", "ann", "cosmo"]
SFT_FILES = ["sft.jsonl"]
MAX_EPOCHS = 2.0  # 单个数据源最多重复几遍（§4.2：不够先重复训 2 遍）
RANK = 0  # 当前进程是第几张卡；日志、验证、断点只在 0 号卡做
MAX_PARAMS = 450_000_000  # 共享内存按这个参数量上限开（float32，每卡一份）
CTRL_SLOTS = 64           # 每卡的控制区：loss、token 数、梯度长度、各数据源计数器
BARRIER_TIMEOUT = 3600    # 秒；另一张卡异常时最多等这么久（正常由父进程先发现并终止）


def log(out_dir, msg, **kv):
    if RANK != 0:
        return
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(Path(out_dir) / "train.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if kv:
        with open(Path(out_dir) / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(kv) + "\n")


def build_model(args, dev):
    if args.small:
        cfg = dict(d=128, d_ff=256, n_q=4, n_kv=2)
    else:
        cfg = dict(d=args.d, d_ff=args.d_ff, n_q=args.n_q, n_kv=args.n_kv)
    meta = Path(args.data_dir) / "meta.json"
    vocab = json.load(open(meta, encoding="utf-8")).get("_vocab", 8192) if meta.exists() else 8192
    cfg["vocab_size"] = vocab
    if args.arch != "v1":
        cfg["arch"] = args.arch
    if RANK == 0:
        json.dump(cfg, open(Path(args.out_dir) / "model_cfg.json", "w"), indent=1)  # 评测/对话按这个建模型
    m = L.ScratchNanoCode10K(max_len=args.seq_len, **cfg)
    return m.to(dev)


def save_ckpt(path, obj):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lr_mult(step, total, warmup, decay_frac):
    if step < warmup:
        return (step + 1) / warmup
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return 1.0
    return max(0.0, (total - step) / max(1, total - d0))


# ---------------------------------------------------------------- 多卡同步（Windows 上 gloo 的 all_reduce 实测死锁，改用共享内存 + Barrier）
class MultiGpuSync:
    def __init__(self, rank, gpus, shm_name, bar):
        self.rank, self.gpus, self.bar = rank, gpus, bar
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.view = torch.from_numpy(np.ndarray((gpus, MAX_PARAMS), dtype=np.float32, buffer=self.shm.buf))
        self.ctrl = np.ndarray((gpus, CTRL_SLOTS), dtype=np.float64, buffer=self.shm.buf, offset=4 * gpus * MAX_PARAMS)
        self.counters = None  # 各卡的数据计数器（最近一次同步时的），只由 0 号卡写进断点
        self.t_sync = 0.0

    def _wait(self):
        self.bar.wait(timeout=BARRIER_TIMEOUT)

    def average_grads(self, params, loss, tok, counters=()):
        """把本卡梯度换成各卡平均；返回各卡平均 loss、合计 token。"""
        t0 = time.time()
        ps = [p for p in params if p.grad is not None]
        flat = torch.cat([p.grad.flatten() for p in ps]).float().cpu()
        n = flat.numel()
        assert n <= MAX_PARAMS, f"梯度长度 {n} 超过共享内存上限 {MAX_PARAMS}"
        assert 3 + len(counters) <= CTRL_SLOTS
        self.view[self.rank, :n].copy_(flat)
        row = self.ctrl[self.rank]
        row[0], row[1], row[2] = loss, tok, n
        row[3:3 + len(counters)] = counters
        self._wait()
        if not (self.ctrl[:, 2] == n).all():
            raise RuntimeError(f"各卡梯度长度不一致：{self.ctrl[:, 2].tolist()}")
        avg = self.view[0, :n].clone()
        for r in range(1, self.gpus):
            avg += self.view[r, :n]
        avg /= self.gpus
        loss_avg, tok_sum = float(self.ctrl[:, 0].mean()), int(self.ctrl[:, 1].sum())
        self.counters = self.ctrl[:, 3:3 + len(counters)].astype(np.int64).tolist()
        self._wait()  # 第二道：保证读完之前没有卡开始覆盖共享区
        flat = avg.to(ps[0].device, non_blocking=True)
        i = 0
        for p in ps:
            k = p.numel()
            p.grad.copy_(flat[i:i + k].view_as(p).to(p.grad.dtype))
            i += k
        self.t_sync += time.time() - t0
        return loss_avg, tok_sum

    @torch.no_grad()
    def verify_params(self, params):
        """逐位比较各卡参数；不一致就用 0 号卡的覆盖。返回最大差值（0 = 完全一致）。"""
        ps = list(params)
        flat = torch.cat([p.flatten() for p in ps]).float().cpu()
        n = flat.numel()
        assert n <= MAX_PARAMS, f"参数量 {n} 超过共享内存上限 {MAX_PARAMS}"
        self.view[self.rank, :n].copy_(flat)
        self._wait()
        diff = max((float((self.view[r, :n] - self.view[0, :n]).abs().max()) for r in range(1, self.gpus)), default=0.0)
        ref = self.view[0, :n].clone() if diff > 0 and self.rank != 0 else None
        self._wait()
        if ref is not None:
            ref = ref.to(ps[0].device)
            i = 0
            for p in ps:
                k = p.numel()
                p.copy_(ref[i:i + k].view_as(p).to(p.dtype))
                i += k
        return diff

    def close(self):
        del self.view, self.ctrl
        self.shm.close()


# ---------------------------------------------------------------- 预训练数据
class Mixture:
    def __init__(self, data_dir, T, total_tokens, seed, out_dir, rank=0, gpus=1):
        meta = json.load(open(Path(data_dir) / "meta.json", encoding="utf-8"))
        self.T, self.seed = T, seed
        self.rank, self.gpus = rank, gpus
        self.srcs, self.data, self.nwin = [], {}, {}
        avail = {}
        for s in MIX:
            p = Path(data_dir) / f"{s}.bin"
            if s in meta and p.exists() and p.stat().st_size > 2 * (T + 2) * 2:
                self.data[s] = np.memmap(p, dtype=np.uint16, mode="r")
                self.nwin[s] = (len(self.data[s]) - 2) // (T + 1)
                avail[s] = len(self.data[s])
                self.srcs.append(s)
        # 按比例分配，单源超过 MAX_EPOCHS 遍就封顶，把多出来的份额按比例分给其它源
        want = {s: MIX[s] for s in self.srcs}
        tot_w = sum(want.values())
        want = {s: w / tot_w * total_tokens for s, w in want.items()}
        for _ in range(10):
            capped = {s: min(want[s], avail[s] * MAX_EPOCHS) for s in self.srcs}
            short = total_tokens - sum(capped.values())
            free = [s for s in self.srcs if capped[s] < avail[s] * MAX_EPOCHS]
            if short <= 1 or not free:
                break
            fw = sum(capped[s] for s in free)
            want = {s: capped[s] + (short * capped[s] / fw if s in free else 0) for s in self.srcs}
        self.plan = capped
        self.w = np.array([capped[s] for s in self.srcs]) / sum(capped.values())
        self.counter = {s: 0 for s in self.srcs}
        self.perm_cache = {}
        lines = [f"  {s}: 有 {avail[s] / 1e6:.0f}M token，计划用 {capped[s] / 1e6:.0f}M（{capped[s] / avail[s]:.2f} 遍，"
                 f"占 {self.w[i] * 100:.1f}%）" for i, s in enumerate(self.srcs)]
        log(out_dir, "数据配比：\n" + "\n".join(lines))
        self.feasible_tokens = sum(capped.values())

    def _perm(self, s, epoch):
        key = (s, epoch)
        if key not in self.perm_cache:
            self.perm_cache = {k: v for k, v in self.perm_cache.items() if k[0] != s}
            rng = np.random.default_rng(self.seed * 1000 + epoch * 31 + self.srcs.index(s))
            self.perm_cache[key] = rng.permutation(self.nwin[s])
        return self.perm_cache[key]

    def batch(self, global_micro, B):
        rng = np.random.default_rng(self.seed * 7919 + global_micro * self.gpus + self.rank)
        choice = rng.choice(len(self.srcs), size=B, p=self.w)
        rows = []
        for c in choice:
            s = self.srcs[c]
            k = self.counter[s]
            self.counter[s] += 1
            j = k * self.gpus + self.rank  # 同一排列按卡隔位取，两卡窗口不重复（单卡时 j = k）
            w = self._perm(s, j // self.nwin[s])[j % self.nwin[s]]
            a = w * (self.T + 1)
            rows.append(np.asarray(self.data[s][a:a + self.T + 2], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))


def val_windows(data_dir, T, n=64):
    out = []
    for s in VAL_SRCS:
        p = Path(data_dir) / f"val_{s}.bin"
        if not p.exists():
            continue
        d = np.memmap(p, dtype=np.uint16, mode="r")
        k = min(n // len(VAL_SRCS), (len(d) - 2) // (T + 1))
        for i in range(k):
            out.append(np.asarray(d[i * (T + 1):i * (T + 1) + T + 2], dtype=np.int64))
    return torch.from_numpy(np.stack(out)) if out else None


@torch.no_grad()
def eval_loss(m, fwd, val, B, dev, amp):
    m.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val), B):
        x = val[i:i + B].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            tot += fwd(m, x).item() * len(x)
        n += len(x)
    m.train()
    return tot / max(1, n)


# ---------------------------------------------------------------- SFT 数据
def load_sft(data_dir, tok, T):
    rows = [json.loads(l) for fn in SFT_FILES for l in open(Path(data_dir) / fn, encoding="utf-8")]
    ex = []
    for r in rows:
        p = tok.encode(r["prompt"]).ids
        c = tok.encode(r["completion"]).ids
        x = [L.EOT, L.EOT] + p + c + [L.EOT]
        mk = [0, 0] + [0] * len(p) + [1] * (len(c) + 1)
        if len(x) > T + 2:
            continue
        ex.append((x, mk))
    return ex


def sft_batches(ex, B, epoch, seed):
    rng = random.Random(seed * 100 + epoch)
    idx = list(range(len(ex)))
    rng.shuffle(idx)
    # 按长度分桶减少填充：每 50 个 batch 内排序
    chunk = B * 50
    batches = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: len(ex[j][0]))
        batches += [part[k:k + B] for k in range(0, len(part), B)]
    rng.shuffle(batches)
    return batches


def collate(ex, ids_list):
    n = max(len(ex[j][0]) for j in ids_list)
    x = torch.zeros(len(ids_list), n, dtype=torch.long)
    mk = torch.zeros(len(ids_list), n, dtype=torch.long)
    for i, j in enumerate(ids_list):
        a, b = ex[j]
        x[i, :len(a)] = torch.tensor(a)
        mk[i, :len(b)] = torch.tensor(b)
    return x, mk


# ---------------------------------------------------------------- 主流程
def make_opts(m, args, prefixes=None):
    muon_p, adam_p = L.param_groups(m, prefixes)
    opts = [L.Muon(muon_p, lr=args.lr_muon, momentum=0.95)]
    if adam_p:
        fused = m.embed.weight.is_cuda
        opts.append(torch.optim.AdamW(adam_p, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-10,
                                      fused=fused))
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    n_m, n_a = sum(p.numel() for p in muon_p), sum(p.numel() for p in adam_p)
    return opts, n_m, n_a


def set_lr(opts, mult):
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * mult


def load_weights(m, path, dev):
    sd = torch.load(path, map_location=dev, weights_only=False)
    sd = sd.get("model", sd)
    m.load_state_dict(sd)


def run_pre_or_sft(args, dev, sync=None):
    rank, gpus = sync.rank if sync else 0, sync.gpus if sync else 1
    out = Path(args.out_dir)
    m = build_model(args, dev)
    amp = dev.type == "cuda" or args.cpu_amp
    ck_path = out / f"{args.stage}_ckpt.pt"
    resume = ck_path.exists() and args.resume
    if args.stage == "pre":
        L.init_for_training(m)
        top = np.load(Path(args.data_dir) / "bigram_top256.npy")
        m.set_bigram_vocab(torch.from_numpy(top).long())
        if m.arch == "v3":
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(str(Path(args.data_dir) / "tokenizer.json"))
            digit_ids = [tk.encode(str(i)).ids for i in range(10)]
            assert all(len(x) == 1 for x in digit_ids), f"数字不是单 token：{digit_ids}"
            m.set_digit_vocab([x[0] for x in digit_ids])
    else:
        load_weights(m, args.init, dev)
    m.train()
    opts, n_m, n_a = make_opts(m, args)
    log(out, f"[{args.stage}] Muon 参数 {n_m / 1e6:.2f}M，AdamW 参数 {n_a / 1e6:.2f}M，设备 {dev}，"
             f"{'BF16 autocast' if amp else 'FP32'}")

    fwd = L.lm_loss
    if args.compile and dev.type == "cuda":
        fwd = torch.compile(L.lm_loss)

    tok_per_step = args.micro_bsz * args.accum * args.seq_len * gpus
    if args.stage == "pre":
        mix = Mixture(args.data_dir, args.seq_len, args.tokens, args.seed, out, rank, gpus)
        if mix.feasible_tokens < args.tokens * 0.999:
            log(out, f"[警告] 数据最多只够 {mix.feasible_tokens / 1e9:.2f}B token（每源≤{MAX_EPOCHS}遍），按此训练")
        total_steps = int(min(args.tokens, mix.feasible_tokens) // tok_per_step)
        val = val_windows(args.data_dir, args.seq_len)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(args.data_dir) / "tokenizer.json"))
        ex = load_sft(args.data_dir, tok, args.seq_len)
        n_tok = sum(len(e[0]) for e in ex)
        steps_per_epoch = math.ceil(len(ex) / (args.micro_bsz * args.accum * gpus))
        total_steps = steps_per_epoch * args.epochs
        log(out, f"[sft] {len(ex)} 条，约 {n_tok / 1e6:.1f}M token/遍，{args.epochs} 遍，共 {total_steps} 步")
        val = None
    step = 0
    t_train = 0.0
    tokens_seen = 0
    if resume:
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        if ck.get("gpus", 1) != gpus:
            raise SystemExit(f"断点是 {ck.get('gpus', 1)} 卡存的，这次用 {gpus} 卡：步数和数据切分都对不上，不能续")
        m.load_state_dict(ck["model"])
        for o, s in zip(opts, ck["opts"]):
            o.load_state_dict(s)
        step, t_train, tokens_seen = ck["step"], ck["t_train"], ck["tokens_seen"]
        if args.stage == "pre":
            mix.counter = ck["counters"][rank] if sync else ck["counter"]
        log(out, f"从断点继续：第 {step} 步，已训 {tokens_seen / 1e9:.3f}B token")
    log(out, f"[{args.stage}] 共 {total_steps} 步，每步 {tok_per_step:,} token（{args.micro_bsz}x{args.accum}x{args.seq_len}）")

    train_params = [p for p in m.parameters() if p.requires_grad]
    if sync:
        d = sync.verify_params(m.parameters())
        log(out, f"[双卡] 开始前参数差值 {d:g}" + ("（已用 0 号卡覆盖）" if d > 0 else "（完全一致）"))
    last_ck = time.time()
    t0 = time.time()
    win_tok, win_t = 0, time.time()
    sft_plan = None
    while step < total_steps:
        mult = lr_mult(step, total_steps, args.warmup, args.decay_frac)
        set_lr(opts, mult)
        loss_acc = 0.0
        step_tok = 0
        for k in range(args.accum):
            if args.stage == "pre":
                x = mix.batch(step * args.accum + k, args.micro_bsz).to(dev, non_blocking=True)
                mk = None
                step_tok += x.numel() - 2 * x.size(0)
            else:
                ep, i = divmod(step * args.accum + k, steps_per_epoch * args.accum)
                if sft_plan is None or sft_plan[0] != ep:
                    sft_plan = (ep, sft_batches(ex, args.micro_bsz, ep, args.seed))
                bl = sft_plan[1]
                x, mk = collate(ex, bl[(i * gpus + rank) % len(bl)])
                x, mk = x.to(dev), mk.to(dev)
                step_tok += int(mk.sum())
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                loss = fwd(m, x) if mk is None else L.lm_loss(m, x, mk)
            (loss / args.accum).backward()
            loss_acc += loss.item() / args.accum
        if sync:
            cnt = [mix.counter[s] for s in mix.srcs] if args.stage == "pre" else ()
            loss_acc, step_tok = sync.average_grads(train_params, loss_acc, step_tok, cnt)
        if not math.isfinite(loss_acc):
            log(out, f"[停止] 第 {step} 步 loss = {loss_acc}（NaN/Inf）")
            sys.exit(3)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += step_tok
        win_tok += step_tok
        if step % args.log_every == 0 or step == total_steps:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            tps = win_tok / (now - win_t)
            t_train += now - win_t
            win_tok, win_t = 0, now
            eta = (total_steps - step) * tok_per_step / max(tps, 1) if args.stage == "pre" else 0
            extra = ""
            if sync:
                extra = f" 同步 {sync.t_sync / args.log_every:.2f}s/步"
                sync.t_sync = 0.0
            log(out, f"step {step}/{total_steps} loss {loss_acc:.4f} gn {float(gn):.2f} lr×{mult:.3f} "
                     f"{tps:,.0f} tok/s 已训 {tokens_seen / 1e9:.3f}B 用时 {t_train / 3600:.2f}h 剩余约 {eta / 3600:.2f}h{extra}",
                stage=args.stage, step=step, loss=loss_acc, tok_s=tps, tokens=tokens_seen, hours=t_train / 3600)
        if sync and (step % args.eval_every == 0 or step == total_steps):
            d = sync.verify_params(m.parameters())
            log(out, f"  [双卡] 第 {step} 步参数差值 {d:g}" + ("（已用 0 号卡覆盖）" if d > 0 else "（完全一致）"),
                stage=args.stage, step=step, param_diff=d)
        if rank == 0 and val is not None and (step % args.eval_every == 0 or step == total_steps):
            vl = eval_loss(m, L.lm_loss, val, args.micro_bsz, dev, amp)
            log(out, f"  验证 loss {vl:.4f}", stage=args.stage, step=step, val_loss=vl)
        if rank == 0 and (time.time() - last_ck > args.ckpt_min * 60 or step == total_steps):
            ck = {"model": m.state_dict(), "opts": [o.state_dict() for o in opts], "step": step,
                  "t_train": t_train, "tokens_seen": tokens_seen,
                  "counter": mix.counter if args.stage == "pre" else None, "args": vars(args)}
            if sync:
                ck["gpus"] = gpus
                if args.stage == "pre":
                    ck["counters"] = [dict(zip(mix.srcs, c)) for c in sync.counters]
            save_ckpt(ck_path, ck)
            last_ck = time.time()
            log(out, f"  已存断点 {ck_path}（第 {step} 步）")
    if rank == 0:
        save_ckpt(out / f"{args.stage}_final.pt", {"model": m.state_dict(), "args": vars(args)})
    log(out, f"[{args.stage}] 完成：{tokens_seen / 1e9:.3f}B token，墙钟 {(time.time() - t0) / 3600:.2f}h")


@torch.no_grad()
def run_gen(args, dev):
    """冻结主干在 4 万道题开头上贪心生成（每题随机取格式 A 或 B 的提示词）。"""
    from tokenizers import Tokenizer
    out = Path(args.out_dir)
    tok = Tokenizer.from_file(str(Path(args.data_dir) / "tokenizer.json"))
    m = build_model(args, dev)
    load_weights(m, args.init, dev)
    m = m.to(torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
    rows = [json.loads(l) for fn in SFT_FILES for l in open(Path(args.data_dir) / fn, encoding="utf-8")]
    by_id = {}
    for r in rows:
        # v1 的 sft.jsonl 有 id/fmt；数学/蒸馏数据没有，用 来源+题目哈希 当 id，fmt 记 "M"
        uid = r.get("id") or (r.get("src", "x") + ":" + hashlib.md5(r["prompt"].encode()).hexdigest()[:12])
        by_id.setdefault(uid, []).append(r)
    rng = random.Random(args.seed)
    prompts = []
    for uid, rs in by_id.items():
        r = rng.choice(rs)
        p = [L.EOT] + tok.encode(r["prompt"]).ids
        if len(p) + args.gen_len <= args.seq_len:
            prompts.append((uid, r.get("fmt", "M"), p))
    if args.gen_limit:
        rng.shuffle(prompts)  # 多个题库时按文件顺序截断会只剩前面的文件
        prompts = prompts[:args.gen_limit]
    prompts.sort(key=lambda x: len(x[2]))
    path = out / "gen.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
    todo = [p for p in prompts if p[0] not in done]
    log(out, f"[gen] 共 {len(prompts)} 题，已完成 {len(done)}，本次 {len(todo)}，每题最多生成 {args.gen_len} 词")
    t0, n_tok = time.time(), 0
    with open(path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.gen_bsz):
            part = todo[i:i + args.gen_bsz]
            outs = L.greedy_generate(m, [p for _, _, p in part], args.gen_len)
            for (uid, fmt, p), o in zip(part, outs):
                f.write(json.dumps({"id": uid, "fmt": fmt, "prompt_ids": p, "gen_ids": o}) + "\n")
                n_tok += len(o)
            f.flush()
            if (i // args.gen_bsz) % 10 == 0:
                el = time.time() - t0
                log(out, f"  gen {i + len(part)}/{len(todo)}，{n_tok / el:,.0f} 词/秒，"
                         f"剩余约 {(len(todo) - i - len(part)) / ((i + len(part)) / el) / 60:.1f} 分钟")
    log(out, f"[gen] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟，共生成 {n_tok:,} 词")


def run_draft(args, dev):
    out = Path(args.out_dir)
    m = build_model(args, dev)
    load_weights(m, args.init, dev)
    amp = dev.type == "cuda" or args.cpu_amp
    for name, p in m.named_parameters():
        p.requires_grad_(name.startswith(L.DRAFT_PREFIXES))
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opts, n_m, n_a = make_opts(m, args, L.DRAFT_PREFIXES)
    log(out, f"[draft] 只训草稿头：可训练参数 {n_tr / 1e6:.2f}M（Muon {n_m / 1e6:.2f}M，AdamW {n_a / 1e6:.2f}M）")
    rows = [json.loads(l) for l in open(out / "gen.jsonl", encoding="utf-8")]
    seqs = []
    for r in rows:
        s = r["prompt_ids"] + r["gen_ids"] + [L.EOT]
        P = len(r["prompt_ids"])
        if len(s) - P >= 8 and len(s) <= args.seq_len:
            seqs.append((s, P))
    rng = random.Random(args.seed)
    steps_per_epoch = math.ceil(len(seqs) / args.micro_bsz)
    total = steps_per_epoch * args.epochs
    log(out, f"[draft] 可用序列 {len(seqs)}，每条 {args.anchors} 个锚点，{args.epochs} 遍，共 {total} 步")
    ck_path = out / "draft_ckpt.pt"
    step = 0
    if ck_path.exists() and args.resume:
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        for o, s in zip(opts, ck["opts"]):
            o.load_state_dict(s)
        step = ck["step"]
        log(out, f"从断点继续：第 {step} 步")
    last_ck, t0 = time.time(), time.time()
    m.train()
    while step < total:
        ep, i = divmod(step, steps_per_epoch)
        order = list(range(len(seqs)))
        random.Random(args.seed * 100 + ep).shuffle(order)
        idx = order[i * args.micro_bsz:(i + 1) * args.micro_bsz]
        n = max(len(seqs[j][0]) for j in idx)
        ids = torch.zeros(len(idx), n, dtype=torch.long)
        anchors = torch.zeros(len(idx), args.anchors, dtype=torch.long)
        valid = torch.zeros(len(idx), args.anchors)
        for b, j in enumerate(idx):
            s, P = seqs[j]
            ids[b, :len(s)] = torch.tensor(s)
            hi = len(s) - 8                      # p+7 <= len-1
            cand = list(range(P, hi + 1))
            pick = rng.sample(cand, min(args.anchors, len(cand)))
            pick += [pick[0]] * (args.anchors - len(pick))
            anchors[b] = torch.tensor(pick)
            valid[b, :min(args.anchors, len(cand))] = 1
        ids, anchors, valid = ids.to(dev), anchors.to(dev), valid.to(dev)
        prev = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        mult = lr_mult(step, total, args.warmup, args.decay_frac)
        set_lr(opts, mult)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            with torch.no_grad():
                h, kv = L.backbone_train(m, ids, prev, return_kv=True)
                hp = h.gather(1, (anchors - 1).unsqueeze(-1).expand(-1, -1, h.size(-1)))
                top2 = m.lm_head(hp).topk(2, dim=-1).indices
                d_rej = top2[..., 1]             # 用主干第二候选模拟“被拒的草稿词”
                use_delta = torch.rand(anchors.shape, device=dev) < 0.5
            h, kv = h.detach(), {k: (a.detach(), b.detach()) for k, (a, b) in kv.items()}
            logits = L.draft_train(m, h, kv, ids, anchors, d_rej, use_delta)
            tgt = ids.gather(1, (anchors.unsqueeze(-1) + torch.arange(1, 8, device=dev)).view(len(idx), -1))
            tgt = tgt.view(len(idx), args.anchors, 7)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
            ce = ce.view(len(idx), args.anchors, 7)
            loss = (ce * valid.unsqueeze(-1)).sum() / (valid.sum() * 7)
        loss.backward()
        if not math.isfinite(loss.item()):
            log(out, f"[停止] draft 第 {step} 步 loss = {loss.item()}")
            sys.exit(3)
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], args.clip)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == total:
            with torch.no_grad():
                acc = ((logits.argmax(-1) == tgt).float() * valid.unsqueeze(-1)).sum((0, 1)) / valid.sum()
                # 期望接受长度：草稿第 k 位要连续都对才被接受
                run = ((logits.argmax(-1) == tgt).long().cumprod(-1).sum(-1).float() * valid).sum() / valid.sum()
            log(out, f"draft step {step}/{total} loss {loss.item():.4f} 每位准确率 "
                     f"{' '.join(f'{a:.2f}' for a in acc.tolist())} 估计τ≈{1 + run.item():.2f} "
                     f"用时 {(time.time() - t0) / 60:.1f}min",
                stage="draft", step=step, loss=loss.item(), tau_est=1 + run.item(), acc=acc.tolist())
        if time.time() - last_ck > args.ckpt_min * 60 or step == total:
            save_ckpt(ck_path, {"model": m.state_dict(), "opts": [o.state_dict() for o in opts], "step": step})
            last_ck = time.time()
    save_ckpt(out / "draft_final.pt", {"model": m.state_dict(), "args": vars(args)})
    log(out, f"[draft] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟")


def apply_config(args):
    if args.mix == "v2":
        MIX.clear()
        MIX.update(MIX_V2)
        VAL_SRCS[:] = ["oci", "ann", "finemath", "omi2", "fwedu"]
    SFT_FILES[:] = [x for x in args.sft_files.split(",") if x]


def gpu_worker(rank, args, shm_name, bar):
    global RANK
    RANK = rank
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)  # 必须在第一次碰 CUDA 之前设
    # 两个进程同时 compile 共用一个缓存目录会撞文件（Windows 上实测 FileNotFoundError）
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / f"torchinductor_rank{rank}")
    apply_config(args)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sync = MultiGpuSync(rank, args.gpus, shm_name, bar)
    try:
        run_pre_or_sft(args, torch.device("cuda:0"), sync)
    finally:
        sync.close()


def run_multi_gpu(args):
    ctx = mp.get_context("spawn")
    bar = ctx.Barrier(args.gpus)
    shm = shared_memory.SharedMemory(create=True, size=(4 * MAX_PARAMS + 8 * CTRL_SLOTS) * args.gpus)
    procs = [ctx.Process(target=gpu_worker, args=(r, args, shm.name, bar), daemon=True) for r in range(args.gpus)]
    code = 0
    try:
        for p in procs:
            p.start()
        while any(p.is_alive() for p in procs):  # 任何一张卡退出异常就全部停掉，避免另一张卡空等扣费
            bad = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            if bad:
                code = bad[0]
                break
            time.sleep(2)
        code = code or next((p.exitcode for p in procs if p.exitcode), 0)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        shm.close()
        shm.unlink()
    if code:
        sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pre", "sft", "rft", "gen", "draft"])
    ap.add_argument("--arch", default="v1", choices=["v1", "v3"], help="v3 = 16 层 + 数字槽位 + 3 层草稿头")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init", default="", help="sft/gen/draft：起点权重")
    ap.add_argument("--tokens", type=float, default=1.8e9)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-bsz", type=int, default=16)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-adam", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-min", type=float, default=20)
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--gen-bsz", type=int, default=128)
    ap.add_argument("--gen-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--small", action="store_true", help="CPU 冒烟测试用的小模型")
    ap.add_argument("--mix", default="v1", choices=["v1", "v2"], help="v2 = 数学+编程配比（data/packed_v2）")
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--n-q", type=int, default=12)
    ap.add_argument("--n-kv", type=int, default=4)
    ap.add_argument("--sft-files", default="sft.jsonl", help="逗号分隔，相对 data-dir")
    ap.add_argument("--cpu-amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpus", type=int, default=1, help="pre/sft 用几张卡（共享内存求平均梯度）")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.gpus > 1 and args.stage in ("pre", "sft", "rft"):
        run_multi_gpu(args)
        return
    apply_config(args)
    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("NANO_CPU_THREADS", "1")))  # 机器忙时多线程反而慢 30 倍
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.stage in ("pre", "sft", "rft"):
        run_pre_or_sft(args, dev)
    elif args.stage == "gen":
        run_gen(args, dev)
    else:
        run_draft(args, dev)


if __name__ == "__main__":
    main()
