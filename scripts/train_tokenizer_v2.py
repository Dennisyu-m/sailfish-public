"""
v2 分词器：字节级 BPE，词表 16384，数字逐位切（每个 0～9 单独一个词，LLaMA / DeepSeek 做法）。
  - id 0 = <|endoftext|>（文档分隔兼 BOS）；任何输入都能无损编码。
  - 预切分：先 Digits(individual_digits=True)，再 ByteLevel 正则。所以 "12345" -> 1 2 3 4 5，" 12" -> Ġ 1 2。
训练语料按 MIX 比例抽样（编程约 45%、数学约 46%、英文约 9%）：
  - 编程：data/clean/<src>/*.parquet + data/clean/_keep（第一轮去重结果）
  - 数学/英文：data/clean2/<src>/*.parquet（text 列，prep_math.py 产出；不存在的源自动跳过）
"""
import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
CLEAN = Path(os.environ.get("NANO_CLEAN", ROOT / "data" / "clean"))
CLEAN2 = Path(os.environ.get("NANO_CLEAN2", ROOT / "data" / "clean2"))
MIX = {
    # 编程（第一轮数据）
    "oci": 0.25, "ann": 0.15, "cosmo": 0.03, "sft2": 0.01, "mago": 0.01,
    # 数学
    "finemath": 0.25, "omi2": 0.12, "numina": 0.06, "metamath": 0.02, "orca": 0.01,
    # 英文
    "fwedu": 0.09,
}


def iter_source(src, rng):
    if (CLEAN / src).exists():
        files = sorted(glob.glob(str(CLEAN / src / "*.parquet")))
        rng.shuffle(files)
        for f in files:
            keep = np.load(CLEAN / "_keep" / src / (Path(f).stem + ".npy"))
            texts = pq.read_table(f, columns=["text"])["text"].to_pylist()
            idx = [i for i in range(len(texts)) if keep[i]]
            rng.shuffle(idx)
            for i in idx:
                yield texts[i]
    else:
        files = sorted(glob.glob(str(CLEAN2 / src / "*.parquet")))
        rng.shuffle(files)
        for f in files:
            texts = pq.read_table(f, columns=["text"])["text"].to_pylist()
            rng.shuffle(texts)
            yield from texts


def sample_texts(budget_chars, seed=0):
    rng = random.Random(seed)
    for src, frac in MIX.items():
        want, got = budget_chars * frac, 0
        for t in iter_source(src, rng):
            yield t
            got += len(t)
            if got >= want:
                break
        print(f"  {src}: 抽样 {got / 1e6:.0f}M 字符（目标 {want / 1e6:.0f}M）", flush=True)


def build(vocab):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2,
                                  show_progress=False)
    return tok, trainer


def check(tok, vocab):
    assert tok.token_to_id("<|endoftext|>") == 0 and tok.get_vocab_size() == vocab
    for s in ['def f(x):\n    """中文 docstring ✓ \t"""\n    return x ** 2  # 测试\x00\n',
              "x = 12345 + 678\n3.14159 * r ** 2, 1e-9, 2024-11-08",
              "$\\frac{1}{3} + \\sqrt{2}$ \\boxed{42}"]:
        assert tok.decode(tok.encode(s).ids) == s, f"往返不一致: {s!r}"
    # 数字必须逐位切
    for s in ["12345", " 678", "3.14159", "1000000"]:
        pieces = tok.encode(s).tokens
        digits = [p for p in pieces if any(c.isdigit() for c in p)]
        assert all(len(p.replace("Ġ", "")) == 1 for p in digits), f"数字没有逐位切: {s!r} -> {pieces}"
    demo = {s: tok.encode(s).tokens for s in ["x = 12345 + 678", "return 3.14159 * r ** 2", "\\frac{1}{3}"]}
    print("切分示例：", demo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=600, help="训练语料抽样字符数（百万）")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--out", default=str(ROOT / "data" / "tokenizer_v2.json"))
    args = ap.parse_args()
    tok, trainer = build(args.vocab)
    tok.train_from_iterator(sample_texts(args.mb * 1_000_000), trainer)
    check(tok, args.vocab)
    tok.save(args.out)
    print(f"已保存 {args.out}，词表 {tok.get_vocab_size()}，往返与逐位切检查通过")


if __name__ == "__main__":
    main()
