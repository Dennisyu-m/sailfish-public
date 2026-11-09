# Sailfish

Sailfish 是一个从零训练的代码和数学小模型，目标是低延迟生成。它的结构是以卷积层为主的主干，加一个用于投机解码的草稿头。

这是研究性质的发布。它速度快，但准确率一般，也不是通用助手。

## 模型

| | Sailfish-v3 | Sailfish-v1 |
|---|---|---|
| 主干 | 16 层，4 x [C, C, C, A] | 10 层，C A C C A C C A C A |
| 宽度 | d=1152，d_ff=3072，18 头，6 个 KV 头 | d=768，d_ff=2048，12 头，4 个 KV 头 |
| 参数 | 共 3.704 亿，活跃约 2.95 亿（其余是只查表的 bigram） | 约 7000 万 |
| 词表 | 16,384，数字逐位切分 | 8,192 |
| 草稿头 | 3 层，每块 8 个 token | 每块 8 个 token |
| 上下文 | 1,024 token（提示词加输出） | 1,024 token |
| 预训练 token | 55 亿 | 18 亿 |

C 是卷积层，A 是注意力层。

## 成绩

以下都是实测，贪心解码，每题只生成一次。

| 评测 | Sailfish-v3 | Sailfish-v1 |
|---|---|---|
| HumanEval | 22.0% | 18.9% |
| HumanEval+ | 21.3% | 18.3% |
| GSM8K（1,319 题） | 16.98% | 未测 |
| MATH-500 | 8.0% | 未测 |
| MBPP | 3.2% | 未测 |
| MBPP+ | 2.6% | 未测 |
| 速度（开草稿头） | 1,065 token/秒 | 1,617 token/秒 |
| 每轮生成 token 数（τ） | HumanEval 3.03，MBPP 2.41 | HumanEval 2.48 |

说明：

- **MBPP 分数低主要是提示词格式不匹配。** MBPP 的提示词没有函数头，而模型精调时学的是补全给定的函数头。在提示词后补上 `def 函数名(` 时，MBPP 是 25.7%，MBPP+ 是 22.5%。这是非标准的诊断结果，不算正式分数。
- **速度**：两个模型都在单张 NVIDIA RTX 5880 Ada（48GB，虚拟 GPU）上测试，使用 `torch.compile` 和 CUDA Graph，单条跑 HumanEval。v3 平均每题生成约 47 个 token。长上下文会更慢，没有测过。
- **没有做同尺寸对照。** 这些结果不能说明 Sailfish 比同尺寸的普通模型更快或更强。
- float32 下投机解码和逐词解码在 HumanEval 上 164/164 一致；BF16 下因数值误差，批量生成时为 147/164。

## v3 训练过程

硬件：2 张 NVIDIA RTX 5880 Ada（每张 48GB），Windows。全流程 22.93 小时。

| 阶段 | 用时 | 内容 |
|---|---|---|
| 预训练 | 19.46 小时 | 55 亿 token，10,490 步，约 7.9 万 token/秒，双卡 |
| 监督精调 | 1.32 小时 | 2 轮，12,372 步，单卡 |
| RFT 采样 | 0.93 小时 | 每题采样 8 次，温度 0.8，保留做对的 |
| RFT 精调 | 4 分钟 | 25,256 条正确样本，低学习率 1 轮 |
| 草稿头训练 | 0.87 小时 | 含生成它的训练数据 |

优化器是 Muon 加 AdamW，使用 BF16 autocast、`torch.compile` 和 warmup-stable-decay 学习率。双卡通过共享内存同步，并定期校验两卡参数一致。

预训练数据配比：

| 领域 | 来源 |
|---|---|
| 编程 41.4% | OpenCodeInstruct 27.7%、opc-annealing-corpus 8.6%、cosmopedia 3.2%、opc-sft-stage2 1.1%、Magicoder 0.8% |
| 数学 47.9% | FineMath 26.6%、OpenMathInstruct-2 13.9%、NuminaMath-CoT 5.3%、MetaMathQA 1.6%、Orca-Math 0.5% |
| 英文 10.7% | FineWeb-Edu |

精调还用到了通过 DeepSeek API 生成并校验过的题目（`scripts/gen_deepseek.py`、`scripts/check_gen.py`）。训练数据按全等、前缀和 13 词片段重合去除了评测题。

## 使用

需要 Python、PyTorch 和 `tokenizers`；跑评测还需要 `numpy`、`pyarrow` 和 `evalplus`。

```bash
python scripts/chat.py
python scripts/chat.py --ckpt weights/v1/model.pt
```

聊天脚本在 CPU 上运行，v3 约 12 token/秒，只能一问一答，不记得上一句。英文的编程和数学题效果最好。

## 复现

| 脚本 | 用途 |
|---|---|
| `scripts/prep_data.py`、`prep_math.py`、`dedup.py`、`build_sft.py`、`select_sft_math.py`、`pack_v2.py`、`train_tokenizer_v2.py` | 清洗和打包数据 |
| `scripts/run_5880.py` | 运行全流程 |
| `scripts/train.py` | 各训练阶段 |
| `scripts/rft_sample.py` | RFT 采样 |
| `scripts/eval_humaneval.py`、`eval_math.py` | 评测和测速 |
| `scripts/test_*.py` | 解码正确性和训练路径等价性测试 |

打包后的数据不包含在内。

## 数据许可

以下许可证来自各数据集的 Hugging Face 页面。各数据集仍遵循各自的条款；CC-BY 和 ODC-By 要求署名，本节即为署名。

| 数据集 | 许可证 |
|---|---|
| nvidia/OpenCodeInstruct | CC-BY-4.0 |
| nvidia/OpenMathInstruct-2 | CC-BY-4.0 |
| OpenCoder-LLM/opc-annealing-corpus | ODC-By |
| OpenCoder-LLM/opc-sft-stage2 | MIT |
| ise-uiuc/Magicoder-OSS-Instruct-75K | MIT |
| HuggingFaceTB/cosmopedia | Apache-2.0 |
| HuggingFaceTB/finemath | ODC-By |
| HuggingFaceFW/fineweb-edu | ODC-By |
| project-numina/NuminaMath-CoT | Apache-2.0 |
| meta-math/MetaMathQA | MIT |
| microsoft/orca-math-word-problems-200k | MIT |
| openai/gsm8k, EleutherAI/hendrycks_math, HuggingFaceH4/MATH-500 | MIT |

## 权重

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `weights/v3/model.pt` | 1.38 GB | `1f71714792853e2a1d2c6f7b143cf25f02b615dd23e9ca0ba29504b45071ea40` |
| `weights/v1/model.pt` | 0.50 GB | `a42b52435611d5a9288dbdd0d624e62c55fad56a5965a4ec9e27e336f1cd04cf` |

## 许可证

代码和模型权重使用 [Apache License 2.0](LICENSE)。训练数据不在此许可证范围内。
