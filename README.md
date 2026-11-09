# Sailfish

Sailfish is a small code and math language model trained from scratch, designed for low-latency generation. It combines a convolution-heavy backbone with a draft head for speculative decoding.

It is a research release. It is fast, but its accuracy is modest and it is not a general assistant.

中文说明见 [README.zh.md](README.zh.md)，技术说明见 [TECHNICAL_NOTES.zh.md](TECHNICAL_NOTES.zh.md).

## Models

| | Sailfish-v3 | Sailfish-v1 |
|---|---|---|
| Backbone | 16 layers, 4 x [C, C, C, A] | 10 layers, C A C C A C C A C A |
| Width | d=1152, d_ff=3072, 18 heads, 6 KV heads | d=768, d_ff=2048, 12 heads, 4 KV heads |
| Parameters | 370.4M total, about 295M active (the rest is a lookup-only bigram table) | about 70M |
| Vocabulary | 16,384, digits tokenized one by one | 8,192 |
| Draft head | 3 layers, block size 8 | block size 8 |
| Context | 1,024 tokens (prompt + output) | 1,024 tokens |
| Pretraining tokens | 5.5B | 1.8B |

C is a convolution layer and A is an attention layer.

## Results

All numbers are measured. Greedy decoding, single attempt.

| Benchmark | Sailfish-v3 | Sailfish-v1 |
|---|---|---|
| HumanEval | 22.0% | 18.9% |
| HumanEval+ | 21.3% | 18.3% |
| GSM8K (1,319 problems) | 16.98% | not measured |
| MATH-500 | 8.0% | not measured |
| MBPP | 3.2% | not measured |
| MBPP+ | 2.6% | not measured |
| Throughput, draft head on | 1,065 tokens/s | 1,617 tokens/s |
| Tokens per decoding round (tau) | 3.03 on HumanEval, 2.41 on MBPP | 2.48 on HumanEval |

Notes:

- **MBPP is mostly a prompt-format mismatch.** MBPP prompts do not include a function signature, and the model was tuned to complete a given signature. Adding `def <name>(` to the prompt gives MBPP 25.7% and MBPP+ 22.5%. This is a non-standard diagnostic, not an official score.
- **Throughput** was measured for both models on one NVIDIA RTX 5880 Ada (48 GB, virtual GPU), using `torch.compile` and CUDA Graph with single-stream HumanEval prompts. v3 averages about 47 generated tokens per problem. Long contexts will be slower and were not measured.
- **No same-size baseline was run.** These results do not show that Sailfish is faster or stronger than a conventional model of the same size.
- Speculative and token-by-token decoding matched on 164/164 HumanEval problems in float32. In BF16, numerical differences cause 147/164 to match in batched generation.

## Training v3

Hardware: 2 x NVIDIA RTX 5880 Ada (48 GB each), Windows. Total pipeline time was 22.93 hours.

| Stage | Time | Details |
|---|---|---|
| Pretraining | 19.46 h | 5.5B tokens, 10,490 steps, about 79k tokens/s, both GPUs |
| Supervised fine-tuning | 1.32 h | 2 epochs, 12,372 steps, one GPU |
| RFT sampling | 0.93 h | 8 samples per problem at temperature 0.8, keeping correct answers |
| RFT fine-tuning | 4 min | 25,256 correct samples, 1 epoch at low learning rate |
| Draft-head training | 0.87 h | Includes generating its training data |

The optimizer is Muon plus AdamW, with BF16 autocast, `torch.compile`, and a warmup-stable-decay learning-rate schedule. Dual-GPU training uses shared memory with parameter verification between cards.

Pretraining mix:

| Domain | Sources |
|---|---|
| Code, 41.4% | OpenCodeInstruct 27.7%, opc-annealing-corpus 8.6%, cosmopedia 3.2%, opc-sft-stage2 1.1%, Magicoder 0.8% |
| Math, 47.9% | FineMath 26.6%, OpenMathInstruct-2 13.9%, NuminaMath-CoT 5.3%, MetaMathQA 1.6%, Orca-Math 0.5% |
| English, 10.7% | FineWeb-Edu |

Fine-tuning also uses checked problems generated through the DeepSeek API (`scripts/gen_deepseek.py`, `scripts/check_gen.py`). Evaluation problems were removed from the training data by exact match, prefix match and 13-word overlap.

## Usage

Python and PyTorch are required, along with `tokenizers`, `numpy`, `pyarrow` and `evalplus` for evaluation.

Download the weights from the [release page](https://github.com/Dennisyu-m/sailfish-public/releases/tag/v1.0). Save `sailfish-v3.pt` as `weights/v3/model.pt` and `sailfish-v1.pt` as `weights/v1/model.pt`.

```bash
python scripts/chat.py
python scripts/chat.py --ckpt weights/v1/model.pt
```

The chat script uses a GPU when one is available, at about 50 tokens/s for v3, and about 12 tokens/s on CPU. It generates token by token without the draft head, so it is much slower than the throughput above. It handles one question at a time, with no multi-turn memory. English coding and math prompts work best.

## Reproducing

| Script | Purpose |
|---|---|
| `scripts/prep_data.py`, `prep_math.py`, `dedup.py`, `build_sft.py`, `select_sft_math.py`, `pack_v2.py`, `train_tokenizer_v2.py` | Clean and pack the data |
| `scripts/run_5880.py` | Run the whole pipeline |
| `scripts/train.py` | Training stages |
| `scripts/rft_sample.py` | RFT sampling |
| `scripts/eval_humaneval.py`, `eval_math.py` | Evaluation and speed test |
| `scripts/test_*.py` | Decoder correctness and training-path equivalence tests |

The packed data is not included.

## Data licenses

These licenses are as listed on each dataset's Hugging Face page. Each dataset remains under its own terms, and CC-BY and ODC-By require attribution, which this section provides.

| Dataset | License |
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

## Weights

| File | Size | SHA-256 |
|---|---|---|
| `sailfish-v3.pt` | 1.38 GB | `1f71714792853e2a1d2c6f7b143cf25f02b615dd23e9ca0ba29504b45071ea40` |
| `sailfish-v1.pt` | 0.50 GB | `a42b52435611d5a9288dbdd0d624e62c55fad56a5965a4ec9e27e336f1cd04cf` |

## License

The code and model weights are released under the [Apache License 2.0](LICENSE). The training data is not covered by this license.
