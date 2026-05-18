# Part 5 — Alignment and Reasoning RL: Qwen 2.5 Math 1.5B

Teaching a 1.5B-parameter math model to reason step-by-step using supervised
fine-tuning, Expert Iteration, and Group Relative Policy Optimization (GRPO)
with verified rewards on the MATH competition dataset.

---

## What This Project Does

Starting from a base model pretrained on math data, the pipeline progressively
improves mathematical reasoning:

1. **Measure** zero-shot baseline performance on MATH using the r1_zero prompt
2. **Fine-tune** on DeepSeek R1 chain-of-thought reasoning traces (SFT)
3. **Bootstrap** reasoning via Expert Iteration: rollout → filter correct → fine-tune
4. **Optimize** with GRPO: policy gradient with group-normalized verified rewards

---

## Section 3 — Zero-Shot Baseline

Evaluates Qwen 2.5 Math 1.5B zero-shot on 5K MATH validation examples using the
r1_zero prompt (`<think>...</think> <answer>...</answer>` format) and the DrGRPO
reward function. Serves as the baseline for all subsequent improvements.

**Key finding:** The base model defaults to `\boxed{}` format from math pretraining
rather than the r1_zero format, resulting in near-zero format compliance. This is
expected — RL training in later sections corrects it.

| Script | What it runs |
|--------|-------------|
| `part_5_3.sh` | vLLM batch inference on MATH validation set |

```bash
cd cs336_alignment/section3_zero_shot
./part_5_3.sh                        # full evaluation on cluster (5K examples)
./part_5_3.sh --max_examples 10      # smoke test (local, falls back to GSM8K)
```

**Output:** `results/section3/zero_shot_eval.jsonl` — one JSON record per example  
**Analysis:** `results/section3/zero_shot_analysis.md` — category breakdown and root cause

### Results

**Local smoke test (10 examples, GSM8K fallback):**

| Category | Count | % |
|---|---|---|
| Correct (format=1, answer=1) | 0 | 0% |
| Format ok, wrong answer (format=1, answer=0) | 1 | 10% |
| No format (format=0) | 9 | 90% |

**Full MATH validation (5K examples, cluster):** pending cluster run.

The 0% accuracy is expected — the base model defaults to `\boxed{}` format from
math pretraining rather than the r1_zero `<answer>` tags. Several responses
contain the correct numerical answer inside `\boxed{}` but are penalised on format.
Two additional failures are parser-strictness edge cases (`\n` vs space between
`</think>` and `<answer>`). See `results/section3/zero_shot_analysis.md` for the
full breakdown.

---

## Section 4 — Supervised Fine-Tuning

Fine-tunes Qwen 2.5 Math 1.5B on DeepSeek R1 reasoning traces from the MATH
dataset. Implements core SFT primitives from scratch (tokenization, log-prob
computation, masked loss, gradient accumulation) then runs a training experiment
varying dataset size and data quality filtering.

### SFT Primitives (`helpers.py`)

| Function | Description |
|----------|-------------|
| `tokenize_prompt_and_output` | Tokenize prompt+response pairs; build response mask |
| `compute_entropy` | Per-token entropy of next-token predictions |
| `get_response_log_probs` | Per-token log-probs from a causal LM; optional entropy |
| `masked_normalize` | Masked sum with constant normalization |
| `sft_microbatch_train_step` | NLL loss, backward pass, gradient accumulation scaling |
| `log_generations` | vLLM rollout + reward + entropy diagnostics for in-loop logging |

### Training Experiment

Runs six experiments sequentially (dataset size ablation + filtered data):

| Run | Training examples | Notes |
|-----|------------------|-------|
| `sft_n128` | 128 | |
| `sft_n256` | 256 | |
| `sft_n512` | 512 | |
| `sft_n1024` | 1024 | |
| `sft_full` | Full dataset | Target: ≥ 15% validation accuracy |
| `sft_filtered` | Full, correct-answer only | Filtered by r1_zero reward function |

```bash
cd cs336_alignment/section4_sft
./part_5_4.sh                        # helper tests + all 6 training runs (cluster)
./part_5_4.sh --tests-only           # helper tests only (CPU, works locally)
./part_5_4.sh --train-only           # training only, skip tests
```

Training uses 2 GPUs: policy on `cuda:0`, vLLM evaluator on `cuda:1`. Evaluates
on MATH validation every 50 optimizer steps and logs accuracy curves to wandb.

Default hyperparameters: `lr=2e-5`, `micro_batch_size=2`, `gradient_accumulation_steps=32`
(effective batch size = 64). To tune:

```bash
uv run python cs336_alignment/section4_sft/train_sft.py \
    --lr 5e-5 --gradient_accumulation_steps 16 --run_name sft_full_lr5e5
```

**Generate plots after training:**

```bash
uv run python cs336_alignment/section4_sft/plot_sft_results.py
```

Reads all `eval_metrics_*.jsonl` files in `results/section4/` and saves:
- `results/section4/sft_ablation_accuracy.png` — accuracy vs step for each dataset size
- `results/section4/sft_filtered_comparison.png` — full vs filtered dataset comparison

**Output files:**
- `results/section4/dataset_info.json` — training set size per run
- `results/section4/eval_metrics_{run_name}.jsonl` — per-step accuracy/entropy/length (appended live)
- `results/section4/final_eval.json` — final validation accuracy
- `results/section4/sft_ablation_accuracy.png` — accuracy curves (after plotting)
- `results/section4/sft_filtered_comparison.png` — filter comparison (after plotting)
- `/data/$USER/sft_n{size}/` — model checkpoints (cluster)

### Results

**Accuracy curves (cluster run):** pending.

**Final validation accuracy (target ≥ 15% on full dataset):** pending.

---

## Repository Layout

```
part5-alignment/
├── cs336_alignment/
│   ├── prompts/                    # r1_zero.prompt, question_only.prompt
│   ├── section3_zero_shot/
│   │   ├── evaluate_zero_shot.py   # vLLM batch evaluation script
│   │   └── part_5_3.sh
│   ├── section4_sft/
│   │   ├── helpers.py              # SFT primitives (tokenize, loss, entropy)
│   │   ├── train_sft.py            # Full SFT training loop
│   │   ├── plot_sft_results.py     # Accuracy curve plots from eval_metrics_*.jsonl
│   │   └── part_5_4.sh
│   ├── section5_expert_iter/       # Expert Iteration (STaR) — coming next
│   └── section6_grpo/              # GRPO with verified rewards — coming next
├── data/                           # Datasets (gitignored)
│   ├── math/                       # MATH competition dataset
│   └── gsm8k/                      # GSM8K (local smoke-test fallback)
├── assets/                         # Downloaded model checkpoints (gitignored)
├── results/
│   ├── section3/                   # zero_shot_eval.jsonl, zero_shot_analysis.md
│   └── section4/                   # dataset_info.json, eval_metrics_*.jsonl, final_eval.json
├── tests/
│   ├── adapters.py                 # Connects implementations to test suite
│   ├── test_sft.py                 # Section 4 helper tests (10 tests)
│   └── test_grpo.py                # Section 6 helper tests
└── Requirements.md                 # Full technical spec for all sections
```

---

## Setup

```bash
uv sync --no-install-package flash-attn
uv sync
```

**First-time model download (local, ~3 GB):**

```bash
uv run huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
    --local-dir assets/Qwen2.5-Math-1.5B
```

Shell scripts auto-detect and download the model if not found locally or on the cluster.

**WSL2 CUDA:** If `torch.cuda.is_available()` returns False, set Lenovo Vantage
graphics mode to **Hybrid** (not integrated-only).

---

## Running Tests

```bash
uv run pytest -v                     # all tests
uv run pytest tests/test_sft.py -v  # Section 4 helpers only
```

Run from the repo root (`part5-alignment/`) so snapshot paths resolve correctly.

---

## References

- DeepSeek-AI et al., 2025 — [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Zelikman et al., 2022 — [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465)
- Shao et al., 2024 — [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Hendrycks et al., 2021 — [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
- Liu et al., 2025 — [Understanding R1-Zero-Like Training](https://github.com/sail-sg/understand-r1-zero)
- Kwon et al., 2023 — [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)