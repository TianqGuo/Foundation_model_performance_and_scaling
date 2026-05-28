# Part 6 — Instruction Tuning & RLHF

Building a generalist dialogue system from Llama 3.1 8B Base through a three-stage
post-training pipeline: zero-shot evaluation → supervised fine-tuning on instruction
data → Direct Preference Optimization (DPO) on human preference pairs.

---

## Section 1 — Overview

This project turns a raw pre-trained language model into an instruction-following,
safety-aware dialogue assistant using modern post-training methods. Starting from
Llama 3.1 8B Base (no chat template, no RLHF), the pipeline applies increasingly
powerful alignment techniques and measures their effect on four diverse benchmarks.

**Model:** [Llama 3.1 8B Base](https://huggingface.co/meta-llama/Llama-3.1-8B) — a
strong 8B-parameter model that has received no instruction tuning or safety training.

**Benchmarks:**
| Benchmark | Measures |
|-----------|---------|
| [MMLU](https://arxiv.org/abs/2009.03300) | Factual knowledge across 57 academic subjects |
| [GSM8K](https://arxiv.org/abs/2110.14168) | Multi-step grade-school math reasoning |
| [AlpacaEval](https://arxiv.org/abs/2305.14387) | Instruction-following quality (pairwise win rate vs GPT-3.5) |
| [SimpleSafetyTests](https://arxiv.org/abs/2311.08370) | Safety — proportion of outputs judged safe by a strong annotator |

**Pipeline:**

| Section | Method | Starting point | Goal |
|---------|--------|---------------|------|
| §2 | Zero-shot baselines | Base model | Measure all 4 benchmarks before any training |
| §3 | Supervised fine-tuning (SFT) | Base model | Teach instruction following from demonstration data |
| §4 | Evaluate SFT model | SFT checkpoint | Direct comparison against §2 on all 4 benchmarks |
| §5 | DPO | SFT checkpoint | Fine-tune on human preference pairs (Anthropic HH) |

---

## Section 2 — Zero-Shot Baselines

Evaluates Llama 3.1 8B Base zero-shot on all four benchmarks using greedy decoding
(temperature 0.0, top-p 1.0). These numbers establish the baseline every downstream
model is compared against. All four evals run via vLLM; AlpacaEval and
SimpleSafetyTests use a Llama 3.3 70B Instruct judge running on 2× A100-SXM4-80GB.

```bash
cd cs336_alignment/section2_zero_shot
./part_6_2.sh                      # full run (all 4 benchmarks)
./part_6_2.sh --smoke-test         # 20-example quick pass (local GPU)
./part_6_2.sh --skip-mmlu          # skip MMLU if already done
./part_6_2.sh --skip-gsm8k         # skip GSM8K if already done
./part_6_2.sh --plot               # regenerate plots from existing results
```

**Output files:**
- `results/section2/eval_mmlu_baseline.jsonl` / `.summary.json` — per-example MMLU predictions and summary metrics
- `results/section2/eval_gsm8k_baseline.jsonl` / `.summary.json` — per-example GSM8K predictions and summary metrics
- `results/section2/alpaca_eval_baseline.json` — AlpacaEval outputs (JSON array)
- `results/section2/leaderboard.csv` — AlpacaEval win rate and LC win rate
- `results/section2/sst_baseline.jsonl` — raw SST generations
- `results/section2/sst_baseline_annotated.jsonl` — SST with safety annotations
- `results/section2/baseline_accuracy_bar.png` — MMLU and GSM8K accuracy bar chart
- `results/section2/mmlu_subject_accuracy_baseline.png` — per-subject MMLU breakdown
- `results/section2/sst_safety_by_category.png` — SST safety rate by harm area

### Summary Results

| Benchmark | Metric | Zero-shot baseline | SFT | DPO |
|-----------|--------|--------------------|-----|-----|
| MMLU | Accuracy | **53.5%** | — | — |
| GSM8K | Accuracy | **16.3%** | — | — |
| AlpacaEval | Win rate vs text-davinci-003 | **37.8%** | — | — |
| AlpacaEval | Length-controlled win rate | **31.6%** | — | — |
| SimpleSafetyTests | % Safe outputs | **66.0%** | — | — |

*(SFT and DPO columns filled in after §3–§5.)*

---

### §2.1 — MMLU

- **Accuracy: 53.5%** (7,515 / 14,042 correct)
- **Throughput: 41.9 examples/second** (335 s for 14,042 examples)
- **Parse failures: 10** (0.07%) — negligible

The base model scores comfortably above the 25% random baseline but well below the
65–70% range achieved by well-tuned models of the same size. Humanities and social
science subjects (Marketing, US Foreign Policy, International Law) score highest,
often above 80%; quantitative and formal reasoning subjects (Abstract Algebra, Formal
Logic, College Mathematics) score lowest, several below 35%. This reflects the
pretraining distribution — factual recall is well-covered, multi-step formal reasoning
is not.

**MMLU accuracy (MMLU vs GSM8K) and per-subject breakdown:**

![Zero-shot benchmark accuracy](results/section2/baseline_accuracy_bar.png)

![MMLU per-subject accuracy](results/section2/mmlu_subject_accuracy_baseline.png)

---

### §2.2 — GSM8K

- **Accuracy: 16.3%** (215 / 1,319 correct)
- **Throughput: 25.3 examples/second** (52 s for 1,319 examples)
- **Parse failures: 6** (0.46%) — model answered in written-out language with no Arabic digits

16.3% is a very weak result and precisely what is expected from an unaligned base
model on multi-step arithmetic reasoning. GSM8K requires planning a solution path and
executing several arithmetic steps correctly — a capability that emerges strongly after
fine-tuning on reasoning traces but is unreliable without it. The model solves simple
one- or two-step problems but fails the majority of problems requiring three or more
arithmetic operations.

Dominant error patterns: arithmetic slips in multi-step chains, stopping at an
intermediate result rather than the final answer, and misreading which entity has
which property (age, price, weight).

---

### §2.3 — AlpacaEval

- **Win rate: 37.8%** (304 wins / 804 comparisons vs text-davinci-003)
- **Length-controlled win rate: 31.6%**
- Annotator: Llama 3.3 70B Instruct; reference: text-davinci-003 (GPT-3.5 class)

The base model is preferred less than 40% of the time. The LC win rate (31.6%) is
lower than the raw win rate (37.8%), confirming that some apparent wins are driven
by response length rather than quality. The base model's main weaknesses are lack
of structured formatting (numbered lists, headers), verbose preambles before
answering, and format drift on longer outputs — all characteristics that instruction
tuning directly targets.

---

### §2.4 — SimpleSafetyTests

- **Overall: 66% safe** (66 / 100 examples judged safe by Llama 3.3 70B Instruct)

| Harm Area | Safety Rate |
|-----------|-------------|
| Suicide, Self-Harm, and Eating Disorders | **100%** |
| Illegal and Highly Regulated Items | **75%** |
| Scams and Fraud | **70%** |
| Physical Harm and Violence | **55%** |
| Child Safety | **30%** ⚠️ |

The 34 unsafe responses split into two failure modes: ~15 cases of direct compliance
(the model generates the harmful content with no refusal) and ~19 cases of soft
refusal (a disclaimer followed by partial engagement with the harmful premise). Child
Safety scores worst because harmful requests are often framed as benign professional
contexts the model has no refusal training to recognise. Suicide/Self-Harm scores
best because safe-messaging content is abundant in pretraining data.

**Safety rate by harm category:**

![SST safety by category](results/section2/sst_safety_by_category.png)

---

## Section 3 — Supervised Fine-Tuning *(in progress)*

Fine-tunes Llama 3.1 8B on a safety-augmented mix of UltraChat-200K instruction data
using packed sequence training (Alpaca prompt format, `seq_length=512`, effective
batch size 32 via gradient accumulation).

---

## Section 4 — SFT Model Evaluation *(pending §3)*

Evaluates the SFT checkpoint on all four benchmarks using the same prompts and
generation settings as §2, enabling direct apples-to-apples comparison.

---

## Section 5 — Direct Preference Optimization *(pending §3)*

Fine-tunes the SFT model on Anthropic HH preference pairs using the DPO objective —
no reward model, no RL loop. Tracks implicit reward accuracy (proportion of
preference pairs where the model assigns higher log-probability to the chosen response)
as the validation metric.

---

## Repository Layout

```
part6-instruction-tuning-rlhf/
├── cs336_alignment/
│   ├── prompts/                    # zero_shot_system_prompt.prompt, alpaca_sft.prompt
│   └── section2_zero_shot/
│       ├── evaluate_mmlu.py        # vLLM MMLU evaluation
│       ├── evaluate_gsm8k.py       # vLLM GSM8K evaluation
│       ├── evaluate_alpaca.py      # vLLM AlpacaEval generation
│       ├── evaluate_safety.py      # vLLM SimpleSafetyTests generation
│       ├── plot_zero_shot_results.py
│       └── part_6_2.sh
├── scripts/
│   ├── alpaca_eval_vllm_llama3_3_70b_fn/  # AlpacaEval annotator config (Llama 3.3 70B)
│   └── evaluate_safety.py          # SST safety annotation script
├── data/                           # Datasets (gitignored)
│   ├── mmlu/                       # MMLU test CSVs
│   ├── gsm8k/
│   ├── alpaca_eval/
│   └── simple_safety_tests/
├── assets/                         # Model checkpoints (gitignored)
│   ├── Llama-3.1-8B/
│   └── Llama-3.3-70B-Instruct/
├── results/
│   └── section2/                   # eval JSONLs, leaderboard.csv, plots, discussion.md
├── tests/
│   ├── adapters.py
│   ├── test_data.py
│   ├── test_dpo.py
│   ├── test_metrics.py
│   └── test_sft.py
├── get_assets.sh                   # Model download / cluster-symlink helper
├── Requirements.md                 # Full technical spec for all sections
└── pyproject.toml
```

---

## Setup

```bash
uv sync
```

**Model assets** (auto-downloaded by `part_6_2.sh`, or run manually):

```bash
bash get_assets.sh
```

On the cluster, models are pre-cached at `/data/a5-alignment/models/` — the script
symlinks them automatically before downloading.

**Open-file limit** (required for large parallel HuggingFace downloads):

```bash
ulimit -n 65536
```

The shell script sets this automatically.

---

## Running Tests

```bash
uv run pytest -v                          # all tests
uv run pytest -k test_parse_mmlu_response # single test
```

---

## References

**Benchmarks**
- Hendrycks et al., 2021 — [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300)
- Cobbe et al., 2021 — [Training Verifiers to Solve Math Word Problems (GSM8K)](https://arxiv.org/abs/2110.14168)
- Li et al., 2023 — [AlpacaEval: An Automatic Evaluator of Instruction-Following Models](https://arxiv.org/abs/2305.14387)
- Vidgen et al., 2024 — [SimpleSafetyTests: A Test Suite for Identifying Critical Safety Risks in Large Language Models](https://arxiv.org/abs/2311.08370)

**Base Model**
- Meta AI, 2024 — [Llama 3.1: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)

**Instruction Tuning**
- Taori et al., 2023 — [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- Ding et al., 2023 — [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations (UltraChat)](https://arxiv.org/abs/2305.14233)

**RLHF & DPO**
- Ouyang et al., 2022 — [Training Language Models to Follow Instructions with Human Feedback (InstructGPT/RLHF)](https://arxiv.org/abs/2203.02155)
- Rafailov et al., 2023 — [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/abs/2305.18290)
- Bai et al., 2022 — [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (Anthropic HH)](https://arxiv.org/abs/2204.05862)
- Ganguli et al., 2022 — [Red Teaming Language Models to Reduce Harms](https://arxiv.org/abs/2209.07858)

**Infrastructure**
- Kwon et al., 2023 — [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- Stanford CS336 Spring 2025 — [Language Models from Scratch, Part 6 (project specification)](https://github.com/stanford-cs336/assignment5-alignment)
