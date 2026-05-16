# Part 5 — Alignment and Reasoning RL

Version 1.0.2 — Spring 2025

---

## Section 1: Overview

### What This Project Implements

1. **Zero-shot prompting baseline** for the MATH dataset of competition math problems (Hendrycks et al., 2021)
2. **Supervised Fine-Tuning (SFT)**, given reasoning traces from a stronger reasoning model (DeepSeek R1, DeepSeek-AI et al., 2025)
3. **Expert Iteration** for improving reasoning performance with verified rewards
4. **Group-Relative Policy Optimization (GRPO)** for improving reasoning performance with verified rewards

> An optional supplement on aligning language models to human preferences is covered in the supplement PDF.

### What This Project Runs

1. Measure **Qwen 2.5 Math 1.5B** zero-shot prompting performance (baseline)
2. Run **SFT** on Qwen 2.5 Math 1.5B with reasoning traces from R1
3. Run **Expert Iteration** on Qwen 2.5 Math 1.5B with verified rewards
4. Run **GRPO** on Qwen 2.5 Math 1.5B with verified rewards

### Code Structure

| Path | Description |
|------|-------------|
| `cs336_alignment/` | Main implementation directory |
| `cs336_alignment/prompts/` | Prompt text files (provided to avoid copy-paste errors from PDF) |
| `tests/*.py` | Test suite — must pass `tests/test_sft.py` and `tests/test_grpo.py` |
| `tests/adapters.py` | Adapter hooks connecting implementation to tests |
| `README.md` | Environment setup instructions |

**Required tests:** `test_sft.py` and `test_grpo.py`. Other test files cover the optional supplement.

### Constraints

- Build most RL-related components **from scratch**
- **Allowed:** vLLM for text generation (§3.1); HuggingFace Transformers for loading Qwen 2.5 Math 1.5B and running forward passes (§4.1)
- **Not allowed:** HuggingFace training utilities (e.g., the `Trainer` class)

---

## Section 2: Reasoning with Language Models

### 2.1 Motivation

One of the remarkable use cases of language models is building generalist systems that can handle a wide range of natural language processing tasks. This project focuses on a developing use case: **mathematical reasoning**. It serves as a testbed for setting up evaluations, performing supervised fine-tuning, and experimenting with teaching LMs to reason using reinforcement learning (RL).

Two key differences from prior parts of this series:

- **Model:** Rather than using the language model codebase and models from earlier parts, this project switches to a modern, high-performance model — **Qwen 2.5 Math 1.5B Base** — because earlier trained models are too weak to display non-trivial mathematical reasoning capabilities.
- **Evaluation:** Rather than using cross-entropy as a surrogate metric, this project evaluates directly on downstream task performance. The benchmark is the **MATH 12K dataset** (Hendrycks et al., 2021), consisting of challenging high-school competition mathematics problems. Model outputs are evaluated by comparing against reference answers.

### 2.2 Chain-of-Thought Reasoning and Reasoning RL

#### Chain-of-thought reasoning with LLMs

Early chain-of-thought approaches fine-tuned language models on simple mathematical tasks using a "scratchpad" to break problems into intermediate steps (Nye et al., 2021). Later work prompts a strong model to "think step by step" before answering, significantly improving performance on mathematical reasoning tasks (Wei et al., 2023).

#### Learning to reason with Expert Iteration

The Self-Taught Reasoner (STaR, Zelikman et al., 2022) frames reasoning as a bootstrapping loop: a pretrained model samples diverse chains-of-thought (CoTs), keeps only those leading to correct answers, and fine-tunes on these "expert" traces. Iterating this cycle improves reasoning capabilities and solve rate. STaR demonstrated that this form of expert iteration (Anthony et al., 2017), using automatic string-match verification, can bootstrap reasoning skills without human-written traces.

#### Reasoning RL with verified rewards

Recent work uses policy gradient methods with verified rewards to improve reasoning performance. OpenAI's o1/o3/o4 (OpenAI et al., 2024), DeepSeek R1 (DeepSeek-AI et al., 2025), and Moonshot's kimi k1.5 (Team et al., 2025) train on math and code tasks where string matching or unit tests verify correctness, demonstrating remarkable improvements in competition math and coding. Follow-up work — Open-R1 (Face, 2025), SimpleRL-Zoo (Zeng et al., 2025), TinyZero (Pan et al., 2025) — confirms that pure RL with verified rewards, even on 1.5B parameter models, can improve reasoning performance.

### 2.3 Model and Dataset

| | |
|---|---|
| **Model** | Qwen 2.5 Math 1.5B Base — continually pretrained from Qwen 2.5 1.5B on high-quality synthetic math data (Yang et al., 2024) |
| **Dataset** | MATH 12K (Hendrycks et al., 2021) — challenging high-school competition math problems |
| **Cluster path** | `/data/a5-alignment/MATH` |

> **Alternative open-source datasets** (if MATH is unavailable due to copyright):
> - **Countdown** (Pan et al., 2025) — synthetic task based on the British TV show; popular small-scale reasoning RL testbed
> - **GSM8K** (Cobbe et al., 2021) — grade-school math problems; easier than MATH, useful for debugging the RL pipeline
> - **Tulu 3 SFT Math** (Lambert et al., 2025) — synthetic problems generated with GPT-4o and Claude 3.5 Sonnet (some answers may be imperfect)
>
> To extract short ground-truth labels (e.g., `1/2`) when not provided directly, use a math answer parser such as [Math-Verify](https://github.com/huggingface/Math-Verify).