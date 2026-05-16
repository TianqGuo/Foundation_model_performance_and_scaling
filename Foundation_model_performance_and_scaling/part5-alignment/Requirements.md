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