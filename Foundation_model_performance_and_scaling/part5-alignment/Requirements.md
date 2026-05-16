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

---

## Section 3: Measuring Zero-Shot MATH Performance

The first step is measuring the performance of the base language model on the 5K example test set of MATH. This baseline is useful for understanding how each subsequent approach affects model behavior.

**Prompt:** Unless otherwise specified, all MATH experiments use the following prompt from DeepSeek R1-Zero (DeepSeek-AI et al., 2025), referred to as the **r1_zero prompt** (file: `cs336_alignment/prompts/r1_zero.prompt`):

```
A conversation between User and Assistant. The User asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and then
provides the User with the answer. The reasoning process is enclosed within <think> </think>
and answer is enclosed within <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think> <answer> answer here </answer>.

User: {question}
Assistant: <think>
```

The model plays the role of the assistant, starting from the open `<think>` tag, generating its reasoning, closing with `</think>`, then producing a final symbolic answer within `<answer> </answer>` tags (e.g., `<answer> 4x + 10 </answer>`). The structured tags enable easy output parsing and allow stopping generation at `</answer>`.

> **Note on prompt choice:** The r1_zero prompt is not optimal for maximizing downstream RL performance due to a mismatch with how Qwen 2.5 Math 1.5B was pretrained. Liu et al. [2025] finds that prompting with the question only (`question_only` prompt) already starts at high accuracy. The r1_zero prompt is used here because RL with it shows clear accuracy improvements within a small number of steps, making it easier to verify correctness quickly. A direct comparison against the `question_only` prompt is included later.

### 3.1 Using vLLM for Offline Inference

Evaluating and training with RL requires high-performance batched inference. This project uses **vLLM** — a high-throughput, memory-efficient inference engine incorporating optimized CUDA kernels and PagedAttention (Kwon et al., 2023).

**Basic usage:**

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

llm = LLM(model=<path to model>)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

`LLM` can be initialized with a HuggingFace model name (auto-downloaded if not cached locally) or a local path.

**Pre-downloaded models on the Together cluster** (do not re-download):

| Model | Path |
|-------|------|
| Qwen 2.5 Math 1.5B Base | `/data/a5-alignment/models/Qwen2.5-Math-1.5B` |
| Llama 3.1 8B Base | `/data/a5-alignment/models/Llama-3.1-8B` |
| Llama 3.3 70B Instruct | `/data/a5-alignment/models/Llama-3.3-70B-Instruct` |

### 3.2 Zero-Shot MATH Baseline

**Prompting setup:** Load MATH examples and format each question using the r1_zero prompt.

**Evaluation metric:** Math evaluation cannot use exact string matching — the model may answer `<answer> 1/2 </answer>` for a ground truth of `0.5`. The answer parsing function takes the model's string output and the ground-truth answer, returning a boolean indicating correctness.

The reward function used is `cs336_alignment.drgrpo_grader.r1_zero_reward_fn` (from Liu et al., 2025). Use this for all MATH evaluation unless otherwise specified.

**Generation hyperparameters:**

```python
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
)
# Stop when the model completes its answer
# https://github.com/sail-sg/understand-r1-zero/blob/c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True
```

---

### Implementation: `math_baseline` (4 points)

**(a)** Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. The script should:

1. Load MATH validation examples from `/data/a5-alignment/MATH/validation.jsonl`
2. Format each example as a string prompt using the r1_zero prompt
3. Generate model outputs for each example
4. Calculate evaluation metrics
5. Serialize examples, model generations, and evaluation scores to disk for downstream analysis

It is recommended to implement a reusable `evaluate_vllm` method (will be reused in later sections):

```python
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """Evaluate a model on a list of prompts, compute metrics, and serialize results."""
```

**Output:** A script to evaluate baseline zero-shot MATH performance.

**(b)** Run the evaluation script on Qwen 2.5 Math 1.5B. Report how many model generations fall into each category:

1. Correct — format reward 1 and answer reward 1
2. Format reward 1, answer reward 0
3. Format reward 0 and answer reward 0

Examine at least 10 cases where format reward is 0: is the issue with the base model's output or the parser? Examine at least 10 cases where format reward is 1 but answer reward is 0: what is going wrong?

**Output:** Commentary on model and reward function performance, with examples from each category.

**(c)** How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?

**Output:** 1–2 sentences with evaluation metrics.