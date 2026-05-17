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

---

## Section 4: Supervised Fine-Tuning for MATH

### Algorithm: SFT

```
Input: initial policy model π_θ_init; SFT dataset D
1: policy model π_θ ← π_θ_init
2: for step = 1, …, n_sft_steps do
3:     Sample a batch of question-response pairs D_b from D
4:     Compute the cross-entropy loss of the responses given the questions using model π_θ
5:     Update the model parameters θ by taking a gradient step w.r.t. the cross-entropy loss
6: end for
Output: π_θ
```

**Goal:** Fine-tune the base model on the MATH dataset to generate chain-of-thought reasoning traces followed by answers, rather than directly predicting answers. The reasoning SFT dataset (from DeepSeek R1) is at `/data/a5-alignment/MATH/sft.jsonl`, where each example is `{"prompt": str, "response": str}`.

> **Note:** SFT is often used as a warm-start for RL fine-tuning in practice: SFT requires high-quality annotated data (reasoning traces), while RL requires only the correct answer as feedback. Even when annotated data is plentiful, RL can find better policies beyond SFT. For this project, SFT and RL phases are treated separately since the model size is too small to show composable gains.

---

### 4.1 Using HuggingFace Models

**Loading a model and tokenizer** (in bfloat16 with FlashAttention-2):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
```

**Forward pass:**

```python
input_ids = train_batch["input_ids"].to(device)
labels    = train_batch["labels"].to(device)
logits    = model(input_ids).logits
loss      = F.cross_entropy(..., ...)
```

**Saving a trained model** — use `.save_pretrained()` for both model and tokenizer. Save under `/data/<username>/` on the cluster due to file sizes; saving the tokenizer alongside the model keeps the directory self-contained.

```python
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)
```

**Gradient accumulation** — 80 GB VRAM is insufficient for large batch sizes, even in bfloat16. Gradient accumulation defers the optimizer step until after `k` microbatches, giving an effective batch size multiplied by `k`:

```python
gradient_accumulation_steps = 4
for idx, (inputs, labels) in enumerate(data_loader):
    logits = model(inputs)
    loss   = loss_fn(logits, labels) / gradient_accumulation_steps
    loss.backward()
    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 4.2 SFT Helper Methods

#### `tokenize_prompt_and_output` (2 points)

Tokenize the question and output strings separately, concatenate, and build a `response_mask` — a boolean mask that is `True` for response tokens and `False` for prompt/padding tokens. The mask is used in the training loop so that the loss is computed only over response tokens.

```python
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Args:
        prompt_strs: list[str]
        output_strs: list[str]
        tokenizer:   PreTrainedTokenizer

    Returns:
        dict with keys:
          input_ids     (batch_size, max_len - 1)  — tokenized prompt+output, final token removed
          labels        (batch_size, max_len - 1)  — input_ids shifted left (first token removed)
          response_mask (batch_size, max_len - 1)  — 1 for response tokens in labels, 0 otherwise
    """
```

Test: `uv run pytest -k test_tokenize_prompt_and_output`

---

#### `compute_entropy` (1 point)

Compute the per-token entropy of next-token predictions. For a discrete distribution p(x) over vocabulary V:

```
H(p) = -∑_{x ∈ V} p(x) log p(x)
```

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: (batch_size, sequence_length, vocab_size)
    Returns:
        (batch_size, sequence_length) — per-token entropy
    """
```

Use a numerically stable method (e.g., logsumexp) to avoid overflow.

Test: `uv run pytest -k test_compute_entropy`

---

#### `get_response_log_probs` (2 points)

Compute per-token conditional log-probabilities from a causal LM, and optionally return per-token entropy. For a prefix x, logits f_θ(x) ∈ ℝ^|V|, and label y ∈ V:

```
log p_θ(y | x) = log [softmax(f_θ(x))]_y
```

```python
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,       # (batch_size, sequence_length)
    labels: torch.Tensor,          # (batch_size, sequence_length)
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Returns:
        "log_probs":     (batch_size, sequence_length) — per-token log p_θ(x_t | x_{<t})
        "token_entropy": (batch_size, sequence_length) — present only if return_token_entropy=True
    """
```

Obtain logits with `model(input_ids).logits`. Use numerically stable methods from `torch.nn.functional`.

Test: `uv run pytest -k test_get_response_log_probs`

---

#### `masked_normalize` (1 point)

Sum over tensor elements and divide by a normalization constant, respecting a boolean mask (positions where `mask == 0` do not contribute).

```python
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Args:
        tensor:             value tensor
        mask:               same shape as tensor; 1 = included, 0 = excluded
        normalize_constant: divide the sum by this value
        dim:                dimension to reduce; None reduces all dimensions
    Returns:
        normalized sum (masked elements do not contribute)
    """
```

Test: `uv run pytest -k test_masked_normalize`

---

#### `sft_microbatch_train_step` (3 points)

A single microbatch update for SFT: compute the NLL loss over response tokens, scale for gradient accumulation, and call `loss.backward()`.

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,     # (batch_size, sequence_length)
    response_mask: torch.Tensor,        # (batch_size, sequence_length)
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Returns:
        loss      — scalar tensor, adjusted for gradient accumulation (for logging)
        metadata  — dict of stats to log
    """
```

`loss.backward()` must be called inside this function. Divide by `gradient_accumulation_steps` before the backward pass.

Test: `uv run pytest -k test_sft_microbatch_train_step`

---

#### `log_generations` (1 point)

Log in-the-loop generations from the SFT/RL model during training. For each example, log at minimum:

1. Input prompt
2. Model response
3. Ground-truth answer
4. Reward info: format reward, answer reward, total reward
5. Average token entropy of the response
6. Average response length; average length for correct vs. incorrect responses

---

### 4.3 SFT Experiment (2 points, ~2 H100 hrs)

Using the helper methods above, implement the full SFT procedure (Algorithm 1) to fine-tune Qwen 2.5 Math 1.5B Base on the MATH reasoning SFT dataset.

**Training setup:**
- Run with **2 GPUs**: one for the policy model (training), one for the vLLM evaluation instance
- Periodically evaluate on the MATH validation set during training
- Use gradient clipping with clip value 1.0

**vLLM initialization and weight loading** (required for 2-GPU setup):

```python
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch  = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch   = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model  = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
```

**wandb metric separation** (train vs. eval axes):

```python
wandb.define_metric("train_step")
wandb.define_metric("eval_step")
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*",  step_metric="eval_step")
```

**(a)** Run SFT varying the number of unique training examples across `{128, 256, 512, 1024, full dataset}`. Tune learning rate and batch size to achieve at least **15% validation accuracy** on the full dataset.

**Output:** Validation accuracy curves for each dataset size.

**(b)** Filter the SFT dataset to only examples where the response contains the correct answer. Run SFT on the full filtered dataset.

**Output:** Filtered dataset size and validation accuracy curve. Compare to unfiltered SFT results.