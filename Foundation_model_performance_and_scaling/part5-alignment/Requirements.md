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

---

## Section 5: Expert Iteration

### Algorithm: Expert Iteration

```
Input: initial policy model π_θ_init; training dataset D_train; number of EI steps n_ei_steps;
       number of rollouts per question G; SFT epochs per EI step n_sft_epochs;
       SFT batch size D_b; sampling temperature T
1: policy model π_θ ← π_θ_init
2: for step = 1, …, n_ei_steps do
3:     rollout dataset D_rollout ← {}
4:     for each question q in D_train do
5:         generate G responses r_1, …, r_G using π_θ with temperature T
6:         for each response r_i do
7:             reward ← r1_zero_reward_fn(r_i, ground_truth(q))
8:             if reward > 0 then
9:                 add (q, r_i) to D_rollout
10:            end if
11:        end for
12:    end for
13:    fine-tune π_θ on D_rollout for n_sft_epochs epochs using batch size D_b
14: end for
Output: π_θ
```

**Goal:** Starting from the base model (Qwen 2.5 Math 1.5B), improve mathematical reasoning through an iterative rollout-filter-finetune loop. Each Expert Iteration (EI) step generates G responses per training question, keeps only those with correct answers (verified via the rule-based reward function), and fine-tunes the model on this filtered dataset. Repeating this loop bootstraps reasoning capabilities without human-written traces.

**Training data:** `data/math/train.jsonl` — 7499 `{problem, solution}` examples.

**Base model:** Qwen 2.5 Math 1.5B (same as SFT; do not warm-start from the SFT checkpoint).

---

### 5.1 vLLM Generation for Expert Iteration

Use vLLM with the following sampling configuration for rollouts (set `min_tokens=4` to prevent degenerate empty responses):

```python
sampling_params = SamplingParams(
    temperature=sampling_temperature,
    max_tokens=max_tokens,
    min_tokens=4,
    n=G,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
```

Setting `n=G` generates G independent responses per prompt in a single vLLM call, which is more efficient than G separate calls.

---

### 5.2 Expert Iteration Experiment (5 points, ~4 H100 hrs)

Using the SFT helper methods from Section 4 and the Expert Iteration algorithm above, run the following experiment:

**Setup:** Run Expert Iteration for `n_ei_steps = 5` steps. After each EI step (rollout + fine-tune), evaluate the current policy on the MATH validation set and record validation accuracy.

**Experiment:** Vary at least one of the following hyperparameters and report the effect on validation accuracy:

- Number of rollouts per question: **G ∈ {1, 4, 16}** (more rollouts → richer filtered dataset per step)
- SFT epochs per EI step: varying `n_sft_epochs`
- SFT batch size: **D_b ∈ {512, 1024, 2048}**

Use gradient clipping with clip value 1.0.

**Logging:** Log validation accuracy after each EI step. Optionally log token entropy and average response length as diagnostic signals.

**(a)** Run the Expert Iteration experiment. Report validation accuracy curves across EI steps for each hyperparameter configuration tested.

**Target:** ≥ 15% validation accuracy after 5 EI steps.

**Output:** Validation accuracy curves (one curve per configuration).

**(b)** In 2 sentences, compare Expert Iteration to SFT from Section 4. Does the data source (model-generated vs. human/R1-generated) matter? Does the iterative loop help?

**Output:** 2-sentence comparison.

**(c)** Plot token entropy over the course of Expert Iteration training. Does entropy increase, decrease, or stay flat? Compare to the SFT entropy trajectory.

**Output:** Entropy plot and 1–2 sentence interpretation.

---

## Section 6 — Primer on Policy Gradients

An exciting finding in language model research is that performing RL against verified rewards with strong base models can lead to significant improvements in reasoning capabilities [OpenAI et al., 2024; DeepSeek-AI et al., 2025]. The strongest open reasoning models (DeepSeek R1, Kimi k1.5) were trained using policy gradients — a powerful RL algorithm that can optimize arbitrary reward functions.

This primer closely follows OpenAI's Spinning Up in Deep RL [Achiam, 2018] and Nathan Lambert's RLHF Book [Lambert, 2024].

---

### 6.1 Language Models as Policies

A causal language model with parameters θ defines a probability distribution over the next token `a_t ∈ V` given the current text prefix `s_t` (the state/observation). In RL terms:

- **State `s_t`**: everything generated so far (prompt + tokens emitted up to step t)
- **Action `a_t`**: the next token to emit
- **Policy `π_θ`**: the LM itself — `a_t ~ π_θ(· | s_t)`, where `π_θ(a_t | s_t) = softmax(f_θ(s_t))[a_t]`

Two primitive operations are needed:
1. **Sampling**: draw `a_t ~ π_θ(· | s_t)` (normal autoregressive generation)
2. **Scoring**: evaluate `log π_θ(a_t | s_t)` (a forward pass with the logits)

The episode ends when the model emits `</answer>` or hits the max token budget.

---

### 6.2 Trajectories

A trajectory (also called episode or rollout) is the full sequence of states and actions:

```
τ = (s_0, a_0, s_1, a_1, ..., s_T, a_T)
```

Key mechanics:
- **Initial state `s_0`**: drawn from a distribution over formatted prompts ρ_0
- **State transitions**: deterministic — `s_{t+1} = s_t ∥ a_t` (just concatenation)
- **T**: length of the trajectory; `a_T` is the end-of-text token or the last token before the budget runs out

Because transitions are deterministic in LLM settings, the only source of randomness is the sampling of each action from the model's distribution.

---

### 6.3 Rewards and Return

A scalar reward `r_t = R(s_t, a_t)` judges the quality of each action. For verified math:

- All intermediate tokens get `r_t = 0`
- The final token gets: `r_T = 1` if correct, `0` otherwise

The **return** `R(τ)` aggregates rewards over the full trajectory. Two common forms:

| Form | Formula | When to use |
|------|---------|-------------|
| Undiscounted (finite horizon) | `R(τ) = Σ_{t=0}^{T} r_t` | Natural episode end (our case) |
| Discounted (infinite horizon) | `R(τ) = Σ_{t=0}^{∞} γ^t r_t` | No natural end, discounts far-future rewards |

For MATH, we use the undiscounted form. Because only the terminal token has nonzero reward, `R(τ)` simply equals `r_T` (1 or 0).

**Objective:** maximize expected return over trajectories drawn from the current policy:

```
J(θ) = E_{τ ~ π_θ} [R(τ)]
θ* = argmax_θ J(θ)
```

---

### 6.4 Vanilla Policy Gradient (REINFORCE)

We want to do gradient ascent on `J(θ)`:

```
θ_{k+1} = θ_k + α ∇_θ J(θ_k)
```

The challenge: `J(θ)` involves an expectation over trajectories, and the trajectories are sampled (not differentiable). The **log-derivative trick** resolves this.

**Key identity (REINFORCE):**

```
∇_θ J(π_θ) = E_{τ ~ π_θ} [ Σ_{t=0}^{T} ∇_θ log π_θ(a_t | s_t) · R(τ) ]
```

**Derivation in plain steps:**

1. Write `J(θ)` as a sum over all possible trajectories weighted by their probability:
   `J(θ) = Σ_τ P(τ|θ) R(τ)`

2. Push the gradient inside the sum:
   `∇_θ J(θ) = Σ_τ ∇_θ P(τ|θ) R(τ)`

3. Apply the log-derivative trick `∇P = P ∇ log P` to convert `∇P(τ|θ)` into `P(τ|θ) ∇ log P(τ|θ)`, which puts things back into expectation form:
   `= E_{τ ~ π_θ} [ ∇_θ log P(τ|θ) R(τ) ]`

4. Expand `log P(τ|θ) = log ρ_0(s_0) + Σ_t log P(s_{t+1}|s_t, a_t) + Σ_t log π_θ(a_t|s_t)`.
   The environment terms (ρ_0 and transition probabilities) don't depend on θ, so their gradients are zero. Only the policy log-probs survive:
   `∇_θ log P(τ|θ) = Σ_t ∇_θ log π_θ(a_t | s_t)`

**Intuition:** the gradient increases the log-probability of every token in trajectories that got reward 1, and decreases it for trajectories that got reward 0. Correct paths become more likely; wrong paths become less likely.

**Sample estimate (used in practice):** collect N rollouts, compute:

```
ĝ = (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T} ∇_θ log π_θ(a_t^(i) | s_t^(i)) · R(τ^(i))
```

Then update `θ ← θ + α ĝ`.

**PyTorch implementation note:** we don't implement this gradient directly. Instead we define a scalar `pg_loss` such that `pg_loss.backward()` produces `ĝ`:

```
pg_loss = (1/N) Σ_{i,t} log π_θ(a_t^(i) | s_t^(i)) · (R(τ^(i)) - b(s_t^(i)))
```

**Important:** `pg_loss` is not a meaningful evaluation metric. Only report train/validation rewards.

---

### 6.5 Policy Gradient Baselines

**Problem:** vanilla REINFORCE has high variance. If many rollouts all get reward 1 (or all get 0), the gradient estimate has high noise and converges slowly.

**Solution:** subtract a **baseline** `b(s_t)` from the return:

```
∇_θ J(π_θ) = E_{τ ~ π_θ} [ Σ_t ∇_θ log π_θ(a_t|s_t) · (R(τ) - b(s_t)) ]
```

**Why this is still unbiased:** the baseline term subtracts `E[ Σ_t ∇_θ log π_θ(a_t|s_t) · b(s_t) ]`. For any fixed `s_t`, the inner expectation over `a_t` is `E_{a_t ~ π_θ}[∇_θ log π_θ(a_t|s_t)]`, which is always zero (the score function identity). So the baseline adds zero in expectation — it reduces variance without adding bias.

**Common choices:**
- `b(s_t) = V^π(s_t)` — the on-policy value function (expected return from state `s_t`)
- `b = mean(R(τ))` over the current batch — simple mean baseline used in GRPO
- `b(s_t) = mean reward for the same question` — the group baseline used in GRPO

The quantity `R(τ) - b(s_t)` is the **advantage**: how much better this trajectory was compared to the baseline expectation.

---

### 6.6 Off-Policy Policy Gradient

**Problem with on-policy learning:** after each gradient update, all old rollouts become stale — the current policy has changed. We must discard them and regenerate, which is expensive (inference >> training for large models).

**Off-policy solution:** reuse rollouts collected from an older policy `π_{θ_old}` to update the current policy `π_θ`. Correct for the mismatch using **importance weights**:

```
ĝ_off-policy = (1/N) Σ_{i,t} [π_θ(a_t^(i)|s_t^(i)) / π_{θ_old}(a_t^(i)|s_t^(i))] · ∇_θ log π_θ(a_t^(i)|s_t^(i)) · R(τ^(i))
```

The ratio `π_θ / π_{θ_old}` is the importance weight — it corrects for the fact that we collected data under `π_{θ_old}` but we're updating `π_θ`. This estimator is unbiased as long as the two policies aren't too different.

**Practical benefit:** take multiple gradient steps per batch of rollouts, amortizing the expensive vLLM inference cost across many updates.

**Key constraint:** if `π_θ` drifts too far from `π_{θ_old}`, the importance weights become large and the estimate becomes unreliable. PPO and GRPO address this with a clipping mechanism that prevents the ratio from exceeding `[1-ε, 1+ε]`.

**Connection to GRPO (Section 7):** GRPO uses:
- Off-policy rollouts (multiple updates per batch)
- Group baseline: for each question, the baseline is the mean reward across its G rollouts
- Clipped importance weights (same clipping as PPO) to bound policy divergence

---

## Section 7 — Group Relative Policy Optimization (GRPO)

GRPO is the policy gradient variant used to train DeepSeek R1 and related reasoning models. It avoids the need for a separate value network by using the model's own group of rollouts as a self-contained baseline.

---

### 7.1 GRPO Algorithm

#### Advantage Estimation

For a question `q`, sample G outputs `{o^(i)}_{i=1}^G ~ π_θ(·|q)` and compute reward `r^(i) = R(q, o^(i))` for each. The **group-normalized advantage** for output `i` is:

**Standard (DeepSeekMath / DeepSeek R1):**
```
A^(i) = (r^(i) - mean(r^(1), ..., r^(G))) / (std(r^(1), ..., r^(G)) + advantage_eps)
```

**Dr. GRPO simplified variant** (removes std normalization, which can over-reward low-variance groups):
```
A^(i) = r^(i) - mean(r^(1), ..., r^(G))
```

`advantage_eps` is a small constant (e.g. 1e-6) to prevent division by zero. `A^(i)` is the same for all tokens in response `o^(i)` — one scalar per rollout, not per token.

#### High-Level Training Loop (Algorithm 3)

```
Input: initial policy π_{θ_init}, reward function R, questions D
1: π_θ ← π_{θ_init}
2: for step = 1, ..., n_grpo_steps:
3:     sample batch D_b from D
4:     set old policy π_{θ_old} ← π_θ
5:     sample G outputs {o^(i)} ~ π_{θ_old}(·|q) for each q ∈ D_b
6:     compute rewards {r^(i)} via R(q, o^(i))
7:     compute advantages A^(i) via group normalization
8:     for train_step = 1, ..., n_train_steps_per_rollout_batch:
9:         update π_θ by maximizing GRPO-Clip objective (Eq. 29)
10:    end for
11: end for
Output: π_θ
```

#### GRPO-Clip Objective

The full objective combines off-policy importance weighting, group-normalized advantages, and PPO-style clipping:

```
J_GRPO-Clip(θ) = E_{q, {o^(i)}} [
  (1/G) Σ_i (1/|o^(i)|) Σ_t
    min(
      ratio_t^(i) · A^(i),
      clip(ratio_t^(i), 1-ε, 1+ε) · A^(i)
    )
]

where ratio_t^(i) = π_θ(o_t^(i) | q, o_{<t}^(i)) / π_{θ_old}(o_t^(i) | q, o_{<t}^(i))
```

**Clipping intuition — rewritten as `g(ε, A^(i))`:**

| Advantage sign | Per-token objective | Effect |
|---|---|---|
| `A^(i) > 0` | `min(ratio, 1+ε) · A^(i)` | Increase token prob, but stop incentivizing once ratio > 1+ε |
| `A^(i) < 0` | `max(ratio, 1-ε) · A^(i)` | Decrease token prob, but stop once ratio < 1-ε |

The clip prevents `π_θ` from straying too far from `π_{θ_old}`. Without it, multiple gradient steps on the same rollout batch would cause the policy to over-optimize on stale data.

---

### 7.2 Implementation

All implementations go in `cs336_alignment/section6_grpo/` (or equivalent section folder). Each function below has a corresponding adapter in `tests/adapters.py` and a pytest test.

---

#### 7.2.1 `compute_group_normalized_rewards` (2 points)

Compute per-rollout rewards and normalize within groups.

```python
def compute_group_normalized_rewards(
    reward_fn,               # Callable[[str, str], dict[str, float]]
    rollout_responses,       # list[str], length = n_prompts * group_size
    repeated_ground_truths,  # list[str], length = n_prompts * group_size
    group_size,              # int, G
    advantage_eps,           # float, small constant for std normalization
    normalize_by_std,        # bool, True = standard GRPO, False = Dr. GRPO
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    # Returns:
    #   advantages   shape (rollout_batch_size,)  — group-normalized rewards
    #   raw_rewards  shape (rollout_batch_size,)  — unnormalized rewards
    #   metadata     dict of stats to log (mean, std, min/max of rewards, etc.)
```

**Key detail:** `rollout_responses` and `repeated_ground_truths` are both length `n_prompts * G`, laid out as `[q1_r1, q1_r2, ..., q1_rG, q2_r1, ...]`. Group i spans indices `[i*G : (i+1)*G]`.

Test: `uv run pytest -k test_compute_group_normalized_rewards`

---

#### 7.2.2 `compute_naive_policy_gradient_loss` (1 point)

Simple policy gradient loss: multiply per-token log-probs by the advantage (negated, so minimizing = gradient ascent).

```python
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # shape (batch_size, 1)
    policy_log_probs: torch.Tensor,           # shape (batch_size, sequence_length)
) -> torch.Tensor:                            # shape (batch_size, sequence_length)
    # per-token loss = -A^(i) * log π_θ(o_t | q, o_{<t})
    # broadcast advantages over sequence_length dimension
```

Used for both `no_baseline` (A = raw reward) and `reinforce_with_baseline` (A = group-normalized reward).

Test: `uv run pytest -k test_compute_naive_policy_gradient_loss`

---

#### 7.2.3 `compute_grpo_clip_loss` (2 points)

GRPO-Clip loss with importance weighting and clipping.

```python
def compute_grpo_clip_loss(
    advantages: torch.Tensor,      # shape (batch_size, 1)
    policy_log_probs: torch.Tensor,  # shape (batch_size, sequence_length) — current policy
    old_log_probs: torch.Tensor,     # shape (batch_size, sequence_length) — old policy
    cliprange: float,                # ε, e.g. 0.2
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # loss shape (batch_size, sequence_length)
    # metadata: clip fraction per token (was the clipped term the min?)
```

**Implementation:**
1. Compute `ratio = exp(policy_log_probs - old_log_probs)` (importance weight, in log space for stability)
2. Compute `clipped_ratio = clip(ratio, 1-ε, 1+ε)`
3. Per-token loss = `-min(ratio * A, clipped_ratio * A)`
4. Broadcast advantages over sequence length

Test: `uv run pytest -k test_compute_grpo_clip_loss`

---

#### 7.2.4 `compute_policy_gradient_loss` — wrapper (1 point)

Dispatcher that routes to the correct loss function based on `loss_type`.

```python
def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,                             # (batch_size, seq_len)
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,                    # (batch_size, 1)
    advantages: torch.Tensor | None = None,                     # (batch_size, 1)
    old_log_probs: torch.Tensor | None = None,                  # (batch_size, seq_len)
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Dispatches:
    #   no_baseline              → compute_naive_policy_gradient_loss(raw_rewards, ...)
    #   reinforce_with_baseline  → compute_naive_policy_gradient_loss(advantages, ...)
    #   grpo_clip                → compute_grpo_clip_loss(advantages, ..., old_log_probs, ε)
```

Test: `uv run pytest -k test_compute_policy_gradient_loss`

---

#### 7.2.5 `masked_mean` (1 point)

Average over response tokens only (ignore prompt and padding positions).

```python
def masked_mean(
    tensor: torch.Tensor,    # data to average
    mask: torch.Tensor,      # same shape; 1 = response token, 0 = prompt/padding
    dim: int | None = None,  # dimension to reduce; None = mean over all masked elements
) -> torch.Tensor:
```

Used to reduce per-token losses of shape `(batch_size, seq_len)` to per-example scalars, and also for per-token entropy and clip fraction statistics.

Test: `uv run pytest -k test_masked_mean`

---

#### 7.2.6 `grpo_microbatch_train_step` (3 points)

Single microbatch forward + backward pass for GRPO.

```python
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,       # (batch_size, seq_len)
    response_mask: torch.Tensor,          # (batch_size, seq_len)
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,     # (batch_size, 1)
    advantages: torch.Tensor | None = None,      # (batch_size, 1)
    old_log_probs: torch.Tensor | None = None,   # (batch_size, seq_len)
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 1. Call compute_policy_gradient_loss → per-token loss (batch_size, seq_len)
    # 2. Apply masked_mean over sequence dim → scalar loss per example
    # 3. Average over batch dim
    # 4. Divide by gradient_accumulation_steps
    # 5. Call loss.backward()
    # Returns: (scalar loss, metadata dict)
```

Test: `uv run pytest -k test_grpo_microbatch_train_step`

---

#### 7.2.7 `grpo_train_loop` (5 points)

Complete GRPO training loop. Implements Algorithm 3 using all the primitives above.

**Default hyperparameters:**

```python
n_grpo_steps                = 200
learning_rate               = 1e-5
advantage_eps               = 1e-6
rollout_batch_size          = 256
group_size                  = 8          # G
sampling_temperature        = 1.0
sampling_min_tokens         = 4
sampling_max_tokens         = 1024
epochs_per_rollout_batch    = 1          # on-policy default
train_batch_size            = 256        # = rollout_batch_size for on-policy
gradient_accumulation_steps = 128        # microbatch size = 2
gpu_memory_utilization      = 0.85
loss_type                   = "reinforce_with_baseline"
use_std_normalization       = True

optimizer = torch.optim.AdamW(
    policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
)
```

**Key relationships:**
```python
assert train_batch_size % gradient_accumulation_steps == 0
micro_train_batch_size = train_batch_size // gradient_accumulation_steps  # = 2

assert rollout_batch_size % group_size == 0
n_prompts_per_rollout_batch = rollout_batch_size // group_size  # = 32 questions × 8 rollouts

assert train_batch_size >= group_size
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size  # = 128
```

**Implementation notes:**
- Use the `r1_zero` prompt template; stop generation at `</answer>`
- Gradient clipping with clip value 1.0
- For off-policy (multiple epochs per rollout batch): compute old log-probs once before the epoch loop and detach; do not recompute per epoch
- `grpo_clip` requires off-policy mode; `reinforce_with_baseline` and `no_baseline` work on-policy
- Evaluate on ≥ 1024 validation examples every 5–10 steps (CoT eval is noisy; small subsets mislead)
- Use `typer` for argument parsing

**Metrics to log per optimizer step:**
- Loss
- Gradient norm
- Token entropy
- Train rewards (total, format, answer)
- Clip fraction (off-policy only)

**Deliverable:** training runs showing validation reward improving over steps, plus a few example rollouts at different points in training to illustrate qualitative improvement.

---

### 7.3 GRPO Experiment (Section 7.3)

Run GRPO on Qwen 2.5 Math 1.5B (base model, not SFT checkpoint) and report:

**(a)** Validation reward curve over training steps.

**(b)** A few sampled rollouts at early, mid, and late training — do the generations look more structured/correct over time?

**(c) Ablation:** Compare the three loss types:
- `no_baseline` — raw reward, no normalization
- `reinforce_with_baseline` — group-normalized reward (with and without std normalization)
- `grpo_clip` — full GRPO-Clip (off-policy, multiple gradient steps per rollout batch)

Report validation reward curves for each variant.

**Target:** ≥ 15% validation accuracy after GRPO training.

---

## Section 8 — GRPO Experiments

Each experiment below uses 2 GPUs (one for vLLM rollout generation, one for policy training). GPU hour estimates are rough guides. Runs may be stopped early if a configuration clearly diverges or is suboptimal before 200 GRPO steps.

---

### 8.1 Learning Rate Sweep (`grpo_learning_rate`) — 2 points, ~6 H100 hrs

Starting from the default hyperparameters (Section 7.2.7), sweep over learning rates and measure final validation answer rewards.

**Deliverables:**
- Validation reward curves for each learning rate tested.
- A model checkpoint that achieves at least **25% validation accuracy** on MATH.
- A 2-sentence discussion on any other trends observed in logged metrics.

> The best learning rate from this sweep should be used for all subsequent experiments.

---

### 8.2 Effect of Baselining (`grpo_baselines`) — 2 points, ~2 H100 hrs

Using the tuned learning rate and on-policy default (`epochs_per_rollout_batch=1`), compare the following loss types:

- `no_baseline` — raw reward, no centering
- `reinforce_with_baseline` — group-mean-centered reward

Both runs use `use_std_normalization=True` (the default).

**Deliverables:**
- Validation reward curves for each loss type.
- A 2-sentence discussion on any other trends observed in logged metrics.

> Use the best-performing loss type for subsequent experiments.

---

### 8.3 Length Normalization

#### 8.3.1 Conceptual Analysis (`think_about_length_normalization`) — 1 point

Two approaches to aggregating per-token losses differ in how they assign credit:

- **`masked_mean`**: average the loss over the unmasked (response) tokens in each sequence.
- **`masked_normalize` with `constant_normalizer`**: sum over unmasked tokens and divide by a fixed scalar (e.g., max sequence length), rather than the per-example token count.

The difference is illustrated with a batch of two responses — 4 tokens and 7 tokens — where the per-token loss is `ratio * advantage`:

```python
from cs336_alignment.section4_sft.helpers import masked_mean, masked_normalize

ratio = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
], requires_grad=True)
advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2],
])
masks = torch.tensor([
    [1, 1, 1, 1, 0, 0, 0],   # 4-token response
    [1, 1, 1, 1, 1, 1, 1],   # 7-token response
])

max_gen_len = 7
masked_mean_result      = masked_mean(ratio * advs, masks, dim=1)
masked_normalize_result = masked_normalize(ratio * advs, masks, dim=1, constant_normalizer=max_gen_len)

# masked_mean       → tensor([2., 2.])
# masked_normalize  → tensor([1.1429, 2.0000])

masked_mean_result.mean().backward()
# ratio.grad:
# [[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
#  [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]]

ratio.grad.zero_()
masked_normalize_result.mean().backward()
# ratio.grad:
# [[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
#  [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]]
```

With `masked_mean`, each token in the shorter response receives a larger gradient (0.25) than each token in the longer one (0.143). With `masked_normalize` using a fixed `constant_normalizer`, every token receives equal weight regardless of response length.

> **Note on parameter naming:** the spec PDF uses `constant_normalizer` in its code examples, but the existing implementation in `cs336_alignment/section4_sft/helpers.py` and the test adapter both use `normalize_constant`. These are the same parameter. The code examples above reflect the PDF; when running them against the actual implementation, use `normalize_constant=max_gen_len` instead.

**Deliverable:** A written comparison (no training runs needed) of the two approaches. Discuss pros and cons of each and describe any specific settings or examples where one seems preferable.

---

#### 8.3.2 Empirical Comparison (`grpo_length_normalization`) — 2 points, ~2 H100 hrs

Run end-to-end GRPO training twice — once with `masked_mean` and once with `masked_normalize` — and compare results.

**Deliverables:**
- Validation answer reward curves for both approaches.
- Commentary on findings, including any metrics with a noticeable trend (especially gradient norm as a stability indicator).

> Fix to the better-performing length normalization for subsequent experiments.

---

### 8.4 Effect of Group Standard Deviation Normalization (`grpo_group_standard_deviation`) — 2 points, ~2 H100 hrs

Compare `use_std_normalization=True` (standard GRPO, divides advantage by group std) versus `use_std_normalization=False` (Dr. GRPO, advantage = reward − group mean only).

Background: dividing by per-group std can introduce an unintended bias — questions where all rollouts score similarly (all correct or all wrong) receive artificially amplified gradients. The Dr. GRPO variant avoids this by omitting the std normalization.

**Deliverables:**
- Validation answer reward curves for both settings.
- Commentary on findings, including any metrics with a noticeable trend (especially gradient norm).

> Fix to the better-performing group normalization for subsequent experiments.

---

### 8.5 Off-Policy GRPO

#### 8.5.1 Implementation (`grpo_off_policy`)

Off-policy GRPO takes multiple gradient steps per rollout batch, amortizing the cost of expensive vLLM generation across many updates.

**Implementation requirements** (may already be present from Section 7.2.7):

- **Multiple epochs per rollout batch** — controlled by `rollout_batch_size`, `epochs_per_rollout_batch`, and `train_batch_size`.
- **`old_log_probs` precomputation** — after each rollout generation phase and before the inner gradient loop, compute per-token log-probs of the rollout responses under the current policy using `torch.inference_mode()`. These are the reference `π_{θ_old}` for importance weighting.
- **Loss type** — use `grpo_clip` for all off-policy runs.

> Adjust `gradient_accumulation_steps` proportionally when changing `train_batch_size` to keep per-microbatch memory usage constant.

---

#### 8.5.2 Off-Policy Hyperparameter Sweep (`grpo_off_policy_sweep`) — 4 points, ~12 H100 hrs

Fix `rollout_batch_size=256`. Sweep over a range of `epochs_per_rollout_batch` and `train_batch_size` values.

**Approach:**
1. **Broad sweep** (< 50 GRPO steps): explore the performance landscape across a wider range.
2. **Focused sweep** (200 GRPO steps): run the most promising configurations to convergence.

**Deliverables:**
- A brief experiment log explaining the chosen sweep ranges and rationale.
- Validation answer reward curves reported against both number of validation steps and wall-clock time.
- Comparison against the on-policy baseline (`epochs_per_rollout_batch=1`, `train_batch_size=256`).
- Commentary on any trends in entropy, response length, and other logged metrics. Compare the entropy trajectory to what was observed in the Expert Iteration experiments.

---

#### 8.5.3 Clip Ablation in Off-Policy Setting (`grpo_off_policy_clip_ablation`) — 2 points, ~2 H100 hrs

Implement a new loss type **`grpo_no_clip`** — off-policy importance-weighted policy gradient without PPO-style clipping:

```
per-token loss = -(π_θ(o_t | q, o_{<t}) / π_{θ_old}(o_t | q, o_{<t})) * A^(i)
```

This is equivalent to `grpo_clip` with `cliprange=∞`. It tests whether the clipping mechanism is necessary for stability in the off-policy setting.

**Deliverables:**
- Implementation of `grpo_no_clip` loss type in `compute_policy_gradient_loss` and the adapter.
- Validation answer reward curves comparing `grpo_clip` and `grpo_no_clip` using the best-performing off-policy hyperparameters from §8.5.2.
- Commentary on findings compared to the clipped run, including entropy, response length, and gradient norm.

---

### 8.6 Prompt Ablation (`grpo_prompt_ablation`) — 2 points, ~2 H100 hrs

The choice of prompt during RL training has a significant effect on model behavior, depending on how the base model was pretrained.

Compare two prompts using the best hyperparameters found above:

| Prompt | File | Reward function |
|--------|------|-----------------|
| **R1-Zero** (default) | `cs336_alignment/prompts/r1_zero.prompt` | `r1_zero_reward_fn` |
| **Question-only** | `cs336_alignment/prompts/question_only.prompt` | `question_only_reward_fn` (in `cs336_alignment/drgrpo_grader.py`) |

The question-only prompt is simply:
```
{question}
```

Use the same prompt for both training and validation when running the question-only experiment.

**Deliverables:**
- Validation answer reward curves for both prompts.
- Commentary comparing the two runs, including entropy, response length, and gradient norm trends.
- An explanation of the observed differences, considering how Qwen 2.5 Math 1.5B was pretrained.