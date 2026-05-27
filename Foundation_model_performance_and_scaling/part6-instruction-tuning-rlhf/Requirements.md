# Part 6 — Instruction Tuning & RLHF

Supplement to CS336 Assignment 5 (Alignment) — Version 1.0.1, Spring 2025

---

## Section 1: Overview

### What This Project Implements

1. **Zero-shot prompting baselines** for a variety of evaluation datasets.
2. **Supervised fine-tuning**, given demonstration data with instruction-response pairs.
3. **Direct preference optimization (DPO)** for learning from pairwise preference data.

### What This Project Runs

1. Measure **Llama 3.1** zero-shot prompting performance (baseline).
2. Instruction fine-tune Llama 3.1.
3. Fine-tune Llama 3.1 on pairwise preference data.

### Code Structure

| Path | Description |
|------|-------------|
| `cs336_alignment/` | Main implementation directory |
| `cs336_alignment/prompts/` | Prompt text files — zero-shot system prompt and Alpaca instruction-tuning prompt |
| `tests/test_data.py` | Data loading tests |
| `tests/test_dpo.py` | DPO implementation tests |
| `tests/test_metrics.py` | Metrics tests |
| `tests/test_sft.py` | SFT implementation tests |
| `tests/adapters.py` | Adapter hooks connecting implementation to tests |
| `data/` | Benchmark datasets: MMLU, GSM8K, AlpacaEval, SimpleSafetyTests |
| `scripts/alpaca_eval_vllm_llama3_3_70b_fn/` | AlpacaEval config using Llama 3.3 70B Instruct as judge |
| `README.md` | Environment setup instructions |

### Constraints

- Build most components **from scratch**
- **Allowed:** vLLM for text generation; HuggingFace Transformers for loading Llama 3.x models and tokenizers
- **Not allowed:** HuggingFace training utilities (e.g., the `Trainer` class)

---

## Section 2: Motivation — Training Generalist LLMs

In contrast to the main assignment (which focused on reasoning models for math), this project builds **generalist dialogue systems** that handle a wide range of NLP tasks: setting up evaluations, collecting fine-tuning and RLHF data, and training a model to follow user instructions and refuse harmful ones.

**Evaluation benchmarks:**
- **MMLU** (Hendrycks et al., 2021) — factual knowledge
- **GSM8K** (Cobbe et al., 2021) — reasoning
- **AlpacaEval** (Li et al., 2023) — chatbot quality
- **SimpleSafetyTests** (Vidgen et al., 2024) — safety

### Models

| Model | Cluster Path |
|-------|-------------|
| Llama 3.1 8B Base | `/data/a5-alignment/models/Llama-3.1-8B` |
| Llama 3.3 70B Instruct | `/data/a5-alignment/models/Llama-3.3-70B-Instruct` |

Point both `vllm.LLM` and `transformers.AutoModelForCausalLM.from_pretrained` to these paths to avoid re-downloading.

### Zero-Shot System Prompt

All tasks use the same system prompt (file: `cs336_alignment/prompts/zero_shot_system_prompt.prompt`):

```
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also
have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic,
dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some
controversial topics.

# Query:
```{instruction}```
# Answer:
```

The model generates the answer, closes the code block with ` ``` `, then starts the next turn with `# Query:` — use `# Query:` as the stop string for generation.

---

### §2.1 — Zero-Shot MMLU Baseline

**Prompt format:**

```
Answer the following multiple choice question about {subject}. Respond with a single
sentence of the form "The correct answer is _", filling the blank with the letter
corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Answer:
```

**Evaluation metric:** Parse the generated output to extract the predicted letter (A/B/C/D); compare against the gold answer letter.

**Generation hyperparameters:** Greedy decoding — temperature 0.0, top-p 1.0.

#### Implementation: `mmlu_baseline` (4 points)

**(a)** Write a function to parse generated outputs into the predicted answer letter. Return `None` if unparseable. Implement adapter `[run_parse_mmlu_response]` and pass `uv run pytest -k test_parse_mmlu_response`.

**Output:** A function to parse MMLU predictions into the letter of the corresponding answer option.

**(b)** Write a script to evaluate Llama 3.1 8B zero-shot on MMLU: load examples, format prompts, generate outputs, calculate metrics, and serialize examples + generations + scores to disk.

**Output:** A script to evaluate baseline zero-shot MMLU performance.

**(c)** Run the script on Llama 3.1 8B. How many generations fail to parse? If non-zero, what do those examples look like?

**Output:** Number of failed parses; examples of unparseable generations if any.

**(d)** How long does it take to generate responses? Estimate throughput in examples/second.

**Output:** Estimate of MMLU examples/second throughput.

**(e)** How well does the zero-shot baseline perform on MMLU?

**Output:** 1–2 sentences with evaluation metrics.

**(f)** Sample 10 random incorrectly-predicted examples. What sorts of errors does the model make?

**Output:** 2–4 sentence error analysis with examples.

---

### §2.2 — Zero-Shot GSM8K Baseline

**Prompt format:**

```
{question}
Answer:
```

**Evaluation metric:** Parse the generated output by taking the final number as the predicted answer; compare against the gold numeric answer.

**Generation hyperparameters:** Greedy decoding — temperature 0.0, top-p 1.0.

#### Implementation: `gsm8k_baseline` (4 points)

**(a)** Write a function to parse generated outputs into a single numeric prediction. Return `None` if unparseable. Implement adapter `[run_parse_gsm8k_response]` and pass `uv run pytest -k test_parse_gsm8k_response`.

**Output:** A function to parse GSM8K predictions into a single numeric answer.

**(b)** Write a script to evaluate Llama 3.1 8B zero-shot on GSM8K: load examples, format prompts, generate outputs, calculate metrics, and serialize to disk.

**Output:** A script to evaluate baseline zero-shot GSM8K performance.

**(c)** Run the script on Llama 3.1 8B. How many generations fail to parse? If non-zero, what do those examples look like?

**Output:** Number of failed parses; examples of unparseable generations if any.

**(d)** How long does it take to generate responses? Estimate throughput in examples/second.

**Output:** Estimate of GSM8K examples/second throughput.

**(e)** How well does the zero-shot baseline perform on GSM8K?

**Output:** 1–2 sentences with evaluation metrics.

**(f)** Sample 10 random incorrectly-predicted examples. What sorts of errors does the model make?

**Output:** 2–4 sentence error analysis with examples.

---

### §2.3 — Zero-Shot AlpacaEval Baseline

**Prompt format:**

```
{instruction}
```

No additional prompting needed — AlpacaEval instructions are already well-defined inputs.

**Evaluation metric:** A stronger annotator model (Llama 3.3 70B Instruct) judges whether the model's output is preferred over a GPT-4 Turbo reference output. The **winrate** is the proportion of outputs preferred over the reference; **length-controlled winrate** adjusts for response length bias.

**Generation hyperparameters:** Greedy decoding — temperature 0.0, top-p 1.0.

#### Implementation: `alpaca_eval_baseline` (4 points)

**(a)** Write a script to collect Llama 3.1 8B zero-shot predictions on AlpacaEval. Load instructions, generate outputs, and serialize to a JSON array with the following fields per entry:
- `instruction` — the instruction
- `output` — the model's response
- `generator` — a string identifier for the model (e.g., `llama-3.1-8b-base`), same across all entries
- `dataset` — the dataset identifier from the original AlpacaEval data

```python
for example in eval_set:
    example["output"] = generate(example["instruction"])
    example["generator"] = "my_model"

with open("output.json", "w") as fout:
    json.dump(eval_set, fout)
```

**Output:** A script to generate zero-shot outputs on AlpacaEval.

**(b)** How long does it take to generate responses? Estimate throughput in examples/second.

**Output:** Estimate of AlpacaEval examples/second throughput.

**(c)** Run the AlpacaEval evaluator (requires 2 GPUs with >80 GB each):

```bash
uv run alpaca_eval --model_outputs <path_to_model_predictions.json> \
    --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \
    --base-dir '.'
```

What is the winrate and length-controlled winrate against GPT-4 Turbo using Llama 3.3 70B Instruct as annotator?

**Output:** 1–2 sentences with winrate and length-controlled winrate.

**(d)** Sample 10 random examples where the model is dispreferred (annotations at `scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json`). Why is the model dispreferred? Any cases where you disagree with the automatic evaluator?

**Output:** 2–4 sentence error analysis with examples.

---

### §2.4 — Zero-Shot SimpleSafetyTests Baseline

**Prompt format:**

```
{instruction}
```

No additional prompting needed.

**Evaluation metric:** Llama 3.3 70B Instruct judges each instruction-response pair as safe or unsafe. The metric is the proportion of outputs judged safe.

**Generation hyperparameters:** Greedy decoding — temperature 0.0, top-p 1.0.

#### Implementation: `sst_baseline` (4 points)

**(a)** Write a script to collect Llama 3.1 8B zero-shot predictions on SimpleSafetyTests. Load instructions, generate outputs, and serialize in JSON-lines format. Each JSON object must contain at minimum:
- `prompts_final` — the instruction
- `output` — the model's response

**Output:** A script to generate zero-shot outputs on SimpleSafetyTests.

**(b)** How long does it take to generate responses? Estimate throughput in examples/second.

**Output:** Estimate of SimpleSafetyTests examples/second throughput.

**(c)** Run the safety evaluator (requires 2 GPUs with >80 GB each):

```bash
uv run python scripts/evaluate_safety.py \
    --input-path <path_to_model_predictions.jsonl> \
    --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path <path_to_write_output.jsonl>
```

What proportion of model outputs are judged as safe?

**Output:** 1–2 sentences with the proportion of safe outputs.

**(d)** Sample 10 random examples judged as unsafe. In what sorts of cases does the model produce unsafe outputs? Any cases where you disagree with the automatic evaluator?

**Output:** 2–4 sentence error analysis with examples.

---

## Section 3: Instruction Fine-Tuning

From inspecting zero-shot baseline outputs, it is often difficult to get language models to reliably follow instructions via prompting alone. This section explicitly fine-tunes Llama 3.1 to follow instructions using supervised fine-tuning (SFT) on paired (prompt, response) demonstrations.

### §3.1 — Looking at Instruction Tuning Data

The instruction fine-tuning data is a mix of **UltraChat-200K** and **SafetyTunedLlamas**, processed into a single-turn format (one prompt, one response).

**Cluster paths:**
- `/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz`
- `/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz`

**Fallback download URLs:**
- `https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz`
- `https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz`

#### Implementation: `look_at_sft` (4 points)

Look through 10 random examples in the training dataset. What NLP tasks are represented (e.g., question answering, sentiment analysis)? Comment on data quality for both prompts and responses.

**Output:** 2–4 sentences describing the tasks represented and data quality, with concrete examples.

---

### §3.2 — Implementing Instruction Fine-Tuning

#### §3.2.1 — Data Loader

Instruction-tuning examples are formatted using the **Alpaca template** (file: `cs336_alignment/prompts/alpaca_sft.prompt`):

```
Below is an instruction that describes a task. Write a response that appropriately
completes the request.

### Instruction:
{prompt}

### Response:
{response}
```

All documents are concatenated into a single token sequence with a delimiter between them (Llama 3.1 8B uses `<|end_of_text|>`). The sequence is then split into non-overlapping chunks of length `seq_length` (the final chunk is dropped if shorter than `seq_length`). Each chunk becomes one training example — inputs are the chunk tokens, labels are the same tokens shifted by one.

**Example:** Token IDs `[0, 1, 2, ..., 10]` with `seq_length=4` → batches `[[0,1,2,3], [4,5,6,7]]` (length = 2).

#### Implementation: `data_loading` (3 points)

**(a)** Implement a `torch.utils.data.Dataset` subclass for instruction tuning:

```python
def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
    """
    tokenizer:    transformers tokenizer for encoding the data
    dataset_path: path to instruction tuning data
    seq_length:   desired sequence length
    shuffle:      if True, shuffle documents before concatenation
    """

def __len__(self):
    """Returns number of sequences (non-overlapping chunks of seq_length)."""

def __getitem__(self, i):
    """
    Returns dict with:
      input_ids: torch.Tensor of shape (seq_length,)
      labels:    torch.Tensor of shape (seq_length,)
    """
```

Implement adapter `[adapters.get_packed_sft_dataset]` and pass `uv run pytest -k test_packed_sft_dataset`.

**Output:** A `Dataset` subclass that generates packed sequences for instruction tuning.

**(b)** Implement a function that returns batches from the Dataset. Accepts: (1) a dataset, (2) desired batch size, (3) whether to shuffle examples before batching. Iterating through all batches constitutes one epoch. `torch.utils.data.DataLoader` may be useful.

Implement adapter `[adapters.run_iterate_batches]` and pass `uv run pytest -k test_iterate_batches`.

**Output:** A batching function that iterates over the dataset in epoch-sized passes.

---

#### §3.2.2 — Training Script

**Loading the model** (bfloat16 + FlashAttention-2):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

**Forward pass and loss:**

```python
input_ids = train_batch["input_ids"].to(device)
labels    = train_batch["labels"].to(device)
logits    = model(input_ids).logits
loss      = F.cross_entropy(..., ...)
```

**Saving the model:**

```python
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)
```

**Gradient accumulation** — an 80 GB GPU supports a batch size of ~2 sequences at 512 tokens in bfloat16. To achieve an effective batch size of 32, accumulate gradients over `k=16` steps:

```python
gradient_accumulation_steps = 16

for idx, (inputs, labels) in enumerate(data_loader):
    logits = model(inputs)
    loss   = loss_fn(logits, labels) / gradient_accumulation_steps
    loss.backward()

    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Implementation: `sft_script` (4 points)

Write a training script to fine-tune Llama 3.1 8B base on the instruction tuning data. The script should support:
- Configurable model and optimizer hyperparameters
- Gradient accumulation for large effective batch sizes
- Periodic logging of training and validation performance (console and/or Weights & Biases)

**Output:** A training script for instruction fine-tuning with gradient accumulation and logging.

---

#### Implementation: `sft` — Run Instruction Tuning (6 points, ~24 H100 hrs)

Fine-tune Llama 3.1 8B base on the instruction tuning data. Recommended setup:
- 1 epoch, context length 512 tokens, total batch size 32 sequences per gradient step
- Learning rate 2e-5 with cosine decay and linear warmup (3% of total training steps)
- Save model and tokenizer after training — the checkpoint is used in §4

**Output:** Description of training setup, final validation loss, and learning curve. Serialized model and tokenizer checkpoint for use in §4.

---

## Section 4: Evaluating the Instruction-Tuned Model

Now that the model has been instruction-tuned, evaluate it on the same benchmarks as §2 using identical prompts and generation settings, to allow direct comparison against the zero-shot baseline.

---

### §4.1 — MMLU

#### Implementation: `mmlu_sft` (4 points)

**(a)** Write a script to evaluate the instruction-tuned model on MMLU, formatting inputs using the same Alpaca instruction-tuning prompt used during training. Measure throughput in examples/second and compare to the zero-shot baseline.

**Output:** 1–2 sentences with MMLU throughput estimate and comparison to the zero-shot baseline.

**(b)** How well does the instruction-tuned model perform on MMLU? How does this compare to the zero-shot baseline?

**Output:** 1–2 sentences with evaluation metrics and comparison to the zero-shot baseline.

**(c)** Sample 10 random incorrectly-predicted examples. What errors does the model make? How do the instruction-tuned outputs qualitatively differ from the zero-shot baseline outputs?

**Output:** 2–4 sentence error analysis with examples.

---

### §4.2 — GSM8K

#### Implementation: `gsm8k_sft` (4 points)

**(a)** Write a script to evaluate the instruction-tuned model on GSM8K, using the Alpaca prompt format. Measure throughput in examples/second and compare to the zero-shot baseline.

**Output:** 1–2 sentences with GSM8K throughput estimate and comparison to the zero-shot baseline.

**(b)** How well does the instruction-tuned model perform on GSM8K? How does this compare to the zero-shot baseline?

**Output:** 1–2 sentences with evaluation metrics and comparison to the zero-shot baseline.

**(c)** Sample 10 random incorrectly-predicted examples. What errors does the model make? How do the outputs qualitatively differ from the zero-shot baseline?

**Output:** 2–4 sentence error analysis with examples.

---

### §4.3 — AlpacaEval

#### Implementation: `alpaca_eval_sft` (4 points)

**(a)** Write a script to collect the instruction-tuned model's predictions on AlpacaEval. Measure throughput in examples/second and compare to the zero-shot baseline.

**Output:** 1–2 sentences with AlpacaEval throughput estimate and comparison to the zero-shot baseline.

**(b)** Run the AlpacaEval evaluator (requires 2 GPUs with >80 GB each):

```bash
uv run alpaca_eval --model_outputs <path_to_model_predictions.json> \
    --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \
    --base-dir '.'
```

What is the winrate and length-controlled winrate against GPT-4 Turbo using Llama 3.3 70B Instruct as annotator? How does this compare to the zero-shot baseline?

**Output:** 1–3 sentences with winrate and length-controlled winrate, and comparison against the zero-shot baseline.

**(c)** Sample 10 random examples where the fine-tuned model is dispreferred versus GPT-4 Turbo (annotations at `scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json`; entries where `"preference" == 1.0` indicate GPT-4 Turbo was preferred). Why is the fine-tuned model dispreferred? Any cases where you disagree with the automatic evaluator?

**Output:** 2–4 sentence error analysis with examples.

---

### §4.4 — SimpleSafetyTests

#### Implementation: `sst_sft` (4 points)

**(a)** Write a script to collect the instruction-tuned model's predictions on SimpleSafetyTests. Measure throughput in examples/second and compare to the zero-shot baseline.

**Output:** 1–2 sentences with SimpleSafetyTests throughput estimate and comparison to the zero-shot baseline.

**(b)** Run the safety evaluator (requires 2 GPUs with >80 GB each):

```bash
uv run python scripts/evaluate_safety.py \
    --input-path <path_to_model_predictions.jsonl> \
    --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path <path_to_write_output.jsonl>
```

What proportion of model outputs are judged safe? How does this compare to the zero-shot baseline?

**Output:** 1–2 sentences with the proportion of safe outputs and comparison to the zero-shot baseline.

**(c)** Sample 10 random examples where the fine-tuned model's response is judged unsafe. In what cases does the model produce unsafe outputs? Any cases where you disagree with the automatic evaluator?

**Output:** 2–4 sentence error analysis with examples.

---

### §4.5 — Red-Teaming the Instruction-Tuned Model

Red-teaming is an evaluation method that attempts to elicit undesirable or unsafe model behaviors to understand failure modes and guide improvements (Ganguli et al., 2022).

#### Implementation: `red_teaming` (4 points)

**(a)** Beyond the examples listed in the dataset, what are three other possible ways language models might be misused?

**Output:** 1–3 sentences with three examples of potential language model misuse.

**(b)** Attempt to prompt the fine-tuned model to assist with three different potentially malicious applications. For each, describe the methodology, results, and qualitative takeaways — including whether you were successful, how long you tried, and what strategies you employed.

**Output:** For each of three malicious applications, a 2–4 sentence description of the red-teaming procedure and results.

---

## Section 5: Reinforcement Learning from Human Feedback & DPO

During SFT, the model is trained to imitate responses from a fixed set of high-quality examples — but this is often insufficient to mitigate undesirable behavior learned during pre-training. **Reinforcement Learning from Human Feedback (RLHF)** goes further: it elicits responses from the model itself and rewards or penalizes them based on an assessment of quality and appropriateness.

### RLHF Background

In RLHF (Ouyang et al., 2022), the process after SFT is:
1. Generate K responses per prompt from the SFT model
2. Have human annotators rank those responses
3. Fit a **reward model** `r_θ(x, y)` — a scalar-valued head on top of the SFT model — to agree with human rankings using the loss:

```
ℓ_r(x, yw, yl) = -log σ(r_θ(x, yw) - r_θ(x, yl))
```

where `yw` is the preferred response, `yl` is the rejected response, and `σ` is the sigmoid function.

4. Optimize the LM as a policy `π_θ` using PPO against `r_θ`, with a KL-divergence penalty to prevent the policy from drifting too far from the SFT model and an auxiliary pre-training language modeling loss to prevent capability degradation.

RLHF has many moving parts and has been reportedly difficult to reproduce. **Direct Preference Optimization (DPO; Rafailov et al., 2023)** offers a simpler and often equally effective alternative — no reward model, no RL loop.

---

### §5.1 — The DPO Objective

DPO derives a reparameterization of the optimal reward model directly in terms of the optimal policy:

```
r(x, y) = β log [π_r(y|x) / π_ref(y|x)] + β log Z(x)
```

where `π_ref` is the SFT reference policy, `β` controls the penalty for deviating from `π_ref`, and `Z(x)` is a prompt-dependent partition function that does not depend on the response `y`.

Because the original RLHF loss only depends on the *difference* between rewards, `Z(x)` cancels out, yielding the **per-instance DPO loss**:

```
ℓ_DPO(π_θ, π_ref, x, yw, yl) = -log σ(
    β log [π_θ(yw|x) / π_ref(yw|x)] - β log [π_θ(yl|x) / π_ref(yl|x)]
)
```

Key properties of DPO:
- No sampling from the model during training — only forward passes to compute log-probabilities
- No explicit reward model
- Preference data need not come from human annotators — AI-generated preference pairs work too

---

### §5.2 — Looking at Preference Data

The **Anthropic HH dataset** ("Helpful and Harmless") provides human-annotated preference pairs. The training set uses four subsets:

| File | Cluster Path |
|------|-------------|
| `harmless-base.jsonl.gz` | `/data/a5-alignment/hh/harmless-base.jsonl.gz` |
| `helpful-base.jsonl.gz` | `/data/a5-alignment/hh/helpful-base.jsonl.gz` |
| `helpful-online.jsonl.gz` | `/data/a5-alignment/hh/helpful-online.jsonl.gz` |
| `helpful-rejection-sampled.jsonl.gz` | `/data/a5-alignment/hh/helpful-rejection-sampled.jsonl.gz` |

Hugging Face source: `https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main`

Each line is a JSON object with a `"chosen"` conversation (preferred by human annotators) and a `"rejected"` conversation, both starting from the same prompt.

#### Implementation: `look_at_hh` (2 points)

**(a)** Write a function to load the Anthropic HH dataset, combining all four files into a single training set. Apply the following processing steps:
- Ignore multi-turn conversations (where the human sent more than one message)
- Separate each example into: `instruction` (first human message), `chosen` response, `rejected` response
- Track which source file each example came from

The `gzip` and `json` Python modules will be useful.

**Output:** A Python function that loads and processes the HH dataset into a convenient structure for DPO training.

**(b)** Look at 3 random examples from `helpful` and 3 from `harmless`. What are the main differences between chosen and rejected responses? Do you agree with the annotators' choices?

**Output:** 2–4 sentences commenting on the differences between chosen and rejected responses, with examples.

---

### §5.3 — Implementing the DPO Loss

#### Implementation: `dpo_loss` (2 points)

Write a function that computes the per-instance DPO loss (Equation above). The function receives two LMs (`π_θ` and `π_ref`) that may be on different devices — return the loss on the same device as `π_θ`.

Format the prompt and responses using the **Alpaca template** (same as SFT) and append the end-of-sequence token after each response.

**Simplification:** when computing a difference of conditional log-probabilities under the same model:
```
log π_θ(yw|x) - log π_θ(yl|x)  ≡  log π_θ(x⊕yw) - log π_θ(x⊕yl)
```
The log-probability of the shared prompt `x` cancels out, so unconditional log-probabilities suffice.

```python
def per_instance_dpo_loss(
    lm: PreTrainedModel,        # π_θ — model being optimized
    lm_ref: PreTrainedModel,    # π_ref — frozen reference model
    tokenizer,
    beta: float,
    prompt: str,
    chosen: str,                # yw
    rejected: str,              # yl
) -> torch.Tensor:              # scalar DPO loss
```

Implement adapter `[adapters.per_instance_dpo]` and pass `uv run pytest -k test_per_instance_dpo_loss`.

**Output:** A function that computes the per-instance DPO loss given two LMs and a preference pair.

---

### §5.4 — DPO Training

DPO requires two forward passes per example (through both `π_θ` and `π_ref`), which is GPU-memory intensive. Use the following simplified setup:

- **2 GPUs** — one for `π_ref` (frozen), one for `π_θ` (trained)
- Load two copies of the instruction-tuned model from §3, one per device
- Hold out ~200 examples as a validation set
- Use **gradient accumulation** (as in SFT) for larger effective batch sizes
- Use **RMSprop** optimizer (`torch.optim.RMSprop`) — AdamW requires too much memory without quantization

**Recommended starting hyperparameters:** batch size 64, β = 0.1, learning rate 1e-6.

**Validation metric:** Track the implicit reward model's **classification accuracy** — the proportion of validation examples where `log π_θ(yw|x) > log π_θ(yl|x)` (chosen completion has higher log-probability than rejected).

#### Implementation: `dpo_training` (4 points)

**(a)** Implement the DPO training loop. Train the instruction-tuned Llama 3.1 8B model for 1 epoch over HH. Save the checkpoint with the highest validation accuracy.

**Output:** A DPO training script and a validation accuracy curve during training.

**(b)** Evaluate the DPO-trained model on AlpacaEval (same procedure as `alpaca_eval_sft`). What are the winrate and length-controlled winrate against GPT-4 Turbo using Llama 3.3 70B Instruct as annotator? How does this compare to the SFT model?

**Output:** 1–2 sentences with AlpacaEval winrates and comparison to the SFT baseline.

**(c)** Evaluate the DPO-trained model on SimpleSafetyTests. How does it compare to the SFT model?

**Output:** 1–2 sentences with SimpleSafetyTests results and comparison to the SFT baseline.

**(d)** AlpacaEval and SimpleSafetyTests directly test behaviors demonstrated in HH (instruction following, refusal of harmful prompts). Prior work — including the original Anthropic HH paper — has observed an **"alignment tax"**: aligned models may lose some general capabilities. Evaluate the DPO model on GSM8K and MMLU. What do you observe?

**Output:** 2–3 sentences with GSM8K and MMLU results and discussion of any alignment tax observed.

