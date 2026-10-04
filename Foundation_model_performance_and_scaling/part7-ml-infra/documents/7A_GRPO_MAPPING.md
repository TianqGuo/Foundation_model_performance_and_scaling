# 7A step 2 — MATH adapters and GRPO mapping

Updated: 2026-10-04. Adapters, data conversion and bounded CPU equation checks
are implemented. No models were loaded or GPU work performed. Full configuration
composition and the cloud agent import remain step 3/cloud checks.

Reference: corrected Part 5 `tight_pilot_FRqpow9W` GRPO run, using
Qwen2.5-Math-1.5B, MATH and `r1_zero.prompt`. Upstream references below are fixed
to verl commit `bec9ef74768dd201881cd4e54cd0385e87caae27`.

## Data, prompt and reward adapters

[math_workload.py](../workloads/math_workload.py) reads the existing JSONL files,
reuses Part 5's evaluation-index helper, and writes Parquet records with `prompt`,
`data_source`, `ability`, `reward_model` and original source indices in `extra_info`.
It retains the full solution ground truth, matching Part 5's precedence; it does
not substitute verl's boxed-answer-only data preparation or grader.

The data bundle contains byte-identical snapshots of the Part 5 grader, prompt
and selection helper, preserving source attribution. The manifest fingerprints
input files, adapters, snapshots, output files and evaluation selection. Cloud
reward calls must pass the bundled grader's absolute path as
`reward.custom_reward_function.reward_kwargs.grader_path`; the callback path is
the absolute cloud path to `workloads/math_workload.py`, with name `compute_score`.
Do not install/import the Part 5 training environment. The custom reward returns
verl's `score` plus `acc`, `format_reward` and `answer_reward` diagnostics. Reward
is 1 only for a correct, correctly formatted response; format alone earns 0.

The initial Parquet prompt is a single user message whose content is the exact
rendered raw Part 5 prompt. Load `raw_prompt.jinja` as the **string** value of
`actor_rollout_ref.model.custom_chat_template`, for both trainer and rollout
tokenizers. It adds no role wrappers or generation suffix. CPU rendered-text
equality passed; token-ID equality, special tokens and response masks still need
cloud validation. Larger models can replace this template/configuration without
changing Ray orchestration.

[r1_agent.py](../workloads/r1_agent.py) subclasses verl's single-turn loop and
delegates all generation to it. Its only change is sampling parameters: stop at
`</answer>` while retaining that string, training minimum 4 tokens, and separate
evaluation request seeds. Step 3 must register the Hydra agent list entry:

```yaml
- name: r1_single_turn
  _target_: workloads.r1_agent.R1SingleTurnAgentLoop
```

Set `actor_rollout_ref.rollout.agent.agent_loop_config_path` to that list file and
`actor_rollout_ref.rollout.agent.default_agent_loop=r1_single_turn`; make the
Part 7 project importable by all workers. Model execution remains cloud-only.
Sources: [RL dataset](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/utils/dataset/rl_dataset.py),
[agent construction](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/experimental/agent_loop/agent_loop.py),
[reward callback](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/trainer/ppo/reward.py).

## Explicit estimator and optimizer mapping

| Part 5 reference | Proposed verl configuration / interpretation |
| --- | --- |
| Mean group baseline, sample std + 1e-6 | `algorithm.adv_estimator=grpo`, `algorithm.norm_adv_by_std_in_grpo=true`. Pinned function uses sample std and epsilon 1e-6; require group size >=2. Mixed and zero-variance groups passed CPU comparison. |
| Token importance ratio, clipping 0.2/0.2 | `actor.policy_loss.loss_mode=vanilla`, `actor.clip_ratio=0.2`, `actor.clip_ratio_low=0.2`, `actor.clip_ratio_high=0.2`, under `actor_rollout_ref`. |
| No dual clipping | Set `actor.clip_ratio_c=1e10`. verl clamps log ratios to [-20,20], so this bound exceeds its maximum ratio and makes the extra negative-advantage cap inactive. Default 3.0 changes the objective. Log-ratio clamping remains a documented difference. |
| Mean of per-response token means | `actor.loss_agg_mode=seq-mean-token-mean`; use the new worker's global response count and rank scaling. Unequal-length loss/gradient and simulated two-rank accumulation checks passed. |
| No KL or entropy objective | `actor.use_kl_loss=false`, `actor.entropy_coeff=0`, `algorithm.use_kl_in_reward=false`; no reference model, learned reward model or critic for GRPO. |
| AdamW lr 1e-5, betas (0.9,0.95), zero weight decay, grad clip 1 | Set `actor.optim.lr=1e-5`, `.betas=[0.9,0.95]`, `.weight_decay=0`, `.optimizer=AdamW`, `.optimizer_impl=torch.optim`, `.lr_scheduler_type=constant`, `.lr_warmup_steps=0`, `.lr_warmup_steps_ratio=0`, `.clip_grad=1.0`; also set `actor.grad_clip=1.0`. Retain optimizer epsilon 1e-8. |
| Recompute frozen trainer log probabilities | `algorithm.rollout_correction.bypass_mode=false`, `.rollout_is=null`, `.rollout_rs=null`; use HF trainer rescoring, not vLLM sampling log probabilities as the old policy. |
| FP32 log-softmax and stable microbatch layout | No proven equivalent switch in the selected engine. verl uses its FlashAttention CE/fallback path and nested/packed tensors, and promotes old log probabilities afterward. Disable fused model kernels, dynamic batching, compilation, PPO shuffling and batch balancing initially; measure frozen-policy agreement before training. Do not claim FP32 normalization or Part 5 stable-layout parity from these settings. |

Sources: [GRPO and policy loss equations](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/trainer/ppo/core_algos.py),
[loss/rank normalization](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/utils/losses.py),
[log-probability implementation](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/utils/torch_functional.py).

## Counts, rollout reuse and evaluation

In the selected **new** worker, `data.train_batch_size` counts prompts.
`rollout.n` counts responses per prompt. The trainer multiplies
`actor.ppo_mini_batch_size` by `rollout.n` before dispatch, so that knob also
counts prompts here; the resulting optimizer batch counts responses globally.
Local minibatches divide by training data-parallel size. Microbatch size is
responses **per GPU**, not prompts or a global accumulation count.

| Configuration | Prompts / iteration | Responses / iteration | Global responses / optimizer update | PPO epochs | Updates / iteration |
| --- | --- | --- | --- | --- | --- |
| First smoke proposal | 4 | 16 (`n=4`) | 16 (`ppo_mini_batch_size=4`) | 2 | 2 |
| Part 5 exploratory pilot mapping | 16 | 64 (`n=4`) | 32 (`ppo_mini_batch_size=8`) | 4 | 8 |

With two training ranks and microbatch size 2, smoke updates process 8 responses
per rank through four microbatches. Three smoke rollout iterations therefore
mean six global optimizer updates, not six per-rank updates added together.
These counts assume complete, divisible batches; step 3 must check actual counts
and scheduler stepping in the framework. No pilot rerun is required.
Source: [trainer update dispatch](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/trainer/ppo/ray_trainer.py),
[worker minibatching](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/engine_workers.py).

Use data shuffle seed 42, actor minibatch seed 42, no within-rollout PPO shuffle,
and temperature 1.0/top-p 1/top-k -1. Training uses engine RNG streams (base seed
plus replica rank), unlike Part 5's per-iteration seed 42 + iteration. Never seed
every response in a group identically. Seed behavior and scheduling differences
prevent bitwise cross-framework comparisons.

Evaluation uses the saved 32-row Parquet subset, no validation shuffle, one greedy
response per prompt, temperature 0 and independent request seed 12345 + original
source index. Its 256-token response cap matches the bounded reference; record
length-cap incidence and retained stop/EOS behavior. verl collapses some finish
reasons, so response length at the cap alone cannot prove a length stop. Use
initial and final evaluation; preserve every denominator, reward diagnostic and
the exact selected indices. The 32-example smoke is not the 128-example pilot.

## Data audit and CPU verification

The raw files have 7,499 training and 5,000 validation records; their SHA-256
fingerprints match the corrected Part 5 pilot. Training solutions at source
indices 715 and 5064 are empty; validation solutions are nonempty. Preparation
fails by default on empty selected targets. Explicit `--drop-empty-solutions`
excludes only empty training solutions within the selected prefix and records
each original index/reason without changing raw files or replacing skipped rows.
This is a declared data-protocol difference from Part 5.

Actual prepared bundle: `results/7a/step2_math_final/`, with 1,023 usable prompts
from the first 1,024 training records (excluded index 715), 32 independently
selected evaluation prompts, reference snapshots and a manifest. Keep the bundle
out of source commits. Another explicit output directory avoids overwrites.

```bash
python workloads/math_workload.py --output results/7a/prepared_math --drop-empty-solutions
VERL_SOURCE=/path/to/pinned/verl python -m unittest discover -s tests -v
```

13 CPU tests passed using Python 3.13, Torch already available locally, and
isolated `/tmp` CPU dependencies (PyArrow 19.0.1, math-verify 0.7.0,
latex2sympy2_extended 1.10.1, pylatexenc 2.10, SymPy 1.14.0, Jinja2 3.1.6).
Cloud Python 3.12 and full dependency resolution remain unverified. Equation
tests hash-check and AST-extract exact pinned functions to avoid importing
verl/Ray/CUDA; they do not establish distributed integration correctness.
Next: step 3 configuration, isolated environment preparation, cloud runner,
numerical/sharding evidence collection and runbook. Paid provisioning remains
pending hardware and budget agreement.
