# Part 7 — Questions and answers

High-level notes from our discussions. Implementation details are in
[PLAN.md](../PLAN.md), [GRPO mapping](7A_GRPO_MAPPING.md) and the
[cloud runbook](7A_RUNBOOK.md). Current status: 7A CPU preparation is complete;
cloud training and distributed integration remain unverified.

## Which model and dataset are we starting with? Can we switch later?

Start with **Qwen2.5-Math-1.5B and MATH**, carried forward from Part 5.
Qwen is the model; GRPO/GSPO are training algorithms. A larger model is a later
practice goal. Model paths, prompt handling, sequence limits and resource settings
are configurable, but each switch still needs architecture/tokenizer support and
GPU-memory validation. A path change alone does not establish compatibility.

## Where do the 1,023 training and 32 evaluation prompts come from?

They are subsets of Part 5's existing MATH files, not newly generated examples.
Preparation selects the first 1,024 training records and explicitly excludes one
empty solution, leaving 1,023. The 32 evaluation records are selected from Part 5's
validation file using its fixed-subset helper with seed 12345. Saved original
indices and file hashes identify the exact selections.

## Is 32 evaluation prompts enough? Is that the Part 5 evaluation size?

It is enough for a bounded infrastructure smoke to check that evaluation works,
but too small for a reliable model-quality comparison. The corrected Part 5 pilot
used **128 fixed evaluation prompts**. A later comparison should align the saved
subset, generation limits and grading protocol, and retain uncertainty and seed
limitations. Smoke accuracy is not directly comparable to that pilot.

## Do we reuse Part 5's prompt and reward function?

Yes. `math_workload.py` preserves the rendered Part 5 prompt and calls its
`r1_zero_reward_fn` through a thin reward callback. Prepared bundles include exact
reference snapshots so cloud workers do not depend on the Part 5 environment.
The training reward is binary: a response must be correct and correctly formatted.
Rendered-text equality passed CPU checks; tokenizer and response-mask behavior
still need cloud validation.

## Why convert JSONL to Parquet? Does verl require it?

Parquet is a supported, convenient format for the structured records our adapter
produces: prompt messages, ground truth, dataset identity and source metadata.
The selected verl dataset loader also supports JSON input; Parquet is our chosen
interchange format, not an algorithm requirement. The important contract is the
record schema and its interpretation. Conversion also creates a portable bundle
with explicit provenance and a saved evaluation selection.

## What does verl coordinate, and what do Ray, FSDP2 and vLLM do?

verl coordinates the RL workflow: data batches, response generation, reward
calculation, advantages, policy updates, weight synchronization, evaluation and
checkpointing. Ray manages processes and resource placement; PyTorch FSDP2 shards
training state; vLLM generates responses. verl includes algorithm implementations
and extension points as well as orchestration. Ray Train does not wrap our verl
workload; it is reserved for optional SFT later in Part 7.

## Why does `math_workload.py` barely mention verl?

It handles data, prompts and rewards without importing the training framework,
so those responsibilities can be checked with lightweight CPU tests.
`r1_agent.py` extends verl's single-turn generation loop to preserve stopping and
sampling settings. `run_7a.py` and the YAML files connect these adapters to the full
framework. Small adapters do not mean verl is absent from the planned training run.

## Why call `run_ppo` when we want GRPO? Is it a placeholder?

No. `run_ppo` is the selected verl release's shared training entry point, including
for GRPO. Our YAML sets `algorithm.adv_estimator: grpo` and generates four responses
per prompt. Advantages use group-relative rewards, while policy updates use a
clipped objective. GRPO shares this objective machinery with PPO, but does not
require a learned value critic. The entry-point name does not select the algorithm.

## Are we running Part 5's custom GRPO implementation?

We reuse its workload and explicitly map its selected algorithm settings to
**verl's implementation**; we do not execute Part 5's training loop. The mapping
includes group normalization, clipping, loss reduction and optimizer settings.
This is not an untouched default preset or an exact reproduction: numerical
precision, tensor layouts, sampling seeds and distributed execution can differ.
`part7_grpo` is our diagnostic loss wrapper; it checks frozen-policy agreement and
then delegates the objective to verl's existing vanilla clipped loss.

## Can we implement a customized GRPO algorithm within verl?

Yes. The Part 5 algorithm's mathematical components can be adapted to verl's
interfaces. Supported settings belong in configuration; custom rewards use
callbacks; custom advantages and policy objectives can use registered functions.
verl can retain responsibility for rollout, distributed execution, synchronization
and checkpointing while those functions define the algorithm.

This is adaptation, not loading the entire Part 5 script unchanged. Custom code
must respect batch shapes, response masks, loss reduction and distributed gradient
scaling. Exact FP32 log-softmax or stable microbatch behavior may require engine
or worker changes. More extensive schedule/backend changes involve deeper
integration. The framework is flexible, but not every choice is a YAML switch.
No custom Part 5 algorithm port is currently requested or implemented.

## Why are there separate environment, configuration and runner files?

| Files | Purpose |
| --- | --- |
| `infrastructure/7a_verl/environment/requirements.txt` | Deliberately selected direct library versions and exact verl commit. |
| `infrastructure/7a_verl/environment/requirements-cu130.in` | Official CUDA wheel URLs used during resolution. |
| `infrastructure/7a_verl/environment/requirements.lock` | Resolved direct and indirect application dependency versions. |
| `infrastructure/7a_verl/environment/setup_7a.sh` | Isolated cloud installation of matching CUDA 13 wheels and import checks. |
| `infrastructure/7a_verl/config/grpo_smoke.yaml` | Algorithm, batching, FSDP2/vLLM resources, evaluation and checkpoint settings. |
| `infrastructure/7a_verl/config/r1_agents.yaml` | Registration of our generation adapter with verl. |
| `infrastructure/7a_verl/run_7a.py` | Configuration/path assembly, bundle checks, evidence capture and framework launch. |
| `infrastructure/7a_verl/verl_diagnostics.py` | Numerical checks and actual rank/sharding evidence during cloud execution. |

This separation keeps installation choices, experiment settings and launch logic
independently reviewable. The runner delegates training to verl rather than
implementing another training loop.
