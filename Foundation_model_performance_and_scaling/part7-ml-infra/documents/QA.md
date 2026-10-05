# Part 7 — Questions and answers

High-level notes from our discussions. Implementation details are in
[PLAN.md](../PLAN.md), [GRPO mapping](7A_GRPO_MAPPING.md) and the
[cloud runbook](7A_RUNBOOK.md). Current status: the bounded 7A cloud smoke,
artifact uploads and pinned model-weight reload are verified. 7B has not started.

## What did 7A verify, and what remains for 7B?

Two A100 40 GB GPUs ran the verl workload with two training ranks, 338 sharded
parameter tensors per rank, sampled sharded gradients/Adam moments, six optimizer
calls per rank, evaluation and checkpoint save. The upload run had nonzero
gradients on steps 2/3. Its BF16 model export and private Hub uploads completed,
and a new smoke loaded that exported model at the recorded commit successfully.
Logs and receipts are retained locally; the instance has been released.

This is bounded infrastructure verification, not a reliable model-quality gain.
The reload smoke had zero training rewards; 32-prompt evaluation scores varied
between runs. Full-state restore, interruption recovery and Ray operations remain
7B work. Multi-node execution and broader numerical/tokenizer equivalence remain
unverified. Full training checkpoints were intentionally not backed up.

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
Rendered-text equality passed CPU checks, and the adapters were exercised in
cloud generation, grading and evaluation. Full token-ID/response-mask equivalence
to Part 5 has not been established; a successful smoke does not prove that parity.

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

## Did we implement FSDP2 or leave vLLM configuration to verl?

We did not implement either engine. Our YAML selects FSDP2, two-rank sharding,
mixed precision and parameter/optimizer offloading. PyTorch implements FSDP2,
and verl integrates it into its workers. Diagnostics check actual sharding;
configuration flags alone were not our completion evidence.

The YAML's `rollout` section explicitly selects vLLM and sets parallelism,
GPU-memory utilization, sequence limits, sampling and evaluation. verl handles
worker coordination and model-weight synchronization; vLLM handles generation
and KV-cache management. Unspecified settings inherit the pinned framework's defaults.

## What changes when we use more GPUs or multiple instances?

Revisit training node/GPU counts, FSDP sharding, global/microbatch sizes, rollout
parallelism and memory limits together. Multiple machines also need Ray cluster
connectivity, worker placement and durable model/data access. Increasing a rollout
parallelism value alone does not configure multi-node training. Choose the topology
for the model and hardware, then verify actual placement, sharding and performance.

## Why does `math_workload.py` barely mention verl?

It handles data, prompts and rewards without importing the training framework,
so those responsibilities can be checked with lightweight CPU tests.
`r1_agent.py` extends verl's single-turn generation loop to preserve stopping and
sampling settings. `run_7a.py` and the YAML files connect these adapters to the full
framework. Both adapters were used in the verified cloud runs; their small size
reflects delegation to the framework.

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
| `infrastructure/7a_verl/part7A.sh` | Single cloud entry point for setup, preparation, training and optional Hub retention. |
| `infrastructure/7a_verl/environment/requirements.txt` | Deliberately selected direct library versions and exact verl commit. |
| `infrastructure/7a_verl/environment/requirements-cu130.in` | Official CUDA wheel URLs used during resolution. |
| `infrastructure/7a_verl/environment/requirements.lock` | Resolved direct and indirect application dependency versions. |
| `infrastructure/7a_verl/environment/setup_7a.sh` | Isolated cloud installation of matching CUDA 13 wheels and import checks. |
| `infrastructure/7a_verl/config/grpo_smoke.yaml` | Algorithm, batching, FSDP2/vLLM resources, evaluation and checkpoint settings. |
| `infrastructure/7a_verl/config/r1_agents.yaml` | Registration of our generation adapter with verl. |
| `infrastructure/7a_verl/run_7a.py` | Configuration/path assembly, bundle checks, evidence capture and framework launch. |
| `infrastructure/7a_verl/verl_diagnostics.py` | Numerical checks and actual rank/sharding evidence during cloud execution. |
| `infrastructure/7a_verl/hf_artifacts.py` | Authentication/account checks, private repository naming, verl model export, direct uploads and commit receipts. |

This separation keeps installation choices, experiment settings and launch logic
independently reviewable. The runner delegates training to verl rather than
implementing another training loop.

## Do checkpoints have to pass through my laptop?

No. A fresh smoke downloads the original base model automatically and needs no
previous checkpoint. To reuse trained weights, the cloud runner can export/upload
a private Hugging Face model and a later instance can download it by repository
ID and commit revision. Full optimizer/training state is needed only for exact
continuation; its upload is optional and goes directly from cloud to the Hub.
Logs/configuration can be stored independently in a private artifact repository.
The runbook documents these options. Hub file presence, model reload and verified
training recovery are distinct checks; recovery remains part of 7B.

For the fewest manual steps, use `bash infrastructure/7a_verl/part7A.sh --upload-hf`.
It reuses Part 6-style saved Hub authentication or prompts once for a write token,
verifies its account and chooses private repository names automatically. A token
provided through instance secrets makes the flow noninteractive.

## Do I need separate CLI authentication or manual environment setup?

No separate login command is required with `--upload-hf`: the runner prepares its
isolated environment, reuses a saved Hugging Face login or `HF_TOKEN`, or prompts
once for a write token. It verifies the token's account before training and chooses
private destinations automatically. This uses the same Hub authentication as
Part 6. A new VM needs a token again unless credentials are supplied through its
secret configuration. Source must contain the latest changes before running.

## How are uploads named, and can I go back to an earlier model?

The default model repository is
`ACCOUNT/part7-7a-smoke-execution-UTC_TIMESTAMP-RANDOM_SUFFIX`; the evidence
repository adds `-artifacts`. Each default execution has a distinct name.
`--output results/7a/grpo-smoke-001` instead uses
`ACCOUNT/part7-7a-grpo-smoke-001`; choose a new name for each run. Repository
variables can override automatic destinations. `hf_upload.json` records the
repository IDs and exact commit revisions.

Use `--model-id REPOSITORY --model-revision FULL_COMMIT_HASH` to start fresh
training from a selected model's weights. Explicitly uploading again to the same
repository creates a new commit; earlier revisions can still be selected while
retained. This is weights-only restart, not restoration of optimizer or progress.

## Should model and artifact repositories be public?

The current runner enforces private repositories; public model uploads are not
implemented. Public release may suit selected useful models with model cards,
while raw operational evidence and training state can remain private. Your account
dashboard showed more public than private storage, but public capacity should not
be treated as unrestricted backup space. Storage is shared across repositories;
creating more repositories does not increase the account allowance. Check current
usage and the [Hub storage policy](https://huggingface.co/docs/hub/storage-limits)
before retaining many versions. Default artifact uploads exclude full checkpoints.
