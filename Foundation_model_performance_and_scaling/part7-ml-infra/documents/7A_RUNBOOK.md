# 7A cloud smoke preparation

Step 3 prepares configuration and commands. GPU execution has not been verified.
Hardware/spending approval is required before renting or running paid resources.
The proposal remains one node with 2×A100 40 GB, at most two hours, at most
$5/hour for the whole instance, and at most $15 total including storage.

## Cloud prerequisites

Use a fresh cloud CUDA 12.8 development environment with `nvcc`, `git`, `uv`,
two visible GPUs, sufficient shared memory (proposed 16 GB), and durable artifact
storage. Confirm the provider image and driver support before rental. Capture its
image identity/digest in `results/7a/environment/image_identity.txt`; a digest-pinned
deployment image is still pending. Do not reuse Part 5's environment.

## Main workflow — Git and one cloud command

Commit and push the reviewed Part 7 changes from your local checkout to the branch
you intend to run. Git is the preferred source-transfer method. The agent still
needs explicit approval before staging/committing/pushing on your behalf.

On a new cloud instance, clone the repository (with Git access configured):

```bash
git clone git@github.com:TianqGuo/Foundation_model_performance_and_scaling.git /workspace/foundation
cd /workspace/foundation/Foundation_model_performance_and_scaling/part7-ml-infra
bash infrastructure/7a_verl/part7A.sh
```

For an existing checkout, update the intended branch and run again:

```bash
cd /workspace/foundation
git pull --ff-only
cd Foundation_model_performance_and_scaling/part7-ml-infra
bash infrastructure/7a_verl/part7A.sh
```

Clone/check out the intended branch if it differs from the repository default.
No separate environment installation, activation, model download, configuration
check or Python launch is required. The shell runner handles those steps.

Make MATH data available before running: either restore the Part 5 raw files to
the sibling `part5-alignment/data/math/` or `/data/a5-alignment/MATH/`, or transfer
the prepared bundle into Part 7's `results/7a/step2_math_final/`. The script prepares
the bundle automatically when raw data and Part 5 references are available. Git
does not deliver ignored datasets/results, model weights or virtual environments.

## Alternative — transfer uncommitted files

For the current uncommitted changes, rsync is an alternative that does not require
a commit/push. From the local Part 7 directory, substitute the approved SSH host
and destination below. Create destination directories first:

```bash
ssh CLOUD_HOST 'mkdir -p /workspace/part7/results/7a'
rsync -az --exclude=.git --exclude='.venv*' --exclude=__pycache__ \
  AGENTS.md PLAN.md README.md infrastructure workloads tests documents CLOUD_HOST:/workspace/part7/
rsync -az results/7a/step2_math_final CLOUD_HOST:/workspace/part7/results/7a/
```

For a Git checkout, use rsync only for a missing data bundle if needed, adjusting
the destination to the checkout's Part 7 directory. Do not overlay source files
unless you intentionally want to test uncommitted changes.

On cloud, run the single entry point:

```bash
cd /workspace/part7
bash infrastructure/7a_verl/part7A.sh
```

## What the shell runner does

`part7A.sh` calls `infrastructure/7a_verl/environment/setup_7a.sh` when the environment is absent. Later
runs reuse a successfully installed environment after checking setup-input hashes
and imports. An incomplete or stale environment fails explicitly and is preserved.
The wrapper checks the prepared bundle, saves a resolved configuration preview,
then runs the smoke, including automatic model download. If the bundle is missing,
it attempts preparation from discovered Part 5 references and raw MATH data; those
must be present on cloud. Transferring the prepared bundle avoids that dependency.
Console output is retained under `results/7a/execution_TIMESTAMP_SUFFIX/`.
Setup and smoke have 30-minute and 60-minute timeouts respectively.

The setup selects Python 3.12.14 and the exact verl commit, installs the resolved
dependency lock, builds FlashAttention after Torch, checks dependency consistency
and imports, and records the installed packages. FlashAttention compilation is
cloud-only and can consume meaningful setup time; stop if the approved time/budget
cannot accommodate setup and the smoke. Setup intentionally fails on an existing
environment/source directory rather than silently changing it. Save setup console
output too. The lock is metadata-resolved; it is not a tested GPU environment.

The launcher automatically calls Hugging Face `snapshot_download` **on cloud**
when the default model snapshot is missing, as Part 5 did. It downloads the snapshot
into `assets/models/Qwen2.5-Math-1.5B`; no manual download command is needed.
Subsequent runs reuse the local snapshot. Automatic preparation occurs only after
the cloud GPU and verl-commit checks; `--check-config` never downloads models.
For a reproducible download, supply `--model-revision COMMIT`; the Hugging Face
cache supports reusing downloads. Record the repository revision; model-file
hashes capture the actual files used. Allow download time within the smoke timeout.
The runner also discovers Part 7's `assets/Qwen2.5-Math-1.5B` and the established
cloud path `/data/a5-alignment/models/Qwen2.5-Math-1.5B`. An existing cloud snapshot
can be supplied with `--model-path`; an invalid explicit path fails rather than
downloading the default model into it. No local model
download/loading is needed. Changing the model requires prompt/tokenizer and
memory validation; the smoke's batch/resource settings are intentionally explicit.

## Optional direct commands

The wrapper performs configuration checking and execution automatically. These
lower-level commands are available for troubleshooting; they are not required
steps when using `bash infrastructure/7a_verl/part7A.sh`.

Configuration composition reads YAML and bundle hashes; it does not import verl,
load a tokenizer or perform a model forward:

```bash
.venv-7a/bin/python infrastructure/7a_verl/run_7a.py --check-config > resolved_preview.yaml
```

Run on the approved cloud host, using a fresh output directory:

```bash
set -o pipefail
timeout --signal=INT --kill-after=60s 60m .venv-7a/bin/python -u infrastructure/7a_verl/run_7a.py --cloud \
  --output results/7a/cloud_smoke_01 2>&1 | tee results/7a/cloud_smoke_01_console.log
```

Optional overrides: `--model-path`, `--model-id`, `--model-revision`, `--data`,
`--verl-source`, `--output`. `--model-id` selects another repository for automatic
cloud download. `--model-revision` applies to automatic preparation; do not combine
it with an explicit existing `--model-path`.
Pass these options to the shell entry point, for example:

```bash
bash infrastructure/7a_verl/part7A.sh --model-path /data/a5-alignment/models/Qwen2.5-Math-1.5B
```
The launcher starts an isolated local Ray cluster through verl and shuts it down
on exit. It delegates rollout, reward, GRPO, synchronization and checkpointing to
verl. Four prompts × four responses form each rollout; two PPO epochs and three
rollouts request six global optimizer updates. Evaluate the same 32 saved prompts
before/after training; this is an infrastructure smoke, not the Part 5 128-prompt
comparison. Fail on overlength prompts rather than silently filtering/truncating.

## Evidence and review

Retain resolved configuration, package/source/model/data hashes, console output,
rollout/validation dumps, checkpoints and `diagnostics/*.jsonl`. The hooks delegate
the policy objective to verl's vanilla implementation. Before the first optimizer
step, compare valid-token frozen actor log probabilities across inference/update
paths; abort both ranks together if nonfinite or maximum absolute difference
exceeds **0.02** (an engineering diagnostic threshold, not FP32-equivalence proof).
Review the measured differences even when below the threshold. This does not
compare vLLM probabilities or prove tokenizer agreement with Part 5.

After the first update, each training rank records world size, visible devices,
global/local parameter and gradient shapes, DTensor placements/mesh, sampled Adam
state shapes, and process CUDA peak allocated/reserved memory. Check both rank
records, physical GPU placement and sharded states; configuration alone is not
evidence. Hooks currently sample three sharded tensors; inspect checkpoint metadata
for broader state coverage. Peaks are process allocator measurements, not whole-GPU
utilization or per-phase memory. Use verl's console timing/throughput metrics and
retained `ray_logs/` for framework evidence; no external tracking login is required.

Review initial rendered/tokenized prompts, included `</answer>` and EOS/length
behavior, valid response masks, numerical checks, finite updates, evaluation and
checkpoint contents. Nonzero exit, missing either training rank, unsharded state,
missing checkpoint or missing evaluation leaves 7A incomplete. Training-state
recovery is a later 7B acceptance item; this smoke disables automatic resume.

Copy the complete output directory, console log, environment records and relevant
Ray worker logs to durable storage and verify readability before stopping/deleting
the rental. Do not delete resources until artifacts are retained. Record actual
hardware, elapsed time and spend. Summarize measured results only after review.
