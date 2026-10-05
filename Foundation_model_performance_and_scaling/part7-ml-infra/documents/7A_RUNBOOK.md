# 7A cloud smoke preparation

Step 3 prepares configuration and commands. GPU execution has not been verified.
Hardware/spending approval is required before renting or running paid resources.
The proposal remains one node with 2×A100 40 GB, at most two hours, at most
$5/hour for the whole instance, and at most $15 total including storage.

## Transfer and environment

Use a fresh cloud CUDA 12.8 development environment with `nvcc`, `git`, `uv`,
two visible GPUs, sufficient shared memory (proposed 16 GB), and durable artifact
storage. Confirm the provider image and driver support before rental. Capture its
image identity/digest in `results/7a/environment/image_identity.txt`; a digest-pinned
deployment image is still pending. Do not reuse Part 5's environment.

From the local Part 7 directory, transfer uncommitted source and the prepared
bundle without a Git push. Substitute the approved SSH host and destination:

```bash
rsync -az --exclude=.git --exclude='.venv*' --exclude=__pycache__ \
  AGENTS.md PLAN.md README.md environment workloads tests documents CLOUD_HOST:/workspace/part7/
rsync -az results/7a/step2_math_final CLOUD_HOST:/workspace/part7/results/7a/
```

On cloud, install into the new environment:

```bash
cd /workspace/part7
timeout --signal=INT --kill-after=60s 30m bash environment/setup_7a.sh --cloud
```

The setup selects Python 3.12.14 and the exact verl commit, installs the resolved
dependency lock, builds FlashAttention after Torch, checks dependency consistency
and imports, and records the installed packages. FlashAttention compilation is
cloud-only and can consume meaningful setup time; stop if the approved time/budget
cannot accommodate setup and the smoke. Setup intentionally fails on an existing
environment/source directory rather than silently changing it. Save setup console
output too. The lock is metadata-resolved; it is not a tested GPU environment.

Place a complete Qwen2.5-Math-1.5B Hugging Face snapshot at
`assets/models/Qwen2.5-Math-1.5B` **on cloud**, recording the repository revision.
The runner also discovers Part 7's `assets/Qwen2.5-Math-1.5B` and the established
cloud path `/data/a5-alignment/models/Qwen2.5-Math-1.5B`. An existing cloud snapshot
can be supplied with `--model-path`. No local model
download/loading is needed. Changing the model requires prompt/tokenizer and
memory validation; the smoke's batch/resource settings are intentionally explicit.

## Check and run

Configuration composition reads YAML and bundle hashes; it does not import verl,
load a tokenizer or perform a model forward:

```bash
.venv-7a/bin/python workloads/run_7a.py --check-config > resolved_preview.yaml
```

Run on the approved cloud host, using a fresh output directory:

```bash
set -o pipefail
timeout --signal=INT --kill-after=60s 60m .venv-7a/bin/python -u workloads/run_7a.py --cloud \
  --output results/7a/cloud_smoke_01 2>&1 | tee results/7a/cloud_smoke_01_console.log
```

Optional overrides: `--model-path`, `--data`, `--verl-source`, `--output`.
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
