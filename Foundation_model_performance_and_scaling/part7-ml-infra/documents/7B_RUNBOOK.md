# 7B — Ray recovery runbook

The cloud runner and recovery hooks are prepared and CPU-checked. **Cloud
verification is pending; 7B is not complete.** Reuse the pinned 7A environment.
No GPU dependency/backend change or VM provisioning is introduced.

## New cloud instance

Agree the current offer and spending cap before renting. The initial target is
one host with two A100 40 GB GPUs, the same workload as 7A, preferably at least 64 GiB host RAM for
checkpoint deserialization and **at least 100 GiB free disk**. Full checkpoints
contain FP32 weights/Adam state and are much larger than BF16 exports. Several
local copies are kept for this bounded experiment; nothing is silently deleted.
Allow private Hub capacity for one full checkpoint (roughly 20 GB for this model;
actual size is recorded by the manifest). Larger models need a separate memory,
disk and compatibility decision.

Deliver source using the usual Git clone/pull workflow after your approved commit
and push. For uncommitted source, rsync from the local Part 7 directory is an
alternative; substitute the SSH host/port and repository destination:

```bash
rsync -az --exclude=__pycache__ -e "ssh -p PORT" \
  AGENTS.md PLAN.md README.md infrastructure workloads tests documents \
  CLOUD_HOST:/workspace/REPOSITORY/Foundation_model_performance_and_scaling/part7-ml-infra/
```

The checkout needs Part 5's existing MATH inputs/references, or transfer the prepared
`results/7a/step2_math_final/` bundle. From Part 7 on the cloud, run:

```bash
bash infrastructure/7b_ray/part7B.sh
```

The shell creates/reuses `.venv-7a`, verifies setup inputs/imports and two GPUs,
selects the image's CUDA 13 toolkit and prepares missing data. `--sync-env` requests
an explicit environment refresh. Existing Hugging Face login or `HF_TOKEN` is reused;
otherwise the experiment prompts once for a write token without printing it.
No separate login/setup command is needed. Optional `--data`, `--output`,
`--model-id`, `--model-revision`, `--model-path` and `--repo` overrides remain
available. Use a fresh output directory; a failed run is preserved for inspection. Setup
is capped at 30 minutes and the experiment at 60 minutes; agree a spending cap
covering these limits before starting on a paid VM.

## What the command executes

1. Run a three-step baseline, recording the training source indices and saving its
   final checkpoint. Initial evaluation is disabled in all phases.
2. Start another Ray runtime from the same initialization model. Save at step 2;
   validate required files and verl's completion marker, write an interruption
   receipt, and raise the planned-stop exception. Ordinary failures are rejected.
3. Create a SHA-256/size manifest, then upload that complete checkpoint directly
   to a private Hub dataset repository. Default name:
   `ACCOUNT/part7-7b-recovery-execution-TIMESTAMP-SUFFIX`.
4. Download the recorded full Hub commit into a **fresh** `restored/global_step_2`
   directory and verify the payload hashes. The original checkpoint is not used
   as the resume source.
5. Start a fresh Ray runtime with native `resume_mode: resume_path`. Require the
   same model snapshot, data/prompt/reward bundle, Python/packages/source and
   single-node two-rank FSDP2 topology. Compare all local model/Adam tensor bytes,
   scheduler state and verl's saved worker RNG immediately after native loading.
   Compare the trainer's loaded global step/data-loader state before generation.
6. Complete step 3 and evaluation/checkpointing. Require actual two-rank sharded
   optimizer-update evidence and the same next source batch as the baseline.
   Record Ray nodes/resources, actor placement and placement groups.
7. Write `recovery_review.json`, timings and immutable Hub receipt; upload small
   experiment evidence under `evidence/` in the same private dataset repository.
   Keep checkpoint and evidence commit IDs separately in `hf_checkpoint.json`.

`run_7b.py` composes/launches one phase; `experiment.py` sequences the fresh
processes and transfers; `recovery_audit.py` wraps native methods without replacing
training or synchronization; `worker_extension.py` activates worker hooks.
Function docstrings explain inputs, outputs, behavior and failure conditions.

## Evidence, timing and limits

Evidence lives under `results/7b/recovery_execution_.../`: per-phase resolved
configuration, identities/status, diagnostics/recovery JSONL, rollouts, validation,
Ray logs, environment records, receipts and final review. The shell's full console
is also in `results/7b/execution_.../`. The uploaded console copy ends before the
final upload messages; final receipt/evidence commit and upload timing remain local.

Timing scopes include each phase, native checkpoint save, manifest hashing,
checkpoint upload, download plus integrity, per-rank native load versus audit
cost, and the delay from completed audit to first optimizer update. The latter
includes intervening synchronization/generation/log-probability work. Record cache
conditions from logs; these are not cold-start or production benchmarks.

This experiment verifies **same-host, fresh-Ray-runtime recovery**. It does not
verify replacement-node recovery. Worker Python/NumPy/Torch/current-device RNG is
checked as saved by verl. Driver-process RNG and vLLM server RNG/cache/in-flight
requests are not checkpointed by these hooks. Next data indices and exact restored
state do not promise identical sampled responses or final weights to the baseline.

All state comparisons execute on cloud GPUs; local tests use synthetic bytes,
primitive state and mocked hooks. Before releasing the instance, review the full
console, both rank audits, successful resumed update/checkpoint, Hub receipt and
`recovery_review.json`. A resume message or file listing alone is insufficient.

To copy only small evidence locally, from Part 7 on your computer:

```bash
rsync -az --partial --info=progress2 -e "ssh -p PORT" \
  --exclude=checkpoints --exclude=restored --exclude=.cache \
  CLOUD_HOST:/workspace/REPOSITORY/Foundation_model_performance_and_scaling/part7-ml-infra/results/7b/ \
  ./results/7b/
```

## CPU checks and helper commands

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_7b*.py' -v
bash -n infrastructure/7b_ray/part7B.sh
```

Set `VERL_SOURCE` to the pinned checkout and install Hydra/OmegaConf in a CPU-only
check environment to include real configuration composition; otherwise that test
is explicitly skipped. No model loading occurs in preview:

```bash
.venv-7a/bin/python -m infrastructure.7b_ray.run_7b --check-config \
  --checkpoint results/7b/EXPERIMENT/restored/global_step_2 --total-steps 3
```

Checkpoint helpers remain available for inspection/retrying transfers. Use
`python -m infrastructure.7b_ray.checkpoint_bundle --help`. A restore requires the
full immutable Hub commit and a new `global_step_N` destination. Inspect failed
partial directories rather than automatically reusing them. See [7B_PLAN.md](7B_PLAN.md)
for acceptance requirements.
