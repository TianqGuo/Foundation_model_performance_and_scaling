# 7B — Checkpoint preparation and recovery runbook

Checkpoint manifests, integrity validation, private Hub transfer helpers and native
resume configuration preview are implemented. Runtime audits, controlled interruption,
Ray observability and the single cloud experiment runner are pending. Do not rent
an instance or treat the helpers as verified recovery yet.

## Environment and checkpoint preparation

Reuse 7A's `.venv-7a`, pinned verl source and MATH bundle. No new GPU libraries are
introduced. All tensor loading remains on cloud GPUs. Local tests use synthetic
byte files and configuration only; helpers never call `torch.load`.

The integrity contract requires both ranks' model/optimizer/extra-state files,
`data.pt`, FSDP2 metadata, model config and tokenizer. Creation requires the producer's
latest-step completion marker and identities. The file-size/SHA-256 manifest is
portable and immutable. Validation rejects missing/empty files, extra rank shards,
changed contents, unsafe paths, symlinks and incompatible topology.

Helper-level commands from Part 7 (the one-command experiment runner is not ready):

```bash
.venv-7a/bin/python -m infrastructure.7b_ray.checkpoint_bundle create \
  --checkpoint results/7b/PRODUCER/checkpoints/global_step_2 \
  --identity results/7b/PRODUCER/identities.json
.venv-7a/bin/python -m infrastructure.7b_ray.checkpoint_bundle verify \
  --checkpoint results/7b/PRODUCER/checkpoints/global_step_2
```

Verify an existing manifest rather than overwriting it. A failed create may leave
its `.tmp` file; inspect the failed producer before retrying. File integrity does
not itself prove optimizer/scheduler/data/RNG restoration.

## Direct cloud retention and restore

Transfers reuse saved Hub authentication or `HF_TOKEN`; the future milestone shell
runner will handle interactive login and naming. Supply a private dataset repository
dedicated to this checkpoint. These commands transfer full state, not BF16 exports:

```bash
.venv-7a/bin/python -m infrastructure.7b_ray.checkpoint_bundle upload --cloud \
  --checkpoint results/7b/PRODUCER/checkpoints/global_step_2 \
  --repo ACCOUNT/part7-7b-checkpoint-001
.venv-7a/bin/python -m infrastructure.7b_ray.checkpoint_bundle restore --cloud \
  --checkpoint results/7b/restored/global_step_2 \
  --repo ACCOUNT/part7-7b-checkpoint-001 --revision FULL_HUB_COMMIT
```

Upload prints a commit receipt; preserve it with logs. Restore requires a full Hub
commit hash and fresh `global_step_N` directory, downloads checkpoint/manifest
payloads and verifies hashes. No full state passes through the laptop. Transfers
remain large and consume private storage. Actual cloud transfers are not verified.

## Resume configuration preview

```bash
.venv-7a/bin/python -m infrastructure.7b_ray.run_7b --check-config \
  --checkpoint results/7b/restored/global_step_2 --total-steps 4
```

Preview reuses 7A configuration, verifies data/reward/prompt bundle hashes,
requires matching two-rank FSDP2 settings and selects native `resume_path`.
Total steps must exceed the saved step; initial evaluation is disabled to avoid
consuming RNG before future state audits. Full composition requires Hydra/OmegaConf
and the pinned upstream YAML. The preview does not load models or launch training.
Model identity at runtime and state restoration still need explicit audits.

## CPU checks and next increment

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_7b_checkpoint.py' -v
```

Next add cloud-only save/load audits, placement/metrics capture and controlled
checkpoint-boundary interruption, then combine the workflow in a milestone shell
entry point. See [7B_PLAN.md](7B_PLAN.md) for acceptance/timing boundaries. 7B is
not complete until the cloud recovery experiment and retained-evidence review pass.
