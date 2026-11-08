# 7E runbook — CPU preparation

The native SFT launcher/worker, metadata-resolved environment and model-free API
gate are prepared. Nineteen CPU tests passed. Native imports, model/tokenizer
loading, GPU training, checkpoint recovery and full E2E remain unverified.
No cloud instance is needed yet: image/RayJob packaging and recovery are pending.

## Current commands

From the Part 7 directory, **locally or on a cloud checkout**, without GPU/model
libraries:

```bash
bash infrastructure/7e_sft/part7E.sh check
bash infrastructure/7e_sft/part7E.sh prepare --output results/7e/native_prepared_sft
bash infrastructure/7e_sft/part7E.sh train --check --bundle results/7e/native_prepared_sft
```

`check` prints proposed settings and unresolved model revision. `prepare` discovers
the sibling Part 5 SFT file, verifies its recorded hash and writes JSONL train/eval
pairs, `manifest.json` and `resolved_config.json`. Output must be fresh. These files
contain existing training data and stay in ignored `results/`; do not commit them.
No downloads, tokenization, Ray startup or tensor/model loading occur in these
CPU commands. If the adapter source changed, prepare a fresh bundle rather than
editing its manifest or overwriting old evidence.

Overrides: `--source`, `--expected-sha256` (required to intentionally accept a
changed source), `--train-count`, `--eval-count`, `--model-repo`, `--model-revision`.
The four-step smoke requires at least 16 training pairs with default batch settings.
`train --check` verifies bundle hashes/order/adapter identity and previews placement.
`train --cloud` is an internal pod-only helper, not a VM setup/E2E command; it
requires prepared pod identity, an immutable image and shared PVC/Secret. Do not
run it yet. Image and E2E actions are not implemented.

## Workflow and code map

```text
part7E.sh check / prepare                     [available now]
  -> config.py: validate workload and bounded two-rank settings
  -> prepare.py: discover source and create a fresh bundle
       -> workloads/sft_workload.py: hash/validate/select exact prompt groups
       -> write train/evaluation JSONL + provenance/configuration

Future cloud RayJob                          [packaging not implemented yet]
  -> run_7e.py                               [source prepared]
       -> verify bundle and native versions/APIs; resolve immutable Hub commit
       -> stage/check complete model/tokenizer assets on shared PVC
       -> Ray Train V2 TorchTrainer: two GPU workers, STRICT_PACK
            -> worker.py: tokenize/record exclusions; verify token identity
            -> native FSDP2 update/evaluation, sharding/placement evidence
  -> baseline/interruption/pinned Hub restore/continuation  [pending]
  -> native distributed checkpoint and state audits        [pending]
  -> retain review evidence and verify owned cleanup       [pending]
```

`response_example` is a pure token-ID helper used by the cloud tokenizer adapter;
it does not instantiate a tokenizer. It masks prompt targets, supervises response
and EOS, and rejects overlength examples. Later code must record exclusions and
pad labels with -100. Dataset split counts here precede token-length validation.

For objectives, candidate environment, decisions and acceptance gates, see
[7E plan](7E_PLAN.md). Git clone/pull remains the preferred source delivery method
when approved; the future runner should prepare data on the VM from cloned Part 5
sources, without a required laptop data transfer. No cloud command is ready yet.

## Native worker responsibilities

| File | Purpose / execution location |
| --- | --- |
| `run_7e.py` | Inside the future RayJob driver: integrity/version gates, automatic model staging, Ray connection and TorchTrainer launch. CPU `--check` imports no native stack. |
| `worker.py` | Ray Train GPU processes: local-only tokenizer/model load, native FSDP2, response-token weighted loss, evaluation, actual sharding/update diagnostics. |
| `runtime_check.py` | Future cloud image build/import gate: native imports, version/API checks and tiny data round trip; no model instance or process group. |
| `environment/requirements.in` / `.lock` | Isolated candidate inputs and 109 pinned packages; metadata resolution does not establish native compatibility. |
| `workloads/sft_workload.py` | Reusable pair/token/label/padding/batch helpers; import without model/GPU libraries. |

Cloud staging and worker reads hash full model/tokenizer assets, and worker bundle
reads recheck hashes/source order. The model is downloaded on cloud automatically,
not manually copied from a laptop. Snapshot integrity reads bytes without tensor
deserialization. No cloud execution or recovery is claimed by these source checks.

For metadata reproduction only, from Part 7 (no installation):

```bash
uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
  --prerelease if-necessary-or-explicit \
  infrastructure/7e_sft/environment/requirements.in \
  --output-file /tmp/part7-7e-review.lock
```

Before cloud use, finish checkpoint/image/RayJob/E2E integration, verify native
imports in the isolated image and agree hardware/budget. No Part 5/6 environment
or accepted 7A–7D image needs an in-place change.
