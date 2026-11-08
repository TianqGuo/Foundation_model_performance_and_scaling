# 7E runbook — CPU preparation

7E is in its first implementation increment. Model/tokenizer/training/recovery and
GPU execution remain unverified. No cloud instance or environment setup is needed
yet; future cloud execution will use this milestone's shell runner.

## Current commands

From the Part 7 directory, **locally or on a cloud checkout**, without GPU/model
libraries:

```bash
bash infrastructure/7e_sft/part7E.sh check
bash infrastructure/7e_sft/part7E.sh prepare --output results/7e/prepared_sft
```

`check` prints proposed settings and unresolved model revision. `prepare` discovers
the sibling Part 5 SFT file, verifies its recorded hash and writes JSONL train/eval
pairs, `manifest.json` and `resolved_config.json`. Output must be fresh. These files
contain existing training data and stay in ignored `results/`; do not commit them.
No downloads, tokenization, Ray startup or tensor/model loading occur.

Overrides: `--source`, `--expected-sha256` (required to intentionally accept a
changed source), `--train-count`, `--eval-count`, `--model-repo`, `--model-revision`.
The four-step smoke requires at least 16 training pairs with default batch settings.
These commands are preparation only; no `train`, `image` or `e2e` action is exposed.

## Workflow and code map

```text
part7E.sh check / prepare                     [available now]
  -> config.py: validate workload and bounded two-rank settings
  -> prepare.py: discover source and create a fresh bundle
       -> workloads/sft_workload.py: hash/validate/select exact prompt groups
       -> write train/evaluation JSONL + provenance/configuration

Future cloud RayJob                          [not implemented yet]
  -> Ray Train TorchTrainer: start two GPU worker processes
       -> tokenize/validate response-only examples
       -> native FSDP2 forward/backward/update and sharding diagnostics
       -> native distributed save/load + explicit state/data restoration audits
  -> baseline/interruption/pinned Hub restore/continuation
  -> retain review evidence and verify owned cleanup
```

`response_example` is a pure token-ID helper for the future cloud tokenizer adapter;
it does not instantiate a tokenizer. It masks prompt targets, supervises response
and EOS, and rejects overlength examples. Later code must record exclusions and
pad labels with -100. Dataset split counts here precede token-length validation.

For objectives, candidate environment, decisions and acceptance gates, see
[7E plan](7E_PLAN.md). Git clone/pull remains the preferred source delivery method
when approved; the future runner should prepare data on the VM from cloned Part 5
sources, without a required laptop data transfer. No cloud command is ready yet.
