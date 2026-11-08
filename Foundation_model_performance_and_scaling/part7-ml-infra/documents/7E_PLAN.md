# 7E — Ray Train/FSDP2 SFT platform reuse

Updated: 2026-11-08. User selected optional 7E after bounded core 7A–7D completion.

## Current scope and first increment

Demonstrate a small SFT workload on the same Kubernetes/Ray/PVC/Hub conventions.
Start with Qwen2.5-Math-1.5B, two actual GPU training ranks on one node, response-only
cross-entropy and native PyTorch FSDP2/distributed checkpointing. Ray Train launches
and supervises the ranks; PyTorch/NCCL handles collectives. verl is not involved.
No Part 6 8B retraining, quality target, multi-node or shared-pool admission is added.

Implemented now: dependency-free configuration, deterministic pair selection,
provenance, response-label boundary helpers and a CPU shell entry point. No training
loop, cloud setup, image build, model/tokenizer loading or checkpoint recovery is
implemented/verified yet. Do not provision a VM for this increment.

Validation: seven targeted CPU tests passed. Actual source preparation verified
3,496 unique prompt groups, 128 training/16 held-out pairs, matching bundle hashes
and disjoint exact prompts. Evidence: `results/7e/step1_prepared_sft`. Shell syntax,
Python syntax/docstrings, documentation links and whitespace checks passed.

## Workload and semantic decisions

- Input: existing Part 5 `data/math/sft_filtered_repo.jsonl`, 3,496 rows. Pin SHA256
  `55bf19e43fba511b5e90fc256c5e415f68049de25caf1204cae25475a93f241f`.
  Reuse exact prompt/response text. This is an existing filtered dataset, not a
  newly graded corpus; its filename does not establish teacher correctness.
- Choose the first response per exact prompt, order prompt groups by a seeded
  SHA256 key, then select 128 training and 16 held-out pairs. Store source indices,
  file/adapter hashes and resolved settings. No prompt group appears in both splits.
  This held-out set measures SFT loss, not the 32-prompt RL/serving evaluation or
  MATH answer accuracy. Exact grouping does not establish semantic deduplication.
- On cloud, encode prompt and response separately without implicit special tokens
  or chat-template changes; concatenate them and append one terminal EOS if absent.
  Mask prompt labels with -100; supervise response/EOS. Hugging Face causal LM
  shifts labels internally. Cloud padding must mask labels/masks appropriately.
- Reject overlength examples before training, record exclusions and resulting
  bundle/token identities. Tokenization of these sources has not been performed;
  sequence-fit counts and nonempty final train/eval sets must be verified on cloud.
- Proposed smoke: sequence cap 2048, per-rank microbatch 1, two accumulated
  microbatches, global batch 4, learning rate 1e-5, total four updates and checkpoint
  after update 2. Use a fixed recorded sample order/cursor and dropout-disabled
  comparison for the bounded restoration audit. No claim of quality improvement.
- Base model repo is configurable; immutable revision must be resolved before cloud
  execution. Preview explicitly leaves it unresolved rather than treating main as
  an accepted identity. Pin tokenizer identity alongside model/data/source hashes.

## Environment and framework decision

Proposed isolated stack: Python 3.12, Ray Train 2.59.0, Torch 2.13.0/CUDA 13 and
Transformers 5.20.2, based on the accepted serving core rather than an automatic
latest-version upgrade. This is a candidate, not a resolved/installed 7E environment.
Use a separate `infrastructure/7e_sft/environment/` and image when implemented;
do not upgrade training/serving milestones in place. No vLLM/verl/FlashAttention
requirement is introduced for SFT; prefer SDPA and the minimal necessary libraries.

Native `TorchTrainer` establishes distributed workers. The worker adapter will
apply public `fully_shard` to transformer blocks/root and create the optimizer
after sharding; avoid accidental DDP wrapping or the legacy FSDP1 path. Use native
sharded distributed checkpoint state APIs, not a custom tensor serialization format.
A small worker loop is necessary for workload integration; do not copy Part 5's
SFT trainer or build another orchestration framework.

Reviewed primary interfaces:
[TorchTrainer 2.59](https://github.com/ray-project/ray/blob/ray-2.59.0/python/ray/train/torch/torch_trainer.py),
[PyTorch state APIs 2.13](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/checkpoint/state_dict.py),
[FSDP2 guide](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
Resolve/pin the remaining environment and audit the exact chosen APIs before builds.

## Implementation sequence and acceptance

1. **Workload/configuration (current):** CPU preparation, provenance, disjoint
   splits and label-boundary tests. See [runbook](7E_RUNBOOK.md).
2. **Native trainer/image:** thin Ray Train worker/launcher, FSDP2, loss accounting,
   isolated lock/image/import gate and RayJob rendering. Reuse existing platform
   helpers with their version/ownership assumptions checked; no fake ready commands.
3. **Checkpoint/recovery:** native model/Adam state plus scheduler, per-rank RNG,
   sample cursor and configuration/data/tokenizer/image identities. Complete files
   before publishing a checkpoint manifest. Audit hashes and same two-rank topology;
   retain full state in private Hub storage and restore from a pinned commit into
   a fresh path/runtime. A model-weight reload is not training-state recovery.
4. **One-command cloud E2E:** setup/check platform; prepare/stage data/model;
   uninterrupted baseline; planned update-2 interruption; pinned Hub restoration;
   continue through update 4; review and cleanup. This is controlled checkpoint-
   boundary recovery, not arbitrary crash/VM/node replacement or bitwise proof.
5. **Review:** require actual DTensor parameter/gradient/Adam sharding and local/global
   shapes on both ranks, nonzero finite updates/loss, fixed held-out loss, exact
   restored state fingerprints and next-sample identities before continuation.
   Record two actual rank/GPU identities, per-phase timing, source/stack identities,
   logs and limits. Retain lightweight evidence before releasing resources.

Hardware and spending cap remain unselected. Cloud execution stays pending until
CPU/native gates and the full runner are ready. Optional Kueue, larger models,
training TP/PP and broader recovery remain separate decisions in [PLAN.md](../PLAN.md).
