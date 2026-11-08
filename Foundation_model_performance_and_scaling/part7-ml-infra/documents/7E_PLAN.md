# 7E — Ray Train/FSDP2 SFT platform reuse

Updated: 2026-11-08. User selected optional 7E after bounded core 7A–7D completion.

## Current scope and status

Demonstrate a small SFT workload on the same Kubernetes/Ray/PVC/Hub conventions.
Start with Qwen2.5-Math-1.5B, two actual GPU training ranks on one node, response-only
cross-entropy and native PyTorch FSDP2/distributed checkpointing. Ray Train launches
and supervises the ranks; PyTorch/NCCL handles collectives. verl is not involved.
No Part 6 8B retraining, quality target, multi-node or shared-pool admission is added.

Implemented: configuration/data preparation, native Ray Train V2 launcher/worker,
automatic pinned model staging, FSDP2 loss/evaluation/sharding diagnostics, an
isolated dependency lock and model-free native import/API gate. These are source
and CPU preparations. No cloud native import, model/tokenizer loading, training,
image build or checkpoint recovery has been executed/verified for 7E.

Validation: 19 targeted CPU tests passed, covering data integrity/split isolation,
label/EOS/padding boundaries, token exclusions, accumulated global-token loss,
equal evaluation collective counts, snapshot integrity and local-execution guards.
Actual data preparation verified 3,496 unique prompt groups and disjoint 128/16
splits. A fresh bundle/launcher preview passed; old bundles deliberately fail if
the adapter hash changes. Metadata-only resolution pinned 109 packages. Pinned
Ray Train V2 constructor keywords were audited without importing native libraries.
Do not provision a VM until image/checkpoint/E2E integration is ready.

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
  after update 2. The worker uses a fixed sample order and disables dropout; cursor/state
  restoration auditing remains the next increment. No claim of quality improvement.
- Base model repo is configurable; immutable revision must be resolved before cloud
  execution. Preview explicitly leaves it unresolved rather than treating main as
  an accepted identity. Pin tokenizer identity alongside model/data/source hashes.

## Environment and framework decision

The isolated [requirements input](../infrastructure/7e_sft/environment/requirements.in)
and [109-package lock](../infrastructure/7e_sft/environment/requirements.lock) target
Python 3.12/Linux x86_64. Core pins are Ray 2.59.0, Torch 2.13.0 (CUDA 13 build
required by the native gate), Transformers 5.20.2 and Hub 2.3.0; NumPy/pandas/Arrow
are 2.3.5/3.1.0/26.0.0. This is metadata-resolved, not installed or cloud-validated.
Ray includes default/dashboard and Train extras for KubeRay jobs. No vLLM, verl or
FlashAttention dependency is added. Native version/API/data checks precede future
image publication. Keep 7A–7D environments/images unchanged.

Two beta transitive releases, OpenTelemetry exporter-prometheus and semantic-
conventions 0.66b2, are recorded exceptions selected by Ray's dependencies. The
resolution uses `if-necessary-or-explicit` with explicit stable framework pins.
CUDA transitive components include versions newer than 13.0; metadata resolution
and a driver baseline alone do not establish native/GPU compatibility.

Ray 2.59 defaults to Train V2. Use its public TorchTrainer/ScalingConfig API,
STRICT_PACK and two one-GPU workers, with no deprecated `trainer_resources` field.
Worker, controller and preemption auto-retries are disabled so missing recovery
cannot be hidden. The future image must retain V2 (`RAY_TRAIN_V2_ENABLED=1`).
The worker uses Ray's devices/process groups, then applies public `fully_shard`
to decoder blocks/root before creating AdamW. It does not apply DDP or legacy
FSDP1. FP32 parameter/optimizer storage uses BF16 compute and FP32 reduction.
The block adapter currently supports the Qwen-style `model.layers` layout; another
architecture requires review rather than an assumption that model ID alone suffices.

The native worker computes a global response-token mean across ranks and gradient
accumulation; it compensates for FSDP averaging and does not additionally divide
by accumulation steps. Uneven held-out counts use zero-weight duplicate forwards
to align collectives. Record local/global shapes of actual DTensor parameters,
gradients and both Adam moments after an update, finite nonzero gradients and a
changed parameter probe. These diagnostics are implemented but not GPU-verified.
Native distributed checkpoint save/load and restoration audits remain pending;
the current worker runs the four-step smoke without a checkpoint or resume path.

Reviewed primary interfaces:
[TorchTrainer V2 2.59](https://github.com/ray-project/ray/blob/ray-2.59.0/python/ray/train/v2/torch/torch_trainer.py),
[Train V2 config](https://github.com/ray-project/ray/blob/ray-2.59.0/python/ray/train/v2/api/config.py),
[PyTorch state APIs 2.13](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/checkpoint/state_dict.py),
[FSDP2 guide](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
The cloud native gate must pass before build/publication acceptance.

## Implementation sequence and acceptance

1. **Workload/configuration (CPU-checked):** CPU preparation, provenance, disjoint
   splits and label-boundary tests. See [runbook](7E_RUNBOOK.md).
2. **Native trainer/image (source/lock prepared; packaging pending):** thin Ray
   Train worker/launcher, FSDP2 and loss diagnostics plus native API gate are prepared.
   Docker recipe, image build/publish control and RayJob rendering remain pending. Reuse existing platform
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
