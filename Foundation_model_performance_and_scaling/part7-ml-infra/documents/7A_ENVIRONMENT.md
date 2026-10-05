# 7A step 1 — Environment and topology decision

Reviewed: 2026-10-05. Bounded 7A cloud execution, two-rank FSDP2 sharding,
checkpoint export, private Hub uploads and a pinned model-weight reload are
verified on two A100-SXM4-40GB GPUs. The instance was released after local log
retention. Exact training-state recovery remains 7B work. Runnable setup is in
the [cloud runbook](7A_RUNBOOK.md).

## Selected starting stack

Use an isolated Part 7 cloud environment, Linux x86_64 and Python 3.12.
[Core package pins](../infrastructure/7a_verl/environment/requirements.txt) select:

| Component | Selection | Basis |
| --- | --- | --- |
| verl | v0.7.1, commit `bec9ef74768dd201881cd4e54cd0385e87caae27` | Fixed tagged source with FSDP2, model-engine workers and hybrid vLLM rollout. |
| PyTorch / CUDA wheels | 2.9.0 / cu130 | vLLM 0.12.0 requires Torch 2.9.0; torchvision 0.24.0 and torchaudio 2.9.0 align. |
| vLLM | 0.12.0+cu130 official wheel | Inside the pinned verl release's declared range through 0.12.0. |
| Ray | 2.49.2, default and cgraph extras | Retained; meets framework minimums. |
| Transformers | 4.57.6 | Meets vLLM 0.12's >=4.56 and <5 requirement. |
| FlashAttention | 2.8.3, cu13/Torch 2.9/cp312/C++11 ABI wheel | Official matching binary; no compilation or separate toolkit installation. |

Sources: [verl metadata](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/setup.py),
[vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.12.0/requirements/cuda.txt),
[vLLM release wheels](https://github.com/vllm-project/vllm/releases/tag/v0.12.0),
[FlashAttention release wheels](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3),
[PyTorch releases](https://pytorch.org/get-started/previous-versions/).

The original CUDA 12.8/Torch 2.8/vLLM 0.11 selection and the subsequent toolkit
installation workaround are superseded by the user's preference to keep the
CUDA 13 image and avoid a multi-GB toolkit download. The verl commit, workload
adapters, GRPO mapping and topology remain unchanged. This is a compatibility
selection subsequently verified in the bounded cloud smoke; it is not an
upstream certification or evidence of compatibility with every CUDA 13 image.

[Resolution inputs](../infrastructure/7a_verl/environment/requirements-cu130.in)
and [dependency lock](../infrastructure/7a_verl/environment/requirements.lock)
select Python 3.12/Linux x86_64 (glibc >=2.31); cloud Python is 3.12.14. Official
CUDA wheels use explicit URLs. vLLM and FlashAttention assets include SHA-256
fragments from publisher metadata; the entire dependency lock is not hash-locked.
Hydra and math-verify share ANTLR 4.9.3. Retain datasets 3.6.0, pandas 2.2.3,
accelerate 1.10.1 and the existing CPU grading pins. Full resolution/import and
CUDA ABI checks are distinct: installation/imports and execution passed on the
reviewed cloud instance; other GPU/image combinations are not implied.
Image digest recording and a digest-pinned deployment image remain pending.
Do not install Megatron, DeepEP, Apex or TransformerEngine for this FSDP workload.

## Training and rollout topology

Start with one node and two GPUs in a shared Ray resource pool. Choose the new
model engine explicitly (`trainer.use_legacy_worker_impl=disable`) and FSDP2
for the actor, including its resolved engine strategy. Use an FSDP group size
of two, sequence parallelism one and reshard-after-forward enabled.
The defaults still say FSDP; inspect resolved configuration rather than trusting
the workload name. Source: [worker selection](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/trainer/main_ppo.py),
[engine implementation](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/engine/fsdp/transformer_impl.py),
[mesh construction](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/engine/fsdp/utils.py).

The smoke selects vLLM TP=1, DP=1, PP=1 per replica: two one-GPU rollout replicas share
the two training GPUs through hybrid lifecycle/weight transfer. Generation is
served asynchronously, but the initial algorithm remains synchronous on-policy
GRPO; asynchronous serving does not imply fully asynchronous training. No third
dedicated inference GPU is required by this design. Verify actual replica count,
process/device placement and memory transitions on cloud. Framework sleep/cache
release and weight synchronization manage phase changes; Ray resource fractions
do not reserve or isolate GPU memory. Sources: [replica lifecycle](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/rollout/replica.py),
[engine worker](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/engine_workers.py).

Implemented first-smoke settings: BF16 compute, FP32 training
parameters where supported by the selected engine, small fixed microbatches,
gradient checkpointing, eager rollout, conservative vLLM memory fraction (0.35),
no KL/reference model, no critic or learned reward model. Use the CPU math grader.
Start with a 256-token response cap and measured prompt-length limit; record
truncation. Disable compilation/dynamic batching initially to simplify numerical
checks. These settings composed and executed successfully for the reviewed two-A100
smoke; they are not memory guarantees for other workloads or hardware.

Retained diagnostics verify ranks 0 and 1 in one training world and distinct
visible GPU identities, 338 sharded parameter tensors per rank, six optimizer
calls per rank and sampled gradient/Adam sharding. Evidence to retain includes
FSDP modules/DTensor placements, logical versus local shapes, and parameter,
gradient and optimizer-state sharding after an update. Record transient all-gathers
and any replicated state. A two-replica vLLM rollout is not evidence of two-rank
training. Single-node success makes no multi-node scalability claim.

## Checkpoint and export path

Retain model, optimizer and `extra` checkpoint contents on both ranks. The
FSDP manager saves scheduler and RNG in `extra`; the trainer separately saves
`data.pt`, a global-step directory and the latest-step marker. Keep the complete
run checkpoint tree when full-state retention is requested. The 7A runner saves
these states on cloud, but its default Hub evidence upload excludes them.
Exported model weights and local logs were retained for the reviewed runs.
Same-topology resume is the first 7B recovery target; changed
world-size recovery and exact rollout-server RNG restoration are unverified.
Sources: [checkpoint manager](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/utils/checkpoint/fsdp_checkpoint_manager.py),
[trainer save/resume](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/trainer/ppo/ray_trainer.py).

The reviewed trainer rejects resume through `default_hdfs_dir`; use a local or
mounted filesystem path and copy the complete tree to durable storage, restoring
it before resume. Do not assume native S3/HDFS recovery. 7B will test restoration
and elapsed recovery time. The pinned `verl.model_merger` FSDP export has now
produced a BF16 model/tokenizer that was uploaded and loaded in a new cloud
smoke at the recorded Hub commit. This verified weights-only path can support
7D; exact training resumption has not been executed. [FSDP merger source](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/model_merger/fsdp_model_merger.py).

## Hardware and proposed budget

Verified first-smoke GPU allocation: **two A100 40 GB GPUs on one cloud node**.
The original host sizing proposal was at least 16 allocated CPU cores, 128 GB
host RAM, 100 GB free disk and 16 GB shared memory, with working peer
communication. Those host-capacity figures are proposals, not audited measurements. Two 48 GB Ampere/Ada GPUs are an alternative;
80 GB GPUs provide more memory margin if the approved offer fits the cap.
These are conservative sizing proposals, not measured requirements. Avoid a
24 GB allocation for the first integration run to reduce memory-debugging risk.

A rough 1.5B FP32 parameter/gradient/Adam-state estimate is 24 GB total, or
12 GB per training rank with ideal two-way sharding, before activations,
all-gathers, logits, rollout weights/cache and allocator overhead. Peak memory
must be measured across initialization and phase changes. Keep model identity,
lengths, batch sizes and parallelism configurable for a later larger-model run.

Require a host driver compatible with CUDA 13 (R580 or newer), and verify CUDA
initialization on the actual GPUs. A CUDA toolkit version alone does not prove
that the host driver supports the wheel runtime.
[NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

**Original proposed cap for future offer review (actual spend not recorded):** one node, maximum two billable hours,
maximum $5/hour for the entire two-GPU instance, and maximum $15 total including
storage/transfer. The compute allowance is $10; reserve $5 for ancillary costs.
This is an offer-selection ceiling, not a quoted market price. Obtain a live
total-price quote before renting; if no suitable offer fits, revise the proposal
before provisioning. Vast.ai pricing varies by offer; [official pricing](https://vast.ai/pricing).
Bound the first job to three rollout iterations, a 32-example fixed evaluation
subset and a final checkpoint; stop debugging when the time/cost cap is reached.
Retain artifacts and release rented resources. No rental or spend is authorized
by this note.

## Verification status and next step

- Completed: exact verl tag-to-commit lookup, downloaded-source review, vLLM
  package-metadata checks, topology/checkpoint inspection and core version pins.
- Completed subsequently: dependency resolution, upstream Hydra composition,
  data fingerprints, CPU adapters/loss checks and cloud runner preparation.
- Verified on cloud: installation/imports, model-file identities, generation,
  nonzero policy gradients in the upload run, two-rank parameter/gradient/Adam
  sharding, evaluation, checkpoint save, BF16 export, private Hub retention and
  a fresh smoke selecting the uploaded model commit.
- Pending: fixed image digest/deployment reproducibility, broader tokenizer and
  numerical equivalence, per-phase memory/placement analysis, full training-state
  restore and multi-node recovery/scaling.
- See [adapters and GRPO mapping](7A_GRPO_MAPPING.md) for the implementation and
  evidence limits. No reward or training code was implemented during step 1 itself.
