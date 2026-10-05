# 7A step 1 — Environment and topology decision

Reviewed: 2026-10-04. Steps 1–3 source review and CPU preparation are complete;
cloud installation and execution remain pending. No resources were provisioned.
Runnable setup is in the [step 3 runbook](7A_RUNBOOK.md).

## Selected starting stack

Use an isolated Part 7 cloud environment, Linux x86_64 and Python 3.12.
[Core package pins](../environment/requirements.txt) select:

| Component | Selection | Basis |
| --- | --- | --- |
| verl | v0.7.1, commit `bec9ef74768dd201881cd4e54cd0385e87caae27` | Fixed tagged source with FSDP2, model-engine workers and hybrid vLLM rollout. |
| PyTorch / CUDA wheels | 2.8.0 / cu128 | vLLM 0.11.0 requires Torch 2.8.0; keep torchvision 0.23.0 and torchaudio 2.8.0 aligned. |
| vLLM | 0.11.0 | Inside verl's declared 0.8.5–0.12.0 range; explicit API branch exists in reviewed source. |
| Ray | 2.49.2, default and cgraph extras | Meets verl's >=2.41 and vLLM's >=2.48 requirements. Selected release pin, not an upstream certification of this combination. |
| Transformers | 4.55.4 | Matches verl's cu128 base recipe and meets vLLM's >=4.55.2 requirement. |
| FlashAttention | 2.7.4.post1 | Matches that base recipe; wheel/build must match Python 3.12, Torch 2.8 and its C++ ABI. |

Sources: [verl package metadata](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/setup.py),
[cu128 base recipe](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/docker/verl0.6-cu128-torch2.8.0-fa2.7.4/Dockerfile.base),
[vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt),
[Ray release](https://github.com/ray-project/ray/releases/tag/ray-2.49.2).

Why this bounded release: it preserves a documented CUDA 12.8/Torch 2.8 path
without following moving latest dependencies. The tagged stable Dockerfile/CI
instead use vLLM 0.17/Torch 2.10 while setup.py caps vLLM at 0.12. Do not combine
that recipe with these pins or copy a `*.latest` image. This discrepancy makes
cloud validation necessary; no claim of an already tested environment is made.
[Tagged Dockerfile](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/docker/Dockerfile.stable.vllm).

Step 3 resolved the [dependency lock](../environment/requirements.lock) for
Python 3.12/Linux x86_64 and selected Python 3.12.14 for cloud setup. CUDA wheels
use explicit official URLs because resolver CUDA-index routing did not expose
the pinned torchdata version. Resolution inputs and FlashAttention metadata
override are checked in alongside the lock. Hydra requires ANTLR 4.9.3, so
math-verify uses its supported `antlr4-9-3` extra; reward tests passed with it.
Pin datasets 3.6.0/pandas 2.2.3/accelerate 1.10.1 to avoid old datasets/PyArrow APIs
and constrain the data layer. The version lock has not been installed/tested on
GPU; it is not an artifact-hash lock. Provider image identity/digest selection is
pending before rental, and a digest-pinned deployment image remains future work.
Verify package imports on cloud before loading the model. Build FlashAttention
inside the cloud environment; never reuse Part 5's legacy wheel.
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

Propose vLLM TP=1, DP=1, PP=1 per replica: two one-GPU rollout replicas share
the two training GPUs through hybrid lifecycle/weight transfer. Generation is
served asynchronously, but the initial algorithm remains synchronous on-policy
GRPO; asynchronous serving does not imply fully asynchronous training. No third
dedicated inference GPU is required by this design. Verify actual replica count,
process/device placement and memory transitions on cloud. Framework sleep/cache
release and weight synchronization manage phase changes; Ray resource fractions
do not reserve or isolate GPU memory. Sources: [replica lifecycle](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/rollout/replica.py),
[engine worker](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/workers/engine_workers.py).

First-smoke settings to finalize in steps 2–3: BF16 compute, FP32 training
parameters where supported by the selected engine, small fixed microbatches,
gradient checkpointing, eager rollout, conservative vLLM memory fraction (0.35),
no KL/reference model, no critic or learned reward model. Use the CPU math grader.
Start with a 256-token response cap and measured prompt-length limit; record
truncation. Disable compilation/dynamic batching initially to simplify numerical
checks. These are proposals, not validated Hydra launch arguments or memory guarantees.

Evidence required: ranks 0 and 1 in one training world, local GPU identities,
FSDP modules/DTensor placements, logical versus local shapes, and parameter,
gradient and optimizer-state sharding after an update. Record transient all-gathers
and any replicated state. A two-replica vLLM rollout is not evidence of two-rank
training. Single-node success makes no multi-node scalability claim.

## Checkpoint and export path

Retain model, optimizer and `extra` checkpoint contents on both ranks. The
FSDP manager saves scheduler and RNG in `extra`; the trainer separately saves
`data.pt`, a global-step directory and the latest-step marker. Keep the complete
run checkpoint tree. Same-topology resume is the first recovery target; changed
world-size recovery and exact rollout-server RNG restoration are unverified.
Sources: [checkpoint manager](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/utils/checkpoint/fsdp_checkpoint_manager.py),
[trainer save/resume](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/trainer/ppo/ray_trainer.py).

The reviewed trainer rejects resume through `default_hdfs_dir`; use a local or
mounted filesystem path and copy the complete tree to durable storage, restoring
it before resume. Do not assume native S3/HDFS recovery. 7B will test restoration
and elapsed recovery time. For 7D, use the pinned `verl.model_merger` FSDP export
path and verify the resulting Hugging Face model/tokenizer on cloud; export and
resumption have not been executed. [FSDP merger source](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/model_merger/fsdp_model_merger.py).

## Hardware and proposed budget

Preferred first smoke: **two A100 40 GB GPUs on one cloud node**, with working
peer communication, at least 16 allocated CPU cores, 128 GB host RAM, 100 GB
free disk and 16 GB shared memory. Two 48 GB Ampere/Ada GPUs are an alternative;
80 GB GPUs provide more memory margin if the approved offer fits the cap.
These are conservative sizing proposals, not measured requirements. Avoid a
24 GB allocation for the first integration run to reduce memory-debugging risk.

A rough 1.5B FP32 parameter/gradient/Adam-state estimate is 24 GB total, or
12 GB per training rank with ideal two-way sharding, before activations,
all-gathers, logits, rollout weights/cache and allocator overhead. Peak memory
must be measured across initialization and phase changes. Keep model identity,
lengths, batch sizes and parallelism configurable for a later larger-model run.

Require a host driver suitable for CUDA 12.8 including JIT/PTX; prefer R570 or
newer and check the selected wheel/image requirements on the actual host.
CUDA 12.x's generic minimum alone does not establish support for all features.
[NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[vLLM 0.11 installation guidance](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html).

**Proposed cap, awaiting user agreement:** one node, maximum two billable hours,
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
- Pending: image identity/digest, cloud installation/imports, model revision
  fingerprints, tokenizer checks and every GPU check.
- Step 2 is now implemented; see [adapters and GRPO mapping](7A_GRPO_MAPPING.md)
  for CPU validation and outstanding numerical checks. No reward or training
  code was implemented during step 1 itself.
