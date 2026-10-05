# Part 7 — ML Infrastructure

Updated: 2026-10-04

Status: 7A steps 1–3 complete within source review, adapters, dependency resolution
and CPU preparation. User-provided cloud logs show the first setup attempt stopped
at the image's CUDA 13.0 toolkit check, before Python installation/training.
Setup now prepares/selects a separate CUDA 12.8 toolkit without rebuilding the
image or installing a driver; the revised cloud installation remains unverified.
The cloud launcher now automatically downloads a missing model snapshot and
reuses existing snapshots; local configuration checks remain download-free.
`infrastructure/7a_verl/part7A.sh` is the single cloud entry point: setup/reuse, optional data preparation,
configuration checks, model preparation, smoke execution and retained console logs.
This project uses `part7-ml-infra/`, the existing workspace name;
the Part 5 plan calls it `part7-ml-infrastructure/`.

## Scope and reference

Carry forward [Stage 4](../part5-alignment/PART5_MODERNIZATION_PLAN.md#stage-4-part-7--ml-infrastructure-planned-implementation-unstarted)
and [working conventions](../part5-alignment/AGENTS.md). Preserve Parts 1–6.
Part 5 modernization is complete; further GSPO tuning is deferred.

Supporting documents live in `documents/`. The local
[Part 5 plan copy](documents/PART5_MODERNIZATION_PLAN.md) is historical reference;
its paths describe the original Part 5 context. Keep `PLAN.md` and `AGENTS.md`
at the project root for ongoing planning and working conventions.

Group infrastructure code by milestone. `infrastructure/7a_verl/` contains the
7A shell/Python runners, diagnostics, configuration and environment files.
`workloads/` holds reusable MATH and generation adapters. Add 7B–7E directories
when their implementation starts; tests and results remain at the project root.

Start with Qwen2.5-Math-1.5B and MATH. The corrected Part 5 GRPO-Clip reference is
`tight_pilot_FRqpow9W`: clipping 0.2/0.2, FP32 log-softmax, stable microbatches,
54/128 fixed-subset accuracy. This single-seed, capped-generation result is
context, not a target or a cross-framework performance claim. Validate framework
semantics explicitly; matching accuracy or bitwise GPU outputs is not required.

Model switching is a design requirement. Keep model identity/path, tokenizer and
prompt handling, sequence limits, batch sizes and training/rollout/serving resource
settings configurable; keep dataset/reward adapters separate from infrastructure.
After validating the initial workload, a larger model is a desired practice step,
with the model and budget chosen later. Each switch requires framework/tokenizer,
checkpoint export and GPU-memory validation; configuration alone does not establish
compatibility. This goal does not expand the initial 7A acceptance criteria.

Shared platform: Kubernetes → KubeRay → Ray. RL uses verl/FSDP2/vLLM via
RayJob; inference uses Ray Serve LLM/vLLM via RayService. Optional SFT uses
Ray Train/FSDP2 via RayJob. Ray Train does not wrap verl.

## Accepted roadmap and completion evidence

| Milestone | Required outcome |
| --- | --- |
| 7A — Framework workload | Pinned isolated verl/Ray/PyTorch/vLLM stack; thin reward/prompt/data adapters; explicit GRPO mapping; one cloud generation/update/evaluation/checkpoint run proving at least two FSDP2 training ranks and actual sharding. Retain config, identities, placement, throughput, memory, logs and checkpoint events. |
| 7B — Ray operations | Recorded topology/resource placement; durable artifacts readable after job exit; controlled interruption and measured recovery. Verify restoration of model, optimizer, scheduler where applicable, step, data/progress and RNG; disclose unsupported state. Use built-in metrics/logs first, small Prometheus/Grafana where useful; W&B optional, no smoke login required. |
| 7C — Kubernetes training | Digest-pinned image and real GPU KubeRay RayJob with explicit resources, sufficient shared memory, durable storage and Secret injection. Retain manifests, placement, status, metrics/logs and checkpoints; validate interruption/recovery and cleanup. CPU manifest checks alone do not complete this milestone. |
| 7D — Serving | Export/merge a 7A checkpoint for vLLM; deploy through RayService with readiness and durable model access. Measure requests/s, generated tokens/s, TTFT, output-token timing, p50/p95 end-to-end latency, errors and replica status; demonstrate restart/readiness recovery and cleanup. |
| 7E — Optional SFT | Small Ray Train/PyTorch or Hugging Face FSDP2 workload reusing platform conventions; verify two-rank sharding and recovery. No Part 6 8B retraining required. |

Record serving hardware, lengths, concurrency, warmup, streaming and measurement
scope; distinguish engine metrics from client latency and memory/cache use from
compute utilization. Training and serving can run sequentially and use separate
pinned images. Fractional Ray GPU requests do not isolate GPU memory.

Check Kubernetes access/cost before 7C: a rented GPU container is not automatically
a GPU node; verify host privileges, GPU plugin access, networking, shared memory
and storage. If blocked by access/cost, develop and measure serving on a cloud GPU
first, then complete RayService deployment later; this fallback does not finish 7D.

## Proposed first 7A implementation steps

Step 1 was authorized on 2026-10-04 and completed at source-review scope; see
[environment/topology decision](documents/7A_ENVIRONMENT.md) and
[core pins](infrastructure/7a_verl/environment/requirements.txt). Full dependency resolution and GPU
validation remain pending. Step 2 was subsequently authorized and completed;
see [adapters and GRPO mapping](documents/7A_GRPO_MAPPING.md). Thirteen CPU tests
passed; real Parquet preparation produced 1,023 training/32 evaluation prompts,
with explicit exclusion of one empty solution in the selected training prefix.
Step 3 was authorized and completed as CPU preparation: a resolved Python 3.12 /
CUDA 12.8 dependency lock, explicit smoke configuration, thin cloud launcher,
numerical/sharding diagnostics and [runbook](documents/7A_RUNBOOK.md).
Eighteen CPU tests passed, including composition against pinned upstream YAML,
portable bundle integrity and the frozen-policy gate. GPU hooks and the complete
installed stack remain unverified; image digest selection is pending before rental.
Steps 4–5 remain proposed for review.

1. **Compatibility and topology decision.** Inspect official documentation and
   source for candidate release versions; select and pin a compatible isolated
   stack. Verify FSDP2 support, rollout colocation/resource requirements and
   checkpoint APIs. Propose hardware supporting at least two training ranks and
   rollout memory, with a bounded smoke budget, before any rental. Do not assume
   Part 5's dedicated trainer/inference split applies to verl.
2. **Minimal adapters and semantic mapping.** Inspect Part 5 prompt/grader helpers
   and reuse dependency-light logic through thin adapters. Convert MATH to verl's
   expected format with source fingerprints. Map baseline/advantage statistics,
   clipping, loss reduction, optimizer, KL/entropy, tokenizer/EOS, sampling seeds,
   fixed evaluation and truncation. Check prompt versus response counts, global
   versus per-rank batches, rollout reuse and total optimizer updates in pinned
   source; document differences from FP32/stable-layout reference behavior.
3. **CPU preparation.** Add only necessary config, adapters, a cloud runner and
   runbook. Check data/reward/config/loss behavior using small synthetic CPU inputs;
   verify syntax and configuration without checkpoints or model forwards.
   Prepare one runnable cloud command with optional path overrides and a source
   transfer procedure that does not require commits/pushes.
4. **Bounded cloud smoke, after provisioning authorization.** Generate responses,
   perform a few optimizer updates, evaluate a saved fixed subset and save a
   checkpoint. Capture world size, ranks/devices, logical/local parameter shapes,
   sharding placements and which training states are sharded. Retain resolved
   config, package/image/model/data identities, job status, throughput, memory,
   checkpoint events and logs outside ephemeral storage before cleanup.
5. **Artifact review and stop.** Verify two-rank sharding and completed workload
   from artifacts; summarize mapping and limitations. Update this plan and add a
   concise results-focused README. Reassess before 7B or a bounded comparison;
   no historical experiment suite retraining is required.

## Boundaries and progress

- [x] Create Part 7 working guidance and proposed implementation sequence.
- [x] Obtain authorization and complete 7A step 1 source review and core pins.
- [x] Complete step 2 adapters, semantic mapping and bounded CPU verification.
- [x] Complete step 3 environment/configuration/runner CPU preparation.
- [ ] Complete 7A and review retained cloud evidence.
- [ ] Complete 7B, then reassess scope and Kubernetes access/cost.
- [ ] Complete 7C and 7D. 7E requires a separate decision to continue.

All model execution is cloud-only. Local work is static/lightweight CPU checks.
Resolved dependency pins, configuration, thin MATH adapters and a cloud launcher
are implemented. Hydra and CPU data/grader dependencies were installed into `/tmp`
for validation; no Part 5
environment was changed. No GPU stack was installed, resources provisioned or
model/GPU execution performed. Numerical, tokenizer and FSDP2 integration checks
remain pending; source-level equation agreement is not GPU verification.
No staging, commits or pushes without approval; agree hardware/budget before
rentals. Keep reusable code minimal and README content focused on measured
results; runbooks hold operational detail.

Core completion is 7A–7D with evidence-backed resume and monitoring claims.
Multi-node scalability, RDMA/InfiniBand, autoscaling and production reliability
remain unclaimed. Megatron, DeepSpeed, SGLang, continued pretraining, Kubeflow,
Airflow, MLflow and OpenRLHF remain deferred; do not expand the roadmap.
