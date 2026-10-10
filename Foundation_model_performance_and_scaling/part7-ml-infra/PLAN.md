# Part 7 — ML Infrastructure

Updated: 2026-10-09

Status: 7A bounded infrastructure smoke and artifact review are complete.
Reviewed local evidence confirms two A100-SXM4-40GB training ranks, actual FSDP2
parameter/gradient/Adam sharding, six optimizer calls per rank, fixed evaluation,
checkpoint save, BF16 export, direct private Hub uploads and a successful new
smoke from the uploaded model's pinned commit. Full training-state restoration
and broader numerical/tokenizer parity remain unverified; those are not implied
by a weights-only reload. 7B design/source review has started; checkpoint integrity/transfer helpers and resume preview are prepared; runtime
recovery audits and cloud verification remain pending.

`infrastructure/7a_verl/part7A.sh --upload-hf` is the single cloud entry point:
setup/reuse, automatic authentication/destination naming, data/model preparation,
smoke execution, export and direct artifact retention. Public uploads are not
implemented. No commits or pushes were performed by the agent.

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
CUDA 13 dependency lock, explicit smoke configuration, thin cloud launcher,
numerical/sharding diagnostics and [runbook](documents/7A_RUNBOOK.md).
Eighteen CPU tests passed, including composition against pinned upstream YAML,
portable bundle integrity and the frozen-policy gate. GPU hooks and the complete
installed stack remain unverified; image digest selection is pending before rental.
Steps 4–5 subsequently completed through user-run cloud execution and retained
artifact review, with the limits recorded below.

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

## Retained first cloud smoke evidence

Local evidence from `smoke_20261005T035949450979Z` was inspected: job status
reports success; diagnostics record ranks 0/1 on distinct visible GPUs, world
size 2, 338 sharded parameter tensors on each rank, sampled sharded gradients
and Adam moments, and six optimizer calls per rank. Both fixed validation dumps
contain 32 rows. Training rollouts 1/2 have zero scores; rollout 3 contains one
positive score among 16 responses. This supports actual distributed execution
but does not establish a reliable improvement in model quality. Full checkpoint
reload/export was not verified before the user chose to discard the instance.
The new Hub authentication/export/upload and pinned-revision reload path still
require cloud verification; 7A is not yet marked complete.

## First Hub upload and reload verification

The locally retained `smoke_execution_20261005T045405Z_dwaxLH` reports success
and has its upload receipt, identities, full console, per-rank diagnostics and
rollout/evaluation dumps. Hardware was two A100-SXM4-40GB GPUs. Both ranks record
six optimizer calls and 338 sharded parameter tensors. The model export and Hub
file-listing checks completed; the receipt records model commit
`40c33b6ad0a7a2879e7b5a8f5f6a52baaa94a584` and artifact commit
`e137c5dba237c8fe539152b31aed42bd09e681c5`. Full state was intentionally excluded.
Fixed evaluation scores were 5/32 before training and 8/32 afterward, which are
smoke observations, not evidence of reliable improvement. Training rollout scores
were 0/16, 1/16 and 1/16. Local weights/checkpoints were not downloaded.

The reload run `smoke_execution_20261005T052347Z_utvjK2` and its full console
are now retained locally. Its identities select the uploaded model repository
and exact model commit above, and job status reports success. Both ranks again
record six optimizer calls, 338 sharded tensors, sampled sharded gradients/Adam
moments and finite frozen-policy gates with maximum observed delta zero.
Checkpoint save, initial/final fixed evaluation and completion are recorded,
with no traceback in the execution console. Reload training scores were zero
for all three 16-response rollouts; evaluation was 9/32 before and 6/32 afterward.
These small, stochastic/mixed-precision smoke results do not establish quality
improvement or regression. The earlier upload run logged nonzero gradients on
steps 2/3. This completes bounded 7A execution/export/reload evidence, not exact
training-state recovery. The user may release the instance after this retention
review; weights are on the Hub and logs/receipts are local. Full checkpoints from
these runs were intentionally not backed up.

## Hugging Face artifact retention

User authorized direct cloud-to-Hub retention on 2026-10-05. The 7A shell runner
now accepts `HF_MODEL_REPO` for a private BF16 model/tokenizer export and
`HF_ARTIFACT_REPO` for a private dataset repository dedicated to the run's evidence.
Full training-state upload is opt-in through `HF_UPLOAD_CHECKPOINT=1`; the default
evidence upload excludes checkpoint weights and optimizer state. Neither repository
is required for disposable smoke runs. `--upload-hf` provides a single-command flow: reuse cached/environment credentials
or prompt once, verify the token account, choose run-specific private destinations
and check access before training. Explicit destinations remain optional overrides.
Authentication uses existing Hub tooling,
with no added GPU dependencies. No repository was created or uploaded by the agent.

The thin `hf_artifacts.py` helper delegates merging to pinned verl and transfers
artifacts directly from cloud. It records commit revisions and verifies remote file
listings. Cloud export, actual Hub upload and pinned-revision workload reload are now verified;
file presence is not verified training-state restoration. All 26 CPU tests passed,
including artifact selection, numeric checkpoint selection and private-repository
enforcement; shell syntax checks passed. The subsequent authentication/name
simplification passed five targeted CPU tests and shell syntax/help checks. A new cloud smoke can use
`--model-id` / `--model-revision` to download exported weights automatically and
start fresh. Full-state recovery remains a 7B requirement, with no resume interface
claimed by 7A. See the runbook for commands, retrying retention and artifact policy.

## 7B implementation sequence

User authorized continuing to 7B on 2026-10-09. The first increment establishes
[the recovery design and acceptance plan](documents/7B_PLAN.md): same-node
two-rank Ray placement/metrics, complete checkpoint manifests, private direct Hub
retention, controlled interruption and native verl full-state resume in a fresh
Ray runtime. Audit model/Adam, scheduler, step/data progress and saved RNG;
disclose unsupported rollout-server state. Checkpoint manifests/integrity validation, private Hub transfer helpers and native
resume preview are implemented under `infrastructure/7b_ray/`; see
[the 7B runbook](documents/7B_RUNBOOK.md). Runtime state audits, interruption,
observability and a single cloud runner remain pending. No model loads or cloud
provisioning occurred.
Preserve 7A and its environment; no Kubernetes or backend change is introduced.

## Boundaries and progress

- [x] Create Part 7 working guidance and proposed implementation sequence.
- [x] Obtain authorization and complete 7A step 1 source review and core pins.
- [x] Complete step 2 adapters, semantic mapping and bounded CPU verification.
- [x] Complete step 3 environment/configuration/runner CPU preparation.
- [x] Complete bounded 7A and review retained cloud evidence.
- [ ] Complete 7B, then reassess scope and Kubernetes access/cost.
- [ ] Complete 7C and 7D. 7E requires a separate decision to continue.

All model execution is cloud-only. Local work is static/lightweight CPU checks.
Resolved dependency pins, configuration, thin MATH adapters and a cloud launcher
are implemented. Hydra and CPU data/grader dependencies were installed into `/tmp`
for validation; no Part 5
environment was changed. No GPU stack was installed, resources provisioned or
model/GPU execution performed. Broader numerical/tokenizer equivalence checks remain pending; FSDP2
sharding and frozen-policy checks are verified in the bounded cloud smoke; source-level equation agreement is not GPU verification.
No staging, commits or pushes without approval; agree hardware/budget before
rentals. Keep reusable code minimal and README content focused on measured
results; runbooks hold operational detail.

Core completion is 7A–7D with evidence-backed resume and monitoring claims.
Multi-node scalability, RDMA/InfiniBand, autoscaling and production reliability
remain unclaimed. Megatron, DeepSpeed, SGLang, continued pretraining, Kubeflow,
Airflow, MLflow and OpenRLHF remain deferred; do not expand the roadmap.
