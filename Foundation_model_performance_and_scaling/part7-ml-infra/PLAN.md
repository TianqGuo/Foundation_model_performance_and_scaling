# Part 7 — ML Infrastructure

Updated: 2026-10-12

Status: 7A bounded infrastructure smoke and artifact review are complete.
Reviewed local evidence confirms two A100-SXM4-40GB training ranks, actual FSDP2
parameter/gradient/Adam sharding, six optimizer calls per rank, fixed evaluation,
checkpoint save, BF16 export, direct private Hub uploads and weights reload.
7B bounded same-host recovery is now complete: reviewed user-run evidence on two
A100-SXM4-80GB GPUs proves controlled checkpoint-boundary interruption, pinned
Hub download into a fresh path, a new Ray runtime, exact model/Adam/scheduler/saved
worker RNG fingerprints and trainer step/data restoration, followed by sharded
updates/evaluation/checkpointing. Driver RNG, vLLM server state and replacement-node
recovery are outside this claim; broader numerical/tokenizer parity remains unverified.

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
[the 7B runbook](documents/7B_RUNBOOK.md). Cloud-only state audits, checkpoint-boundary interruption, Ray placement capture
and the single `part7B.sh` experiment runner are now prepared. Targeted CPU checks
include pinned upstream configuration composition. Validation: 31 targeted configuration/diagnostic/checkpoint/recovery CPU tests
passed against the pinned upstream YAML, plus five existing Hub helper tests;
CLI, shell syntax, Python syntax/docstrings and whitespace checks passed.
These hooks, direct full-checkpoint transfers and fresh-runtime recovery were
subsequently verified by the user-run experiment reviewed below. The agent did
not load models or provision cloud resources.
Preserve 7A and its environment; no Kubernetes or backend change is introduced.

## 7C implementation sequence

User authorized 7C implementation on 2026-10-11. The first increment prepares a
CUDA 13 image using the existing isolated lock, a digest-only RayJob renderer,
and an existing-cluster GPU preflight under `infrastructure/7c_kuberay/`.
See [7C plan](documents/7C_PLAN.md) and [preparation runbook](documents/7C_RUNBOOK.md).
Ten CPU tests pass, including the actual pinned KubeRay v1.4.2 RayJob schema,
resource/storage/Secret settings and GPU-placement validation. No image build,
registry publication, cluster contact, provisioning or GPU run has occurred.

The initial manifest has a CPU head and one two-GPU worker on a selected node,
with a durable shared PVC, explicit shared memory and file-mounted Hub Secret.
Actual cluster access, base/registry digests, device-plugin/storage compatibility
and a budget remain to be agreed. A thin launcher now connects verl
to KubeRay's existing Ray runtime; 7B's local driver GPU guards/startup cannot be
used unchanged. Training, interruption/recovery and cleanup remain pending;
this preparation does not complete 7C or justify renting another GPU container.

## Optional milestone and deferred follow-ups

Core completion remains 7A–7D. These items do not block 7C/7D and do not authorize
implementation or paid resources. Reassess after the core stages before choosing
additional experiments.

- **7E — Optional SFT:** Ray Train/FSDP2, with two-rank sharding and recovery, as
  already accepted in the roadmap. Continuing requires a separate decision.
- **Larger-model practice:** reuse configurable workload/model/resource settings;
  select model, hardware and budget and validate compatibility/memory first.
- **Training TP/PP:** explore a compatible verl training backend, such as Megatron,
  after the current stages. Ray schedules workers; the training backend implements
  tensor/pipeline parallelism. This is exploratory, not a backend migration plan.
- **Shared-cluster scheduling:** evaluate Kueue integration with KubeRay RayJob
  when queues, quotas or admission control are needed. Discuss JobSet only if the
  chosen job architecture requires it; it is not a prerequisite for RayJob.
  Resource requests/limits and placement remain required within core 7C.
- **Broader recovery:** consider replacement-host/node-failure recovery and
  multi-node experiments separately. Reviewed 7B proves same-host fresh-runtime
  recovery only; future claims need their own retained evidence.
- **Additional monitoring:** add Prometheus/Grafana where useful, or optional W&B,
  beyond the built-in Ray logs/placement/metrics used initially.

Other previously named frameworks remain deferred, as listed below. These are
recorded interests, not extra committed Part 7 milestones.

## Boundaries and progress

- [x] Create Part 7 working guidance and proposed implementation sequence.
- [x] Obtain authorization and complete 7A step 1 source review and core pins.
- [x] Complete step 2 adapters, semantic mapping and bounded CPU verification.
- [x] Complete step 3 environment/configuration/runner CPU preparation.
- [x] Complete bounded 7A and review retained cloud evidence.
- [x] Complete bounded 7B recovery and review retained evidence.
- [x] Prepare first 7C container/manifest/GPU-preflight increment with CPU checks.
- [ ] Confirm Kubernetes access/cost before cloud 7C execution.
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

7B cloud startup follow-up: user logs show two idle A100-SXM4-80GB GPUs,
492 GB free disk and a stopped Python process (`STAT T`) before baseline config
creation. Likely terminal-input job control at first Hub login under `timeout`.
Authentication is now performed in the foreground before the timed experiment.
This startup fix still requires user-run cloud confirmation; no recovery is claimed.

## Reviewed 7B cloud recovery

Run `recovery_execution_20261011T022932Z_Ponp8Z` (2026-10-11 UTC) is retained
locally, with all three phase statuses successful and the interruption explicitly
planned after complete step 2. The full console has no traceback. Baseline,
interrupted and resumed phases have distinct Ray node IDs/worker PIDs on the same
physical host. Placement records bind two WorkerDict actors to two logical GPUs;
rank diagnostics use distinct visible devices 0/1 and record 338 sharded parameter
tensors each. Optimizer calls per rank are 6 baseline, 4 before interruption and
2 after restoration. The interrupted phase stops before its step-2 rollout dump;
this expected absence does not invalidate its completed checkpoint.

Both rank audits exactly match all local model/Adam state, scheduler and saved
worker RNG fingerprints. Trainer step/data fingerprints match step 2; resumed
step 3 consumes source indices `[78, 536, 279, 461]`, matching the baseline.
The restored path is `restored/global_step_2`, not the producer checkpoint.
The resumed phase completes a 16-response rollout, 32-prompt final evaluation
and step-3 checkpoint. Saved/current Python, package, source, data and model
identities match. The local copy has 1,290 evidence files (~5.8 MiB), without
checkpoint tensor files; full state is retained directly on the private Hub.

Private dataset: `sclion/part7-7b-recovery-execution-20261011T022932Z-Ponp8Z`.
Checkpoint commit: `f270b988649abdd9d461baad8e654ecd2eb44dfd`.
Evidence commit: `e1df90574ae7acf70cfef67ee0b282b1a92878e9`.
Receipt and review are retained locally. Upload took 82.0 s; download plus
integrity 35.4 s. Native per-rank load took 7.55–7.59 s; audit overhead
20.58–20.60 s; completed audit to first optimizer update 22.87–22.88 s.
These scopes include this host's cache/network conditions, not cold-start or
production benchmarks. Total recorded phase/transfer durations are about
11 minutes, excluding initial environment setup/authentication.

This completes 7B at its accepted bounded scope. It does not establish vLLM/server
or driver RNG recovery, replacement-host recovery, exact sampled-output/final-weight
parity or model-quality improvement. Cloud artifacts are uploaded and local small
evidence is reviewed; the user may release the instance. No commits or pushes
were performed. The foreground-authentication fix was not exercised by this run,
which used a saved-login workaround; it remains CPU-checked.

## Second 7C increment — native phases

Added `run_7c.py`: existing-Ray connection, worker-side model preparation, native
verl TaskRunner on the GPU worker, and reused 7B interruption/restore audits.
The renderer supports baseline/interruption/resume jobs; resumed state must match
model/data/source/package identities and the recorded image digest. Cloud paths
must stay under the persistent `/workspace` mount. Fifteen CPU tests pass,
including all four phase manifests against the pinned CRD; shell syntax, CLI,
Python syntax/function docstrings and scoped whitespace checks pass.

Live image/build/operator compatibility, fresh-RayJob recovery, direct Hub backup,
job submission/evidence capture and cleanup remain pending. This does not complete
7C; see the updated plan/runbook for supported commands and limitations.

## Third 7C increment — job lifecycle

Added `job_control.py` and `part7C.sh job`: exact generated manifest validation,
explicit Kubernetes context, server dry-run/create, bounded monitoring, pod/driver
logs and lightweight PVC evidence retention before optional owned-job cleanup.
Actual submitted settings keep the cluster alive until collection. Cleanup
verifies job UID and RayCluster/Ray-pod absence; PVC/Secrets/nodes are preserved.
Failures retain resources for inspection. Twenty-five 7C CPU tests pass, including
mocked API outcomes and archive exclusions. No live cluster was contacted.

Cloud image/publication/access/storage compatibility and end-to-end recovery still
need verification. Direct Hub state retention/restore, portable data staging and
multi-phase experiment orchestration remain pending. 7C is not complete; the
runbook documents the new single-job flow and cleanup scope.

## Fourth 7C increment — direct Hub state retention

Added `checkpoint_transfer.py` and optional upload/restore arguments through the
renderer/controller/native launcher. Transfers use the mounted Secret, private
repositories, complete 7B checkpoint manifest validation and immutable commit
receipts. Fresh-path pinned downloads are verified before native resume; upload
failure prevents successful phase cleanup. Thirty-two CPU tests pass, including
mocked transfer behavior and renderer/controller option preservation. CLI, syntax,
docstrings and scoped whitespace checks pass. No network transfer, model loading,
GPU execution, image build, cluster contact, provisioning, staging or Git commit/push
was performed. Registry publication, portable data staging and multi-phase workflow
remain pending; real image/Hub/GPU Kubernetes/recovery/cleanup verification is required.

## Fifth 7C increment — portable data and multi-phase runner

Added a bounded CPU staging pod helper and `part7C.sh experiment` coordinating
preflight/baseline/interruption/private-Hub-restore/resume, lightweight evidence
retention and cleanup. Review combines the verified 7B state-audit rules with fresh
worker pod/Ray IDs, same-node/image placement, matching pinned transfer receipts,
native fresh-path resume and expected sharded optimizer calls. Forty CPU tests
pass; CLI, shell, syntax/docstring and scoped whitespace checks pass. No live
Kubernetes/Hub transfer, model/GPU execution or Docker build was performed.

The source workflow is now prepared; Kubernetes access, base/published image
digests, image build and hardware/budget remain unresolved. Live preflight,
training, pinned recovery, durable artifact review and cleanup must still be
verified before 7C completion. Do not provision a normal GPU container assuming
it supplies Kubernetes. No commits/pushes or changes to Parts 1–6 were made.

## 7C cloud-readiness follow-up

Added read-only `cluster_check.py` / `part7C.sh cluster` for node readiness/capacity,
current placement constraints, shared PVC access/capacity and candidate operator/
RayJob API availability. Forty-six 7C CPU tests pass; CLI, shell, syntax/docstrings
and scoped whitespace checks pass. No real cluster or credentials were inspected.
Next is confirming Kubernetes access and an agreed hardware/budget, building and
publishing the image, then retaining live experiment evidence. 7C stays incomplete.

## 7C fresh Ubuntu GPU VM setup

Added `setup_cluster.py` / `part7C.sh setup --check|--cloud`: explicit fresh-host
checks, pinned K3s and Helm-managed NVIDIA/KubeRay charts, pre-bound same-node
Retain PV/PVC, protected user kubeconfig and optional hidden-prompt Hub Secret.
Preserves host CUDA/Docker setup; no VM provisioning or local GPU execution.
CPU Ray heads/staging templates explicitly hide unallocated NVIDIA GPUs.
Fifty-one CPU tests pass. Lambda diagnostics confirm a suitable-looking KVM host,
not working Kubernetes; the user destroyed it before setup. All bootstrap/image/
GPU-pod/recovery verification remains pending. Source setup objects are retained;
credentials are excluded. VM-backed storage survives pods but not VM deletion.
No changes to Parts 1–6, commits or pushes.
