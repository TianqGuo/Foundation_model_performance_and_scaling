# Part 7 — ML Infrastructure

Updated: 2026-10-25

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

7C is complete at its bounded same-node scope. Reviewed fresh-install E2E run
`part7-e2e-20261018-031219` on two H100 80 GB GPUs passed GPU preflight,
training/interruption/private-Hub restore, native state audits and four verified
job cleanups. Local synced evidence was independently re-reviewed. 7D serving
has passed cloud image build/model-free imports/publication, with lightweight
evidence retained locally. GPU inference/HTTP/E2E remain unverified. The published
latest image uses Ray 2.59/vLLM 0.27/Torch 2.13 and passed model-free cloud
build/import/publication checks. All five logs/receipts were checked locally;
full GPU serving/E2E remains pending and the build VM has been released.
See [current 7C results](documents/7C_PLAN.md) and [runbook](documents/7C_RUNBOOK.md).
Historical increment notes below describe the status at their original time;
this summary, checklist and milestone documents state the current status.

`infrastructure/7a_verl/part7A.sh --upload-hf` is the single cloud entry point:
setup/reuse, automatic authentication/destination naming, data/model preparation,
smoke execution, export and direct private artifact retention. The 7C training
image is separately published publicly on Docker Hub. No commits or pushes were
performed by the agent.

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
`workloads/` holds reusable MATH and generation adapters. Implemented milestone
directories also include `7b_ray/`, `7c_kuberay/` and the first `7d_serve/`
increment. Add an optional 7E directory only if selected; tests and results remain
at the project root.

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
These are platform-management relationships, not a strict above/below library
stack. verl defines the RL workflow and uses Ray to schedule/call actors. FSDP2/
PyTorch Distributed uses NCCL for GPU collectives, which do not pass through Ray's
actor messaging. Current 7C is one head pod and one two-GPU worker pod with two
training ranks, not one actor/task. Kubernetes places pods; Ray places application
tasks/actors within its cluster. Cross-worker-pod training remains unverified.

## Accepted roadmap and completion evidence

Core scope is **7A–7D**. The table below states current completion and the evidence
needed for each milestone; optional extensions are listed separately and do not
expand 7D's acceptance criteria.

| Core milestone | Current status | Completion evidence / remaining acceptance |
| --- | --- | --- |
| 7A — Framework workload | **Complete: bounded cloud smoke.** | Pinned verl/FSDP2/vLLM workload on two A100 40 GB GPUs; actual parameter/gradient/Adam sharding, generation, updates, fixed evaluation, merged export, private Hub retention and pinned weights reload reviewed. This is infrastructure evidence, not model-quality improvement or full training-state recovery. See [7A runbook](documents/7A_RUNBOOK.md). |
| 7B — Ray operations and recovery | **Complete: same-host recovery.** | Two A100 80 GB ranks; step-2 checkpoint interruption, complete private Hub retention/pinned download, fresh Ray runtime and step-3 continuation. Model/Adam/scheduler/saved worker RNG, trainer step/data and next batch restoration reviewed. Driver/vLLM server state and replacement-host recovery excluded. See [7B plan](documents/7B_PLAN.md). |
| 7C — Kubernetes training | **Complete: fresh-install, same-node E2E.** | One Lambda VM/two H100 80 GB GPUs; fresh K3s/KubeRay setup, automatic data preparation/PVC staging, GPU preflight, native two-rank training, checkpoint interruption/pinned Hub recovery in distinct RayJob pods/runtimes, retained evidence and verified cleanup. Accepted run `part7-e2e-20261018-031219`; no multi-node/replacement-node/bitwise claim. See [7C results](documents/7C_PLAN.md). |
| 7D — Serving | **In progress: serving configuration, image diagnostic, benchmark and lifecycle CPU preparation; cloud unverified.** | Existing 7A BF16 model export/revision selected; configurable native Serve settings, RayService renderer and model-staging code pass CPU/schema checks. Candidate dependency lock/build diagnostic, HTTP measurement helper and owned-service lifecycle/restart controller are prepared with CPU tests. Still required: compatible cloud-validated serving image, cloud execution of the composed E2E runner, actual export serving through RayService, HTTP readiness, measured requests/s/generated tokens/s/streaming TTFT/output-token timing/p50–p95 latency/errors/replica status, owned-worker restart/readiness recovery, retained evidence and cleanup in one cloud E2E workflow. See [7D plan](documents/7D_PLAN.md) and [runbook](documents/7D_RUNBOOK.md). |

Record serving hardware, input/output lengths, concurrency, warmup, streaming and
client/measurement scope. Distinguish engine metrics from client latency and GPU
memory/cache allocation from compute utilization. Training and serving can run
sequentially and use separate pinned images. Fractional Ray GPU requests do not
isolate GPU memory. CPU/schema checks alone never establish cloud completion.

| Optional extension after core work | Status | Proposed evidence / decision |
| --- | --- | --- |
| 7E — Supervised-training platform reuse | Optional; not started. | Small Ray Train/FSDP2 SFT workload with two-rank sharding and recovery; no required Part 6 8B retraining. Requires a separate decision. |
| Shared resource pool / Kueue | Optional after 7D; not implemented. | Multiple model/image/configuration workloads sharing a limited pool; configurable resource profiles, queues/quotas/priority and storage/credential isolation. Demonstrate competing workloads, waiting/admission/resource release and cleanup. Validate version/workload support; quota reservation is not physical placement. JobSet is an alternative for suitable jobs, not a wrapper required around RayJob. |
| Larger models, training TP/PP, broader recovery and monitoring | Deferred options; not acceptance requirements. | Select the experiment, compatibility/topology, hardware and budget before implementation; measure each new claim separately. |

See [optional milestone and deferred follow-ups](#optional-milestone-and-deferred-follow-ups)
for detailed scope. These options do not authorize implementation or paid
resources and do not block 7D. The platform is configurable, but concurrent
multi-workload/shared-pool and multi-node operation are not currently verified.

For future cloud runs, confirm hardware, Kubernetes access and spending cap first.
An ordinary restricted GPU container is not a Kubernetes host. If cluster access
is unavailable, cloud-GPU serving can be developed separately, but only actual
RayService deployment completes the Kubernetes-serving acceptance criterion.

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
- **Shared resource pool — optional after 7D:** evaluate Kueue admission for
  multiple training/serving workloads competing for a limited GPU pool. Validate
  the selected Kueue/KubeRay versions and workload-type support before integration.
  Separate workload settings (model revision, image digest, algorithm/serving
  configuration) from resource profiles (GPU type/count, CPU/RAM, replicas,
  parallelism) and admission policy (queue, quota, priority, interruption policy).
  Generalize fixed names, node placement and artifact/storage paths; define
  namespace, credential and storage isolation rather than assuming unique names
  provide security. Keep training and serving admission/recovery policies distinct.
  Acceptance: submit at least two independently configured competing workloads;
  demonstrate one waiting when quota is exhausted, admission after resources are
  released, correct GPU placement and resource accounting, artifact/credential
  isolation, and owned-resource cleanup. Test configured priority/preemption only
  if enabled; quota reservation alone is not proof of physical schedulability.
  JobSet is an alternative for suitable distributed jobs, not a required wrapper
  around RayJob. This follow-up requires a separate scope/hardware/budget decision;
  it does not block 7D or claim current shared-pool/multi-node readiness.
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
- [x] Confirm Kubernetes access and review fresh cloud 7C execution evidence.
- [x] Complete bounded 7C and retain/review fresh E2E evidence.
- [x] Prepare first 7D model configuration, RayService/staging code and CPU/schema checks.
- [ ] Complete serving image, lifecycle/benchmark/restart runner and cloud 7D E2E review.
- [ ] Separately decide optional 7E/shared-pool or other follow-ups after core work.

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


2026-10-13 cloud bootstrap correction: the user-run Lambda setup reached the
kubeconfig write and exposed invalid `Path.open(opener=...)`. Corrected to builtin
exclusive mode-0600 open; added a narrowly validated early-failure continuation
that skips K3s reinstallation. Fifty-three CPU tests pass, including actual private
file creation and continuation guards. Cloud completion remains pending; no
installation success, GPU pod or 7C completion claim is made.


NVIDIA placement correction: user diagnostics show Ready K3s and Running KubeRay,
but plugin DaemonSet desired=0 and no allocatable GPUs. The pinned NVIDIA chart
requires a GPU-present/NFD label, while this bootstrap does not install NFD.
Setup now labels the hardware-verified node `nvidia.com/gpu.present=true` before
chart creation. Fifty-four CPU tests pass. Live GPU plugin readiness remains
pending; current instance can be repaired by that node-label command without
reinstallation. MPS desired=0 is expected with no MPS-capable label.


Image-build preparation follow-up: reviewed pasted successful Lambda bootstrap
receipt `setup_20261014T040415Z` (two H100 GPUs advertised, Bound PVC, operator/API,
Hub Secret created). This is not GPU kernel or token-authentication evidence.
The build helper now resolves a supplied base tag to an immutable digest, supports
sudo Docker, retains logs/identity hashes and clears inherited image boot hooks.
No image was built/published; registry/base selection and live preflight are pending.
No commits/pushes or changes to Parts 1–6.


Docker Hub publication preparation: user selected public `sclion/part7-training`.
Added `part7C.sh image --check|--cloud [--publish]` using candidate official NVIDIA
CUDA 13.0.3 Ubuntu 22.04 base and uv 0.12.19, resolved to recorded digests. Build
installs its own utilities, preserves host environments, tests Python startup with
GPU visibility disabled, and optionally publishes via hidden-token temporary
Docker authentication. Registry digest receipts are verified by pull; no Docker,
Hub publication or GPU execution performed locally. Fifty-seven CPU tests pass;
cloud build and GPU preflight remain pending. No Git commits/pushes.


Reviewed user image-build/publication output: Python 3.12.14 startup succeeded and
`sclion/part7-training@sha256:211e567f0bfd55617db36102b6e00891e6a136420850788c2874045dffcb3294`
was pushed and pulled by digest. Post-publication cleanup failed because sudo Docker
created root-owned buildx files in the temporary auth directory. Fixed cleanup to
use the same privilege level, scoped only to its newly allocated temporary path.
Fifty-eight CPU tests pass, including cleanup after failures and sudo scoping;
one optional pinned-schema test is skipped because its temporary jsonschema/CRD
fixtures are unavailable in the current local environment (previously verified).
Actual corrected cleanup and GPU/import preflight are still pending cloud evidence;
publication success does not imply GPU compatibility. No rebuild is needed solely
for this controller-side cleanup correction. No Git commits/pushes.


Layer optimization after reviewed cloud history: final recursive ownership change
added 16.7 GB above an 8.49 GB installation layer; source COPY layers were tiny.
Moved ownership changes into the dependency installation RUN and source COPY
ownership flags; uv cache now uses a BuildKit mount with copy link mode. This
prevents the separate ownership-copy layer and excludes package cache from images.
No package/base/model/algorithm changes. CPU suite remains 58 passing/one optional
schema skip; Docker/GPU execution was not run locally. Optimized cloud size/time
and GPU compatibility remain pending; one dependency rebuild is required before
future source-only cache reuse. Existing published digest preserved.


Node-readiness correction: user logs show successful K3s startup and a Ready
`part7-gpu` after the initial named-node wait failed; exact original stderr was
not retained, so a registration race is suspected, not proved. Setup now polls
registration/transient API availability before waiting for Ready, with progress
messages. `--continue-after-k3s ORIGINAL_FAILED_DIRECTORY` also permits reusing an
exact, user-owned mode-0600 kubeconfig before storage/chart creation, without
reinstallation. Existing symlinks/different configs are rejected. Sixty CPU tests
pass; one optional schema test skipped. Cloud continuation still needs review.


Clean-retest requirement: user correctly identified missing post-fix fresh-install
E2E evidence. Added full install control-flow CPU scenarios (external host/API
commands simulated; actual private config/receipt writes and validators execute).
Sixty-two CPU tests pass with one optional schema skip. These are not cloud E2E.
Runbook now describes guarded uninstall of the dedicated pre-storage failed K3s
cluster and the normal fresh setup command, preserving repository/build evidence
and Docker image. User cloud clean installation plus GPU preflight must be
reviewed before declaring the path verified. No local K3s uninstall/install,
cloud mutation, provisioning, commits or pushes performed.


First preflight submission was rejected before cluster creation: the controller
held the cluster with shutdownAfterJobFinishes=false but retained the renderer's
finished-job TTL, an invalid KubeRay combination. Controller now removes TTL when
holding pods for evidence; it also prints status changes and stops promptly on
UID-scoped InvalidRayJobSpec events. Sixty-three CPU tests pass; one optional
schema check skipped. Cloud submission/preflight still pending; no pods/training
were started by the rejected job. This controller-only fix needs no image rebuild.


GPU preflight reached pods after ~81-second image pulls, but KubeRay health probes
failed with `wget: command not found`, causing head/worker restarts. Added wget
in a small Dockerfile layer after dependency installation (preserves the large
cached environment). Image publication now requires a nonroot no-network/no-GPU
check of bash/wget/tar/git/ray/python before push. Sixty-four CPU tests pass and
one optional schema test skips; fixed image build/pod readiness remain pending.
Old failed RayJob resources must be retained/released before fresh two-GPU retry.
Kubernetes reinstall is unnecessary. No local Docker/model/GPU execution or Git
commits/pushes performed.


Preflight with wget-fixed image reached Ray RUNNING and GPUProbe workers but failed
inside vLLM/PyTorch import: UID 1000 had no passwd entry, so getpass.getuser raised
KeyError in compiler-cache initialization. Added a real UID/GID 1000 account via
small late image layer. The no-GPU image check now resolves passwd/getpass, checks
writable HOME and imports torch._inductor.codecache before publication. Sixty-four
CPU tests pass, one optional schema check skips; image check itself must execute
on cloud (local tests inspect invocation only). No inference/model training or
GPU success claim. Cloud rebuilt-image/preflight evidence remains pending.


One-command cloud E2E runner added after user feedback. `part7C.sh e2e --cloud`
composes native fresh setup or existing-cluster validation, automatic cloud-side
data preparation, staging and the complete preflight/training/Hub recovery audit.
`--fresh-setup` rejects existing K3s to distinguish genuine fresh-install evidence.
Confirmed Part 5 train/validation are Git-tracked; local rsync is not required.
Data is prepared with the existing published image, original five input hashes,
1023/32 selection, readonly Part 5 mount and no container GPU/network. Controller
needs no image rebuild. Sixty-nine CPU tests pass; one optional schema skips.
User pasted clean setup and later GPU preflight/controller success were reviewed;
full wrapper cloud execution/training-recovery remains pending. Local orchestration
tests simulate external systems and are not cloud E2E. No provisioning, Git commit/
push, host dependency changes or edits to Parts 1–6.


## 7C closeout and 7D handoff

Accepted UTC run `part7-e2e-20261018-031219`: fresh K3s installation, automatic
1,023/32 bundle preparation/staging, GPU preflight, baseline/interruption/pinned
private-Hub recovery all passed on one Lambda VM with two H100 80 GB GPUs.
Model/Adam/scheduler/saved worker RNG and trainer step/data audits passed across
distinct worker pods/Ray runtimes, with step-2 -> step-3 and next indices
`[78, 536, 279, 461]`. Optimizer calls/rank were 6/4/2. All four cleanup receipts
verified. Local retention: 87 files / 1,292,168 bytes / zero checkpoint tensors;
local `experiment.review` re-execution passed. Full checkpoint/manifest retained
at `sclion/part7-7c-part7-e2e-20261018-031219-interrupt` revision
`122e784fddc0a315a65e5ad9b2ed932dcc41b117`. See 7C_PLAN for image identity/timings.

7C is complete within its accepted same-node scope. No automatic crash/reboot,
replacement-node, vLLM server/driver RNG restoration, bitwise parity or scalability
claim. Next review 7D's exported 7A model identity, compatible serving image and
RayService/benchmark/restart plan before implementation/provisioning. The existing
training-state repository is not a serving-ready export. Keep larger-model/TP/PP,
Kueue, broader recovery and 7E optional. No commits/pushes without approval.


## First 7D increment — pinned model and RayService preparation

User authorized 7D implementation after releasing the 7C VM. Added milestone
`infrastructure/7d_serve/` with configurable immutable 7A model identity/native
Serve LLM settings, RayService renderer reusing 7C pod conventions, cloud-only
locked/atomic snapshot staging and a shell renderer entry point. Serving stays
one replica/one whole GPU and uses the existing 7A BF16 export rather than 7C
training-state tensors. Added 7D plan/runbook and generated reading example.
Eight new CPU tests pass including KubeRay 1.4.2 RayService schema and synthetic
snapshot-integrity tests. Serving dependency overlay is candidate input only;
no claim that native Ray Serve LLM works with the training stack until resolved
and import-tested. Serving image, lifecycle/benchmark/restart and one-command
cloud E2E are next. No rent/build/model download/load/GPU execution, registry
publication, Git commit/push or changes to Parts 1–6/accepted 7A–7C code.


## Serving dependency/import gate — 2026-10-20

Added a metadata-resolved candidate serving overlay (197 packages), a separate
Dockerfile based on the accepted immutable 7C image, and `part7D.sh image --cloud`
for build/import diagnostics. This does not change 7A–7C or publish images.
The lock preserves the candidate Ray 2.49.2/vLLM 0.12 GPU stack; successful
resolution does **not** establish native Serve compatibility. Native Serve imports,
configuration validation and explicit AsyncLLM constructor signature checks must
pass inside the candidate image. Failures retain build/import evidence and block
acceptance. These checks never instantiate an engine or load/download a model.

API review found version-sensitive Serve-to-vLLM calls requiring this extra gate;
a serving-only Ray/version change may still be necessary. No serving image has
been built or validated on cloud. Lifecycle, HTTP/load/restart checks and the
single-command full E2E runner remain pending. Do not rent a VM for this partial
workflow yet. Local tests cover signature mismatch detection with synthetic
functions; they are not native runtime or cloud E2E tests.


## 7D HTTP measurement increment — 2026-10-20

Added dependency-light `infrastructure/7d_serve/benchmark.py` and the shell
`benchmark` action for the future E2E controller. It sends bounded concurrent raw
`/v1/completions` requests over a cloud-VM loopback port-forward, with fixed
warmup, temperature/seed, token cap and R1 `</answer>` stop handling. It does not
load models or apply a chat template. JSONL input contains already rendered
prompt strings; source-bundle validation/preparation will be wired by E2E.

SSE parsing requires an explicit terminal event, finish reason and valid engine
usage. Successful request and generated-token throughput use the full measured
wall interval; failures remain in error counts/rate. Reports include p50/p95
latency, first-text TTFT and an explicitly labeled text-event-span/output-token
estimate. SSE chunks are never counted as tokens. Warmup is excluded, failed
warmup aborts, and reports hash prompts rather than retain generated text. The
request cap is 128, concurrency at most 16 and output cap at most 512 tokens.

Seven synthetic tests verify framing/usage, truncated/error responses, timeouts,
zero-token output, failure accounting, warmup exclusion, input bounds and request
semantics. No HTTP/model/GPU execution occurred. Native endpoint support for
streamed usage and stop-string inclusion still needs cloud validation. Service
lifecycle/readiness/restart/evidence cleanup and single-command E2E are next;
7D is not complete and no VM rental is needed yet.


## 7D lifecycle/controller increment — 2026-10-20

Added `infrastructure/7d_serve/service_control.py`, a VM-side controller for the
future single-command E2E workflow. It validates the exact renderer output before
creation, refuses existing service names/evidence directories, records created
service/cluster UIDs and rejects changed service generations or cluster ownership.
Readiness requires current-generation KubeRay Ready, a nonempty healthy native
Serve deployment status and a ready endpoint, followed by HTTP catalog checks and
real completion warmup. Waits are bounded with progress messages.

The controller opens a private VM-loopback port-forward to its owned head pod,
runs the HTTP benchmark, retains status/placement/current and previous pod logs
(including model init), and deletes one owned worker with an atomic UID precondition.
Recovery requires a different worker UID on the same node and successful inference.
Observed pod readiness loss, HTTP probe failures and recovery timing are recorded;
HTTP downtime is not assumed. It benchmarks again after recovery. Cleanup requires
a matching successful review, uses an atomic service UID precondition, verifies
service/cluster/pod absence and preserves PVCs/Secrets. Failures retain resources
and best-effort evidence; a local tunnel is always stopped.

Eleven additional synthetic lifecycle tests cover status/generation/owner gates,
UID-protected deletion, restart/recovery, review-before-cleanup, failure retention
and tunnel termination. No kubectl, HTTP, model or GPU execution occurred. The
controller is an internal E2E helper, not a new manual stage users must run.
Full cloud E2E wiring, prompt provenance preparation and native serving image/API
compatibility remain pending; do not rent an instance yet.


## 7D E2E composition increment — 2026-10-22

Added `infrastructure/7d_serve/e2e.py` and `part7D.sh e2e --check/--cloud --image`
as the composition layer. CPU preview verifies accepted source hashes and renders
settings without model/network operations. Cloud mode requires an explicit serving
image digest, checks native imports before Kubernetes changes and verifies the
image's pod-used source hashes against the checkout. A failed import gate stops
before installation and retains image diagnostics and an overall failure receipt.

The workflow then installs/reuses accepted K3s/KubeRay, checks storage/operator,
Secret existence and RayService API access, prepares the same seed-12345 fixed 32
MATH raw prompts directly from cloned/hash-verified Part 5 files, and invokes the
owned service/HTTP/restart/cleanup controller. Prompt JSONL and source/selection/
adapter hashes are retained without solutions; no laptop data copy, Parquet or
model import is needed for this preparation. Model/revision/ID and serving limits
remain configurable. Fresh mode refuses existing installations; no automatic reset.

Five additional CPU tests cover real frozen prompt provenance, stale image sources,
fresh-mode refusal, image-gate failure before setup and the simulated complete
reuse path. The CPU preview passed. This is source preparation, not cloud E2E
execution. The candidate Ray/vLLM Serve integration still needs a compatible native
image/import review and build/publication workflow before recommending rental.
The normal full-run source command now exists, but no 7D cloud success is claimed.


## Serving candidate update and publication gate — 2026-10-22

7D now selects Ray Serve LLM 2.53.0, whose official LLM dependency metadata targets
vLLM >=0.12.0, and resolves a separate 219-package Python 3.12/Linux lock. Eight
GPU/tokenizer/schema constraints are carried from 7A: Torch/torchaudio/torchvision
CUDA 13 wheels, flash-attn, CUTLASS DSL, Transformers, NumPy and Pydantic. The
training lock is not imposed wholesale, so the serving Ray version can differ.
No packages were installed locally and no model was loaded. Accepted 7A–7C files
and published image are unchanged; only the derived serving image will differ.

Pinned source review found the newer public deprecation import and vLLM
`init_app_state` signature handling. Updated native gate imports to the 2.53
adapter layout and included executor/frontend/metrics imports. These are checks
to execute on cloud, not evidence of native compatibility already passing.

Added a thin serving-image publisher reusing 7C credential-directory/digest helpers.
The shell image action supports --check/--cloud, optional --publish and repository
override; only --publish authenticates/uploads. Build/import failure prevents push.
The existing public repository uses new 7d timestamp tags; output records an immutable
registry digest, verified pull and explicit inference/E2E-unverified fields. Tests
simulate no-publish, failed native gate and successful registry receipt sequencing.

Source wiring, separate candidate resolution and image publication support are
prepared. Next is a bounded cloud build/import check, then the complete E2E run
only if that gate passes. Agree hardware and total spending cap before rental.
See the current stack and exact sequence in the [7D runbook](documents/7D_RUNBOOK.md).
No provisioning, Docker build, image push, inference or Git commit/push occurred.


## 7D first build failure correction — 2026-10-22

User log showed the serving Dockerfile passed install-style `-r` to `uv pip sync`.
Removed the invalid flag and verified the extracted Docker invocation against
real uv in an offline, empty temporary virtual environment (no dependencies/models
installed). Added regression coverage and WSL rejection before host GPU/Docker
operations, because the logged path/prompt appears local and the earlier systemd/
GPU guard admitted WSL. The native cloud image build/import/E2E remains pending;
the failed build did not reach image push or model execution. No Git operations.


## Cloud native-data import correction — 2026-10-24

User's cloud serving image built but failed native imports: resolved PyArrow 26
required NumPy >=2 at runtime despite the retained NumPy 1.26.4 candidate. No push
or model execution followed. Pin the accepted 7C pair PyArrow 19.0.1/pandas 2.2.3
in the separate serving environment and re-resolve the lock. Add a real in-memory
NumPy/pandas/Arrow filesystem/Parquet round-trip check before expensive image export,
and include it in the native runtime receipt. Full cloud Ray/vLLM import, serving
and E2E verification remain pending. No changes to accepted 7A–7C implementations.

Correction verification: the regenerated serving lock has 220 packages. An isolated
Python 3.12 data-only native test passed; the combined suite ran 110 tests with
108 passes/two optional schema skips. Full cloud image/native imports still need
the corrected rerun. No model or GPU execution was performed locally.


## Cloud filesystem import correction — 2026-10-24

The corrected data pair passed cloud image creation, but native Ray Serve import
then failed on missing boto3 from its eagerly imported S3 provider. Pinned source
review confirms Azure/GCS delegate to PyArrow; no separate SDK imports are needed
there. Added boto3 explicitly and regenerated the serving-only lock, including
botocore/jmespath/s3transfer. No storage destination or credential changes.

Added a no-client filesystem-registry import helper/test. Moved the complete
model-free native check into the final Docker build RUN before expensive export;
post-build checks still prevent publication. Thus future import/config failures
stop earlier without exporting another image. Full cloud serving remains pending;
this failed invocation did not push an image or load a model. No Git operations.

Verification: real CPU registry imports passed with exact boto3/botocore 1.43.117
without clients, network calls or Torch. Combined suite: 111 tests, 109 passes and
two optional schema skips. Cloud native image/serving verification still pending.


## Native schema validation correction — 2026-10-24

The cloud build advanced through native imports and LLM configuration validation,
then the checker incorrectly called `ServeDeploySchema.model_validate`. Ray 2.53's
ServeDeploySchema inherits its Pydantic-v1 compatibility BaseModel, so that v2-only
method is unavailable despite Pydantic 2 being installed. Replaced it with native
constructor validation in a small helper. Dependency pins and the renderer are
unchanged. The upstream schema uses
[Ray's Pydantic compatibility base](https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/serve/schema.py).

A real CPU test with Ray 2.53 / Pydantic 2.13.5 validates the renderer's complete
Serve configuration, requires the native v1 schema instance, rejects an invalid
application and confirms no Torch/vLLM import. The test does not start Ray/Serve,
load a model or contact Kubernetes. Full cloud build/import/inference remain
pending; the failure occurred before image export/publication. No Git operations.

Verification: 112 combined CPU tests ran, 110 passed and two optional schema
checks skipped, including real Ray/Pydantic constructor validation in the isolated
CPU fixture. Diff checks passed; full cloud image/serving remains unverified.


## Latest-stable serving review input — 2026-10-25

User prefers the latest stable compatible serving dependencies. Added a separate
metadata-only `environment/latest-review.in` and a runnable local uv compile
command in the 7D runbook. It excludes pre-releases and writes to a separate
results/dependency_review candidate lock, leaving the production lock, published
fallback and training environments unchanged. The candidate has not been resolved
or accepted in this increment; review native APIs, CUDA wheels/drivers, base image
and adapter compatibility before another cloud build. No new VM needed yet.


## Latest-compatible serving implementation — 2026-10-25

Agent completed the local review; no user-run local dependency command is needed.
PyPI metadata selects stable Ray 2.59.0 with its exact vLLM 0.27.0 dependency,
Torch 2.13.0 and Transformers 5.20.2. Standalone latest vLLM 0.32/Torch 2.14.1
are not valid substitutes for those integration pins. The 236-package serving lock
is independent of the old training ABI; four OTel beta-only transitive releases
are explicit exceptions. A strict prerelease ban caused an unintended downgrade,
so reviewed core pins plus if-necessary-or-explicit are used instead.

New Docker recipe uses the official vLLM v0.27 amd64 digest with prebuilt core
version guards, root-only image dependency sync, real nonroot Ray identity and
no large recursive ownership changes. Model init uses system Python. New checks
cover explicit versions, CUDA 13 Torch, R580 host-driver baseline, shared completion
payload preservation, Ray's engine_cls alias, newer Hub API signatures and source
hashes including the shared benchmark. Actual full native/model/GPU checks remain
in cloud; none were run locally.

Real isolated CPU Ray 2.59 tests passed Serve schema validation/invalid rejection,
NumPy 2.3.5+pandas 3.1+Arrow 26 round-trip, filesystem registry imports with clients/
network blocked and Hub 2.3 staging API shape. An AST argument-shape audit against
pinned upstream Ray/vLLM source passed without importing/constructing an engine.
See documents/7D_DEPENDENCY_REVIEW.md and current runbook for identities and limits.
No changes to Parts 1–6 or accepted 7A–7C code/images/environments; the published
0.13 image/evidence remain retained. No cloud provisioning/build/push or Git ops.


## Verification completed for the new candidate

119 combined 7C/7D CPU checks passed, with no skips. This includes real pinned
KubeRay 1.4.2 RayJob/RayService CRD validation, actual Ray 2.59/Pydantic schema
checks, the selected NumPy/pandas/Arrow native round trip, filesystem imports with
clients/network blocked, Hub staging APIs and model-free orchestration tests.
The separate upstream AST audit checked one constructor call's argument shape.
Shell syntax, local preview commands, relative documentation links and whitespace
checks passed. These are not cloud image/native-GPU/inference measurements.

No user-run local review command is needed. New image build/import/publication is
the next cloud test, followed by E2E only if it passes; hardware/spending agreement
is still required before rental. No commits, pushes or new cloud resources.

Final recovery-proof refinement: require the old worker pod to be absent before
accepting a new ready UID plus successful inference; a terminating worker may
still serve requests. A regression case covers that overlap. Final combined
verification: 119 tests passed, no skips. No GPU/model execution or Git operations.


## Latest image retained; E2E handoff — 2026-10-25

The new serving image passed cloud build, complete model-free imports/config/
protocol checks, publication and registry pull verification. All five lightweight
files were independently checked at
`/home/ubuntu/results/7d/image_20261025T014429Z`; the dependency-lock hash matches
this checkout. Observed Ray 2.59.0/vLLM 0.27.0/Torch 2.13.0+cu130/Transformers
5.20.2 and CUDA runtime 13.0 match the intended core stack. Published image:
`sclion/part7-training@sha256:8ef0e9142d29a464c6a9b3eba17b1bfcba7311c52d08590a66200cb9019072dc`.

The user destroyed the build VM after retention. No further implementation change
is required before the first full E2E attempt. Next: user-run fresh E2E on an agreed
full x86_64 two-GPU cloud VM using that digest. Kubernetes setup/model staging/HTTP
inference/load/restart/cleanup are not yet cloud-verified for 7D. Do not rebuild or
republish merely to run E2E. No agent provisioning, Git staging/commit/push.
