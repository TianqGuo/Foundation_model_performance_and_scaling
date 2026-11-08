# Part 7 — ML Infrastructure

Updated: 2026-11-08

**Core 7A–7D is complete at the bounded cloud scopes below.** The cloud instances
have been released after local evidence review. Optional work requires a separate
decision; no cloud run is required for documentation closeout.

This file records the current roadmap, acceptance and open decisions. Detailed
milestone results and operating procedures live in the linked documents, rather
than a growing sequence of development updates.

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
directories also include `7b_ray/`, `7c_kuberay/` and `7d_serve/`. Add an optional 7E directory only if selected; tests and results remain
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
| 7D — Serving | **Complete: bounded cloud serving and same-node recovery.** | Accepted run `part7-7d-20261025-044105`: pinned export, one GPU/replica, 16/16 initial and 4/4 post-recovery HTTP requests, streaming metrics, new worker UID/inference recovery in 225.6 s, reviewed retained evidence and owned cleanup. Final run reused Kubernetes; combined fresh-install run with final image remains unverified. See [7D results](documents/7D_PLAN.md). |

Record serving hardware, input/output lengths, concurrency, warmup, streaming and
client/measurement scope. Distinguish engine metrics from client latency and GPU
memory/cache allocation from compute utilization. Training and serving can run
sequentially and use separate pinned images. Fractional Ray GPU requests do not
isolate GPU memory. CPU/schema checks alone never establish cloud completion.

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

## Documentation ownership

| Document | Responsibility |
| --- | --- |
| Repository README and [Part 7 README](README.md) | Short capabilities, representative measurements and scope; link to detail. |
| [AGENTS.md](AGENTS.md) | Working conventions, authorization and preservation rules. |
| `PLAN.md` | Current milestone status, acceptance limits and optional decisions. |
| [7A environment](documents/7A_ENVIRONMENT.md), [algorithm mapping](documents/7A_GRPO_MAPPING.md), [runbook](documents/7A_RUNBOOK.md) | Training stack, workload semantics and execution/results. |
| [7B plan](documents/7B_PLAN.md), [runbook](documents/7B_RUNBOOK.md) | Native checkpoint/recovery evidence and procedure. |
| [7C plan](documents/7C_PLAN.md), [runbook](documents/7C_RUNBOOK.md) | Kubernetes training evidence and workflow. |
| [7D plan](documents/7D_PLAN.md), [runbook](documents/7D_RUNBOOK.md) | Serving acceptance/results and execution/troubleshooting. |
| [7D dependency review](documents/7D_DEPENDENCY_REVIEW.md) | Dated dependency decisions and compatibility rationale. |
| [QA.md](documents/QA.md), [manifest guide](documents/MANIFEST_GUIDE.md) | Concepts, explanations and diagrams; link to commands/results. |
| [Part 5 plan copy](documents/PART5_MODERNIZATION_PLAN.md) | Historical reference; not current Part 7 status. |

Update each fact in its owning document. Summarize and link elsewhere instead of
copying long results or instructions. Keep the existing plan/runbook pair per
milestone; add a new file only for a distinct responsibility. At closeout, replace
obsolete progress notes with current decisions, evidence and useful troubleshooting.
Preserve a compact account of major changes and their reasons in milestone plans,
including failed approaches that explain the final design. Source history and
retained run artifacts supply the detailed development record.
Keep the Q&A graphs; merge related questions and link to current operational docs.

## Verified boundaries and next decisions

- 7A is a short infrastructure smoke, not a model-quality improvement claim.
- 7B restores model/Adam/scheduler/saved worker RNG and trainer step/data state.
  Driver RNG and vLLM server RNG/cache/in-flight requests are outside recovery.
- 7C proves fresh Kubernetes setup and same-node two-rank recovery in distinct
  RayJob pods/runtimes. Accepted evidence: `results/7c/part7-e2e-20261018-031219`.
- 7D proves one replica/GPU serving and same-node worker replacement with HTTP
  recovery. Accepted evidence: `results/7d/part7-7d-20261025-044105` and
  `results/7d/image_20261025T043048Z`. Fresh setup passed earlier; the final
  corrected image passed E2E in reuse mode. A combined fresh-install run with that
  image remains unverified; this is a validation follow-up, not a new core stage.
- No replacement-node/VM recovery, bitwise rollout equivalence, multi-node
  scalability, RDMA/InfiniBand, autoscaling or production-reliability claim.
- Keep model identity/tokenizer/prompt/resource settings configurable, but test
  compatibility, exported weights and memory for each new model/topology.
- Agree hardware and spending cap before provisioning. All model/GPU execution
  stays on cloud; local checks remain static/lightweight CPU. No Git staging,
  commits or pushes without approval. Retain evidence before resource release.

Megatron, DeepSpeed, SGLang, continued pretraining, Kubeflow, Airflow, MLflow and
OpenRLHF remain deferred interests, not committed implementation work.
