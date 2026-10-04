# Part 5 modernization and Part 7 infrastructure plan

Updated: 2026-10-03

## Objective and scope

Refresh the existing Part 5 implementation incrementally, using relevant ideas from CS336 Spring 2026 while preserving the Qwen2.5-Math-1.5B / MATH experiments and historical results. Work in approximately one-hour sessions, with useful stopping points.

The shorter revised plan intentionally reduces the initial commitment. It does not reject the original technical roadmap: estimator variants, GSPO, pruning, broader instrumentation, and a controlled experiment remain available as later stages. Complete baseline cleanup and the rollout backend first, then decide whether to continue.

Stages 1–3 are complete within the bounded Part 5 implementation/validation
scope. GSPO hyperparameter tuning is deferred. Stage 4 will be implemented as
separate **Part 7 — ML Infrastructure**, in `../part7-ml-infrastructure/`:
7A distributed verl/FSDP2/vLLM workload; 7B Ray operations, observability and
recovery; 7C KubeRay RayJob training; 7D Ray Serve/vLLM through RayService.
7E Ray Train/FSDP2 SFT is optional. Stage 4 implementation is unstarted;
this revision updates the plan only and preserves Parts 1–6.

Part 6 remains deferred because the 2025 and 2026 supplements retain the same core instruction-SFT and DPO workflow. This plan does not require reproducing the full 2026 assignment or changing the model/dataset.

Execution constraint: model loading, inference, GPU smoke runs, and full training run on cloud machines such as Vast.ai, not on the user's local GPU. Local verification is limited to static checks and lightweight CPU tests without loading model checkpoints. Prepare commands for cloud validation and mark GPU checks pending until cloud results are available.

## Sources reviewed

- 2025 requirements: `old_requirements/cs336_spring2025_assignment5_alignment.pdf`
- 2025 supplement: `old_requirements/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`
- 2026 requirements and starter: `/mnt/d/Repos/Foundation_model_performance_and_scaling/2026_spring/assignment5-alignment/`
- Current local Part 5 implementation, including `section7_grpo/train_grpo.py`, its helpers, the shared SFT evaluation helper, scripts, tests, and README.
- [Official vLLM weight-transfer documentation](https://docs.vllm.ai/en/stable/training/weight_transfer/). Use documentation matching the selected dependency version when implementing.

## Stage 1: Baseline cleanup

Estimated active effort: 2–3 hours. No historical retraining required.

- [x] Capture the actual cloud baseline environment and model configuration through `run_config.json`. Cloud records were downloaded and reviewed locally; package versions, model configuration and resolved device/batch settings are confirmed.
- [x] Run a short cloud GPU smoke test. Both cloud smoke runs passed on 2026-10-01 UTC, as reported in the user's console output. Local CPU verification: all 14 existing GRPO tests and 6 new regression checks passed.
- [x] Implement a saved fixed evaluation subset for future runs, using an evaluation RNG independent of training sampling. Periodic/final evaluation share it; the cloud artifact was verified and its indices reproduced locally.
- [x] Make `skip_eval` consistently disable periodic and final evaluation.
- [x] Decouple inference GPU selection from `skip_eval`: rollouts still need vLLM when evaluation is disabled.
- [x] Clarify README/script wording: the historical no-std ablation is “GRPO without group-std normalization,” rather than the complete 2026 Dr. GRPO configuration.
- [x] Document that historical accuracy curves used resampled evaluation subsets; preserve their numbers and artifacts.
- [x] Correct partial accumulation scaling by actual example counts, preserving evenly divisible configurations. Verify gradient equivalence with a CPU full-batch reference.

Primary files: `cs336_alignment/section7_grpo/train_grpo.py`, `cs336_alignment/section7_grpo/helpers.py`, `cs336_alignment/section7_grpo/part_5_7.sh`, and `README.md`.

Completion criterion: a recorded, repeatable baseline with consistent evaluation semantics and accurate labels.

Implementation status (2026-09-30): Stage 1 code and documentation are ready for cloud validation. New logs and checkpoints use unique run directories; historical artifacts were not rewritten. GPU errors now abort instead of silently dropping microbatches and accumulated gradients. Python syntax, shell syntax and tracked-file whitespace checks passed. No local checkpoint loading, inference or GPU training was performed; no new accuracy result is claimed. Stage 2 implementation has since started; see its status below.

Cloud validation update (2026-10-01 UTC): user-provided console output reports `PASS` for both `stage1_fixed_eval` and `stage1_skip_eval` in `validation_pdQvedPa`. The script checks three periodic evaluation records plus final evaluation for the enabled run, no evaluation records/final file for the disabled run, resolved device placement, configuration/subset artifacts, and saved checkpoint files. The displayed disabled run completed three optimizer updates with finite losses and gradient norms and returned to the shell. A PyTorch NCCL process-group cleanup warning appeared on shutdown; it did not block completion and remains a lifecycle concern for Stage 2. Full JSON artifacts and both console logs have now been reviewed locally. Actual environment: Python 3.12.14, Torch 2.5.1 / CUDA 12.4, vLLM 0.7.2, Transformers 4.51.3, FlashAttention 2.7.4.post1 and W&B 0.22.3 on two A100-SXM4 40 GB GPUs. Source, dependency, prompt and dataset fingerprints match local files. The 32 evaluation indices were reproduced from seed 12345 over the matching 5000-record validation dataset. Final metrics match the last periodic evaluation. The fixed-evaluation run had zero reward/gradient at steps 2–3, consistent with zero group-centered advantages; the skip-evaluation run had finite nonzero gradient norms at every step. Neither console log has a traceback. The downloaded folder contains results/configs/console logs; smoke checkpoints remain separate cloud paths and were not inspected locally. Stage 1 cloud execution validation has passed; no accuracy benchmark claim is made.

W&B compatibility update: `pyproject.toml` and `uv.lock` now pin W&B 0.22.3. Normal cloud initialization installs support for longer API keys without manual SDK patches. The legacy GPU dependency versions remain unchanged; cloud authentication validation is pending because smoke runs disable W&B.

## Stage 2: Modern rollout backend

Estimated active effort: 5–8 hours, with additional allowance for dependency compatibility and GPU debugging. The combined 8–11-hour estimate for Stages 1–2 is a planning estimate, not a deadline.

- [x] Select and pin a compatible Python/Torch/CUDA/vLLM environment. The local 2026 starter's Python 3.12 and vLLM 0.19.1 are candidates, not an instruction to copy hardware-specific wheels blindly.
- [x] Add a dedicated rollout module, `cs336_alignment/section7_grpo/rollout_backend.py`, for generation, server lifecycle, and synchronization.
- [x] Implement the owned vLLM server subprocess on the inference GPU (cloud smoke verified).
- [x] Replace CPU policy-weight transfer and deep vLLM internal access with NCCL weight synchronization.
- [x] Pause generation during updates and invalidate caches derived from old weights.
- [x] Preserve completion ordering, question/response grouping, sampling settings, and stop behavior.
- [x] Retain generated token IDs and finish reasons for alignment checks and truncation metrics.
- [x] Adapt evaluation integration: `section4_sft/helpers.py::log_generations()` currently expects in-process vLLM results. Keep existing SFT/EI callers working if shared interfaces change.
- [x] Add startup timeouts, failure handling, and clean shutdown of processes owned by the run.
- [x] Add synchronization/generation timing and environment recording. Smoke timings were measured and reviewed; no comparative speedup is claimed.
- [x] Verify changed output weights/restoration, prefix-cache reset requests, response alignment and clean shutdown in two cloud smoke runs. Numerical correctness after changes to cached hidden states was not independently tested.

Completion criterion: a usable, documented modern backend with a verified two-GPU smoke run and basic timing evidence.

Implementation update: the separate `server_environment/` manifest and lock select
Python 3.12, Torch 2.10.0/cu129, vLLM 0.19.1, Transformers 4.57.6 and W&B 0.22.3.
The trainer initially uses SDPA, avoiding legacy FlashAttention wheels. `.venv`
and the legacy default backend remain available. The server adapter preserves
shared SFT/EI evaluation callers; modern GRPO trains on native completion IDs,
including EOS/stop tokens, and checks prompt-token alignment. Explicit modern
rollout/evaluation seeds and attention/token choices are recorded, so historical
numerical comparability is not assumed. A cloud-only controlled weight/cache
probe restores the policy before training. Setup and two-run cloud smoke scripts
are documented in `CLOUD_RUNBOOK.md`. Local verification passed: 35 targeted CPU tests (14 existing GRPO, six
Stage 1 regressions and 15 Stage 2 checks), Python/shell syntax, whitespace and
offline dependency-lock consistency. Tests use mock transports and tiny CPU
tensors; no model checkpoints were loaded or GPU work run locally.
Implementation checkboxes describe code, not GPU verification; Stage 2 cloud results have now been reviewed; implementation and execution validation are complete.

**Stop point:** Stages 1–2 are a complete, worthwhile upgrade. Reassess interest, time, GPU cost, and remaining issues before starting Stage 3.

## Stage 3: Unified estimator configuration and GSPO (complete)

Suggested order: preserve old behavior through the configuration refactor, add GSPO, then run one comparison. No commitment to every estimator variant is required.

- [x] Separate reward scoring, advantage construction, policy-gradient loss, and loss aggregation.
- [x] Introduce independent configuration options:

| Option | Values |
|---|---|
| Baseline | `mean`, `none` |
| Advantage normalization | `std`, `none`, `mean` |
| Importance weighting | `none`, `noclip`, `grpo`, `gspo` |
| Loss normalization | `sequence`, `constant` |

- [x] Preserve old CLI settings as compatibility aliases, with resolved options recorded for each run.
- [x] Verify loss/gradient equivalence for existing configurations on CPU (both normalization modes; half-precision ratio arithmetic is an explicit precision change).
- [x] Add GSPO sequence-level importance weighting and clipping.
- [x] Check response masking, unequal lengths, positive/negative advantages, clipping, and gradients on CPU. GSPO's geometric-mean weighting already introduces length normalization; combining it with constant normalization needs a separate mathematical check.
- [x] Run a bounded GRPO-Clip versus GSPO comparison using the same model, data, prompt, rollout reuse, fixed evaluation subset, and training budget.
- [x] Use matched seeds. Treat a single-seed run as a pilot; repeat promising results before claiming superiority.

Record accuracy/reward, response length, entropy, gradient norm, clearly defined clip fractions, synchronization and rollout times. Use the same modern backend for the GRPO/GSPO comparison. Any separate legacy-versus-modern comparison must hold the algorithm constant and account for dependency/token-handling differences; historical numbers are not the Stage 3 control.

Completion criterion: one controlled comparison, reproducible configurations, and an honest account of uncertainty.

Implementation update (2026-10-01): `estimators.py` resolves the four knobs
and separates scoring/advantages/surrogates/aggregation; legacy flags remain
aliases. GSPO uses a response-masked geometric-mean ratio and sequence clipping.
Constant normalization explicitly scales its sequence objective by length/C;
the planned first comparison uses sequence normalization. FP32 ratio arithmetic
for half-precision inputs is recorded here as a precision change; historical
runs are not assumed bitwise equivalent. Zero-advantage pruning is deferred.
`stage3_cloud_compare.sh` prepares matched smoke and 20-step pilot runs with
rollout reuse; `compare_runs.py` validates settings/artifacts and writes a
summary. 59 targeted CPU checks passed, including all 23 estimator checks and the existing GRPO/Stage 1/Stage 2 suites. All five comparison-validator tests also passed (64 targeted CPU checks total). Shell/Python syntax and whitespace checks passed. No checkpoint loading or GPU inference/training was run locally. The original cloud smoke/pilot subsequently passed and artifacts were reviewed
on 2026-10-03. The numerical follow-up and corrected pilot subsequently passed
cloud validation; see the retained validation notes below.

## Stage 4: Part 7 — ML Infrastructure (planned, implementation unstarted)

Create `../part7-ml-infrastructure/` when implementation is authorized. Stage 4
is the roadmap phase; Part 7 is its separate repository project. Keep Parts 1–6
as reference implementations, with only necessary cross-links after Part 7 has
reviewable results. Do not modify Part 5 training to accommodate the framework.

Shared platform: **Kubernetes → KubeRay → Ray**, with sibling workload paths:

| Workload | Framework/engine | Kubernetes execution |
|---|---|---|
| RL post-training | verl + FSDP2 + vLLM | RayJob |
| Online inference | Ray Serve LLM + vLLM | RayService |
| Optional supervised training | Ray Train + PyTorch/Hugging Face FSDP2 | RayJob |

Ray Train does not wrap verl. Reuse framework coordination and synchronization;
write only necessary adapters/configuration. Shared infrastructure does not
require one dependency image: training and serving may have different pinned
images. Prefer existing helpers and cohesive modules; create files/directories
only when their responsibility is needed. README presents capabilities, measured
results and interpretation caveats; run instructions go in a runbook.

### 7A: Framework workload — verl + FSDP2 + vLLM

- [ ] Select and pin a compatible verl/Ray/PyTorch/vLLM/FSDP2 release combination
  in an isolated environment or image. Preserve existing Part 5 environments.
- [ ] Reuse the Part 5 reward grader and prompt through thin adapters without
  importing its legacy dependency environment or copying its training loop.
  Convert MATH data to the framework format and retain input fingerprints.
- [ ] Map baseline/advantage statistics, clipping, loss reduction, optimizer,
  KL/entropy terms, prompt/tokenizer/EOS handling, sampling seeds and evaluation
  protocol explicitly. Check prompt counts versus response counts, global versus
  per-rank batches, rollout reuse and total optimizer updates in the pinned code.
  Document differences instead of assuming flag-name equivalence.
- [ ] Run lightweight CPU data/reward/config/loss checks, then a short cloud GPU
  workload that generates responses, updates, evaluates and saves checkpoints.
- [ ] Demonstrate at least two actual training ranks using FSDP2. Record world
  size, rank/device placement, logical/local parameter shapes and sharding
  placements; verify and document which training states are sharded. A config
  flag or world-size-one run does not prove distributed sharding. Single-node
  evidence does not establish multi-node scalability. Verify actual rollout
  colocation/resource needs before selecting GPUs; do not assume Part 5's
  dedicated trainer/inference split maps directly to verl.
- [ ] From the first cloud run, record resolved config, image/package/model
  identifiers, worker/GPU placement, job status, rollout/training throughput,
  memory measurements, checkpoint events and relevant logs.

Reference: corrected GRPO-Clip in `tight_pilot_FRqpow9W` (54/128 accuracy,
clipping 0.2/0.2, FP32 log-softmax, stable microbatches). Preserve Qwen2.5-Math-1.5B
and MATH. Framework mapping/numerical checks are required; matching accuracy or
bitwise GPU outputs across frameworks is not an acceptance criterion. GSPO
clipping/reuse tuning remains deferred and does not block this milestone.

**Stop point 7A:** one reproducible framework run with demonstrated two-rank
sharding, retained artifacts and a concise mapping/limitations note. A bounded
comparison may follow; no historical experiment suite retraining is required.

### 7B: Distributed operations — placement, observability and recovery

- [ ] Expose actor/worker topology, GPU/resource requests, placement and job
  status. Explain training/rollout GPU sharing and measured memory behavior;
  fractional Ray GPU requests do not isolate GPU memory.
- [ ] Preserve checkpoints, configs, logs and metrics outside ephemeral instance
  storage and verify readability after job exit. S3/object storage is an option,
  not a requirement; verify the chosen release's storage integration.
- [ ] Measure checkpoint duration/events, restart events and recovery time.
  Inject one controlled interruption and demonstrate checkpoint-based resumption.
- [ ] Verify what resume restores: weights, optimizer, scheduler where applicable,
  global step, data/progress position and RNG. Document unsupported or unverified
  state; distinguish a weights-only restart from a training-state resume.
- [ ] Establish basic monitoring using built-in Ray metrics/logs first. Add a
  small Prometheus/Grafana deployment where useful, with retained measurements.
  W&B remains optional tracking; smoke checks require no login. Do not introduce
  another experiment tracker solely to duplicate metrics.

**Stop point 7B:** recorded resource placement, durable artifacts and a tested
recovery procedure with explicit state-restoration limits.

### 7C: Kubernetes training — Docker + KubeRay RayJob

Prefer this before Kubernetes serving, but verify cloud access and cost first.
A standard Vast.ai GPU container is not automatically a Kubernetes GPU node.
Confirm host privileges, GPU/device-plugin access, networking, shared memory and
storage. Set a spending cap before provisioning; managed Kubernetes, multi-node
clusters and autoscaling are not initial requirements.

- [ ] Build a reproducible training image and pin its digest. Make GPUs available
  through the NVIDIA device plugin or a suitable GPU operator.
- [ ] Define a RayJob/RayCluster with a CPU head where feasible, GPU workers,
  explicit CPU/RAM/GPU requests/limits, sufficient shared memory, and placements
  compatible with the verified 7A topology.
- [ ] Configure durable data/checkpoint/artifact storage and environment/Secret
  injection; never store credentials in source or container images.
- [ ] Run one bounded GPU training/evaluation job; preserve manifests, job status,
  placement, metrics/logs and checkpoints after completion/resource cleanup.
- [ ] Exercise interruption/recovery on this platform and verify the documented
  resume behavior. Provide runnable cloud commands and measured evidence.

**Stop point 7C:** a real GPU RayJob with reproducible deployment, durable
artifacts and validated recovery. CPU-only manifest checks are preparation,
not evidence that GPU deployment worked.

### 7D: Serving — Ray Serve LLM + vLLM + RayService

- [ ] Export/merge a 7A checkpoint into a vLLM-compatible model artifact and verify
  tokenizer, model identifiers and inference behavior. Serve this checkpoint
  rather than introducing an unrelated model/benchmark.
- [ ] Configure Ray Serve LLM and deploy through KubeRay RayService with explicit
  resources, readiness checks and durable model access. Training is a batch
  RayJob; serving is a long-running RayService. They need not run concurrently
  on a small GPU allocation.
- [ ] Measure requests/sec, generated tokens/sec, TTFT, output-token timing,
  p50/p95 end-to-end latency, errors and replica status. Record input/output
  lengths, concurrency, warmup, streaming mode, hardware and measurement scope.
  Distinguish engine metrics from client-visible latency and GPU memory/cache
  utilization from GPU compute utilization; add an appropriate exporter if needed.
- [ ] Reuse basic monitoring; preserve a small load-test report and useful
  Prometheus/Grafana views rather than building a separate monitoring project.
- [ ] Demonstrate service restart/readiness recovery and owned-resource cleanup.

If suitable Kubernetes infrastructure is temporarily unavailable or too costly,
develop and measure the same Ray Serve service on a **cloud GPU machine** first,
then deploy through RayService later. Never run model serving on the user's local
GPU. This fallback does not complete the Kubernetes-serving acceptance criterion.

**Stop point 7D:** exported-checkpoint serving through RayService with measured
load behavior, retained operational evidence and demonstrated service recovery.

### 7E (optional): Reuse the platform for SFT

- [ ] Run one small Ray Train + PyTorch/Hugging Face FSDP2 SFT workload using the
  same operational conventions for containers, placement, storage and monitoring.
- [ ] Verify two-rank sharding and recovery rather than repeat the existing SFT
  research experiments. Record the minimum workload-specific adaptations needed.

Purpose: demonstrate platform reuse beyond RL. This is not required to complete
core Part 7 and does not require retraining the Part 6 8B models.

### Completion, boundaries and implementation authorization

Core Part 7 is **7A–7D**: actual distributed framework training, observable Ray
operations, durable artifacts and recovery, GPU RayJob deployment, and measured
RayService inference. Resume and monitoring claims must match retained evidence.
This is project experience, not production ownership or frontier-scale operation.
Multi-node scalability, RDMA/InfiniBand, autoscaling and production reliability
remain unclaimed unless independently implemented and measured.

Megatron, DeepSpeed, SGLang, continued pretraining, Kubeflow, Airflow, MLflow and
OpenRLHF are deferred extensions, not required technologies. Stop expanding the
roadmap; reassess after 7A and 7B in roughly one-hour work sessions.

This plan update authorizes documentation only. Part 7 files/environments,
implementation and paid provisioning have not started. Begin 7A after explicit
implementation authorization; select suitable hardware and budget before rentals.

Implementation references (use documentation matching the pinned versions):
- [verl GRPO configuration](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [verl training engines](https://verl.readthedocs.io/en/latest/workers/engine_workers.html)
- [RayJob](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html)
- [RayService](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayservice-quick-start.html)
- [Ray Serve LLM observability](https://docs.ray.io/en/latest/serve/llm/user-guides/observability.html)
- [Ray Train](https://docs.ray.io/en/latest/train/train.html)


## Deferred items from the original roadmap

These remain useful options, rather than required work for the initial upgrade.

### Additional estimator presets

| Preset | Baseline | Advantage normalization | Loss normalization |
|---|---|---|---|
| Standard GRPO | mean | std | sequence |
| GRPO-constant | mean | std | constant |
| Dr. GRPO | mean | none | constant |
| RFT | none | none | constant |
| Assignment MaxRL variant | mean | mean | constant |

The assignment's constant-normalized MaxRL variant differs from the original paper's normalization. Label it accordingly. Existing primitives cover much of these presets, but matching behavior and normalization must be verified before claiming implementation.

### Zero-advantage pruning

Implement only after accumulation and normalization are well defined.

- Compute rewards and group statistics before filtering.
- Preserve response/old-log-probability alignment and the original objective denominator.
- Handle all-zero-advantage batches and incomplete accumulation groups.
- Verify gradient equivalence between pruned and unpruned versions.
- Measure whether saved forward/backward work produces useful end-to-end speedup.

Simply averaging over retained responses changes gradient scale. If auxiliary objectives are added later, check whether zero policy-gradient advantage still permits skipping those responses.

### Broader performance instrumentation

Add scoring and training timings, generated tokens/second, peak memory, truncation counts, skipped-batch counts, and pruning statistics if needed to investigate a concrete bottleneck. Basic sync/rollout timing is enough for Stage 2.

### Part 6 maintenance

Consider dependency compatibility or measured DPO scoring bottlenecks after Part 5 is finished. No new model, full retraining, or replacement of DPO is required by this plan.

## Session cadence and documentation

- Aim for one reviewable change or verification result per one-hour session.
- End each stage with working commands, tested configuration, results, and known limitations.
- Preserve historical experiment artifacts and distinguish their evaluation protocol from new runs.
- Keep GPU runtime and debugging allowance separate from active implementation estimates.
- The original 4–6-week estimate describes the broader roadmap; it is not a commitment. Reassess after each useful stopping point.

Stage 1 is complete: code, CPU checks, cloud smoke execution and downloaded artifact review passed. Stage 2 implementation, cloud execution and local artifact review are complete. Stage 3 corrected cloud smoke and 20-step exploratory pilot artifacts have been reviewed; the frozen-scoring correction passed validation. W&B authentication was not exercised; checkpoint existence was checked on cloud, but checkpoint contents were not inspected locally.


Stage 2 cloud debugging update (2026-10-01): the first cloud attempt reached
server startup/NCCL initialization but stopped in the controlled probe because
it incorrectly required untied embeddings. The probe now selects an output row
whose token is absent from the fixed prompt, preserving the prompt hidden states
when input/output embeddings are tied. All 16 backend CPU checks passed, including tied and untied
models plus restoration after failure. The subsequent cloud rerun passed; this initial traceback alone did not
verify weight transfer or cache correctness.


Stage 2 completion (2026-10-01 UTC): both runs in `validation_tyNT5Avp`
passed on two A100-SXM4 40 GB GPUs. Downloaded artifacts were checked locally
under `results/section7/stage2_cloud/`:
controlled tied-embedding output-row change/restoration, 48 rollout records per
run with prompt/token alignment, three fixed-subset evaluations plus final
versus none for skip-eval, finite gradients, successful NCCL/cache-reset events,
and both server exits with code 0. Python 3.12.14, Torch 2.10.0+cu129 / CUDA 12.9,
vLLM 0.19.1, Transformers 4.57.6, W&B 0.22.3, no standalone flash-attn.
Per-step rollout sync: 0.20–0.28 s; generation: 3.61–3.86 s (16 responses,
256-token cap, eager inference). These are execution/timing checks, not accuracy
or speedup benchmarks. Cache-reset requests were checked; the output-row probe
does not independently prove correctness of changed cached hidden states.
Checkpoints were not downloaded, W&B login was not tested, and the instance is
destroyed. Historical results remain preserved; no full historical rerun is
required. Stage 3 was subsequently authorized; see its current checklist above.


Artifact organization update (2026-10-03 UTC): copied Stage 2 and Stage 3
folders were moved from the 2026 starter into Part 5 `results/section7/`
(`stage2_cloud/` and `stage3_cloud/`). All 65 file hashes were verified unchanged.


Stage 3 diagnostic audit (2026-10-03 UTC): copied pilot artifacts confirm
identical initial 64-response rollouts and identical first logged optimizer
loss/gradient norm (0.1010 / 0.249). Subsequent GRPO/GSPO trajectories diverge;
final accuracy is 48/128 vs 38/128. Core GSPO masked sequence-ratio and clipped
surrogate match equations (5)/(7) of the paper. The equal-epsilon pilot used
0.2 for both methods; paper section 5.1 uses GSPO lower/upper epsilon
3e-4/4e-4 versus GRPO 0.2/0.27. Equal numeric bounds are not equivalent policy
constraints, and the paper values are not automatically optimal for Qwen1.5B.
Pilot native-response truncation fractions were 42.7% (GRPO) and 31.7% (GSPO),
so the 256-token cap materially limits this experiment. A shared numerical
weakness remains: log-softmax is computed in the model logits dtype, normally
BF16, before ratios are promoted to FP32. Promoting afterwards does not recover
lost log-probability precision. Improve/check log-prob precision and instrument
old/current policy agreement before testing much tighter GSPO bounds; no causal
claim about the pilot gap follows from these observations. No hyperparameters
or training code were changed during this audit; instance is destroyed.
Reference: https://arxiv.org/html/2507.18071v2 (sections 4.1 and 5.1).

Additional audit verification: all 24 estimator CPU tests passed, including
an independent analytic check that GRPO and GSPO have identical on-policy
gradients for unequal response lengths. This supports the core loss, not a
claim that every GPU integration/numerical issue is excluded.


Stage 3 numerical follow-up (2026-10-03): training old/current log-softmax
now defaults to FP32; `--log_prob_precision model` preserves historical BF16
normalization. Shared SFT/EI helper defaults remain unchanged. Model weights
and forwards remain BF16; FP32 normalization can increase GPU memory use.
`system_metrics.jsonl` records per-microbatch policy agreement before each
rollout's first optimizer update (masked token/sequence log ratios, dtype,
counts). These are frozen HuggingFace comparisons, not vLLM scoring parity.
Inspect deviations against proposed clipping bounds; no universal tolerance
or cause of the historical pilot gap is asserted.

Optional `--gspo-tight` uses GSPO lower/upper epsilon 0.0003/0.0004 with
GRPO 0.2. This is an exploratory clipping change, not equivalent trust
regions or an optimal Qwen setting. Other comparison settings remain checked.
Existing outputs are preserved; new GPU validation is pending. Cloud commands
after publishing/pulling source and running the Stage 2 setup on a fresh instance:

```bash
bash cs336_alignment/section7_grpo/stage3_cloud_compare.sh
# Inspect policy_agreement_before_first_update in the new system_metrics.jsonl.
# Optional tighter-clipping smoke, then a 20-step exploratory comparison:
bash cs336_alignment/section7_grpo/stage3_cloud_compare.sh --gspo-tight
bash cs336_alignment/section7_grpo/stage3_cloud_compare.sh --pilot --gspo-tight
```

No dependency changes or W&B login are required for these runners. Preserve
the newly printed results folder before destroying the cloud instance.

- [x] Implement FP32 training log-prob normalization and frozen-policy diagnostics.
- [x] Add opt-in asymmetric GSPO clipping experiment without changing old artifacts.
- [ ] Validate the numerical follow-up on cloud and review agreement against tight bounds.

Numerical follow-up local validation: all 34 precision/estimator/comparison
CPU checks passed; Python compilation, runner shell syntax and source whitespace
checks passed. The broader suite also passed: 70 total CPU checks covering precision,
estimators, comparison validation, legacy GRPO, Stage 1 and Stage 2.

FP32 cloud artifact review (2026-10-03): `smoke_sztI4jia` contains both
complete three-step runs with matched configs, sources and fixed subsets;
old/current log probabilities are FP32 and both server exits are clean (0).
Both final accuracies are 3/32 (execution smoke, not a benchmark). Frozen-policy
sequence-ratio deviations before the first update range from 0.00305 to 0.00714;
maximum token log-prob differences reach 0.1946. These exceed the proposed GSPO
3e-4/4e-4 clipping bounds substantially. FP32 normalization is cloud-verified,
but frozen-policy agreement is not satisfactory for that tight experiment.
Defer tighter clipping and investigate forward/batching/mode numerical parity;
no cause is yet established, and original pilot artifacts remain unchanged.


Scoring investigation (2026-10-03): Qwen cloud config uses BF16, SDPA,
attention_dropout=0 and no gradient checkpointing. Old scoring is eval/no-grad
with original microbatches; training is train/grad with shuffled microbatches
and variable padded widths. First-step diagnostics contain both exact matches
and nonzero differences; a simple global old-policy change is not established.
PyTorch documents that mathematically equivalent batch/slice computations need
not be numerically identical, but this does not prove the source of our gap:
https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html

A cloud-only scoring probe now scans the first rollout's shuffled microbatches
without updates and selects the largest sequence-ratio discrepancy. It compares
original versus shuffled layouts, repeated identical forwards, gradient tracking,
train/eval mode, and wider right padding; it repeats those controls with an FP32
model forward. FP32 model comparisons use an FP32 original-layout reference,
not the BF16 old policy. This temporary diagnostic changes no algorithm or
training defaults; model mode/dtype, registered buffers and RNG are restored.
The entropy scalar logging warning is fixed using detached entropy.

After committing/pushing local source updates and pulling on a fresh two-GPU
cloud instance, restore the usual MATH data and run from the Part 5 root:

```bash
bash cs336_alignment/section7_grpo/stage2_cloud_setup.sh
bash cs336_alignment/section7_grpo/stage3_cloud_compare.sh --parity-only
```

This runs one rollout and scoring forwards, no optimizer updates, evaluations
or checkpoint saves. It uses the existing environment, disables W&B, and writes
a new `results/section7/stage3_cloud/parity_*` folder. Copy that entire folder
locally before destroying the instance. `PASS` means the probe ran and the
server stopped, not that numerical parity passed. Return `scoring_parity_probe.json`,
`run_config.json` and console logs for diagnosis. Tighter clipping and another
20-step pilot remain deferred. No model forwards were performed locally.

Investigation validation: all 74 CPU regression checks passed, including four
new synthetic tests that separate layout/mode/gradient effects, select the worst
microbatch, and verify restoration on success/failure. Python compilation,
runner shell syntax and source whitespace checks passed. The new cloud probe
remains pending; no numerical fix or explanation of the pilot gap is claimed.

Structure review: consolidated scoring probes into existing policy_diagnostics.py,
removed scoring_parity.py, and combined their tests in test_policy_precision.py.
Reusable rollout-probe preparation now lives with the diagnostics instead of
inside train_grpo.py. CLI commands and artifact schemas are unchanged. Future
module additions require a clear responsibility after checking existing helpers.
Structure validation: all 75 CPU checks passed, including native-token alignment
through the reusable probe preparation helper. Python/shell syntax and source
whitespace checks passed. Cloud probe validation remains pending.

Cloud scoring probe review: `parity_yXRX1YHa` completed with zero optimizer
updates, no final evaluation, and clean server exit (0). Original-layout
rescoring exactly matches frozen old values. For selected responses [14,4],
changing the microbatch layout produces a BF16 sequence-ratio deviation of
0.007141836 (0.71418%); with FP32 model forwards this falls to 1.143626e-6
(0.00011436%). Repeated forwards, train/eval and grad/no-grad controls are
exact matches in both precisions; adding eight padding positions also gives
zero difference for this batch. The tested discrepancy is isolated to BF16
forward sensitivity to the changed layout; this is not proof that it caused
the original pilot accuracy gap. Recommended next correction: retain stable
microbatch composition and padded shape across frozen/current scoring and
rollout reuse, then verify with a cloud smoke before tighter clipping. FP32
model training is an alternative with greater memory/compute cost. No README
update or historical artifact changes were made.


Stable scoring correction: modern server training defaults to
`--microbatch_layout stable`, preserving original response order, companions
and padded width within each microbatch across precomputation and epochs.
Whole microbatches are shuffled, including a short final group; accumulation
uses the actual example count. Actual optimizer-batch sizes are logged.
Legacy backend defaults remain `reshuffle`; `--microbatch_layout reshuffle`
and `--log_prob_precision model` reproduce the old batching/normalization choices.
This changes optimizer grouping and training RNG consumption for modern runs,
so the historical pilot is not a before/after control. Both estimators use the
same resolved layout in new comparisons. A shared guard aborts before the
first update if frozen token-log differences exceed 1e-4 or sequence-ratio
deviations exceed 1e-5. These are engineering tolerances for this workload,
not a general guarantee. The parity-only investigation retains reshuffled
layout to reproduce the observed issue. No new source files or README updates.

After committing/pushing and pulling the updated source on a two-GPU cloud
instance, restore the usual MATH data and run from Part 5:

```bash
# Fresh instance only:
bash cs336_alignment/section7_grpo/stage2_cloud_setup.sh
# Three-step matched comparison with stable scoring:
bash cs336_alignment/section7_grpo/stage3_cloud_compare.sh
```

Copy the printed new smoke folder back and review agreement before choosing
tighter clipping or another pilot. Cloud verification of this correction is pending.
Stable-layout validation: all 81 CPU regression checks passed, including
shape-sensitive frozen scoring, partial-group gradient equivalence, legacy
permutation reproduction and agreement-guard rejection. Python/shell syntax
and source whitespace checks passed. Cloud correction verification is pending.

Stable-layout cloud validation: copied `smoke_uqt1CugC` contains both complete
three-step runs, matched sources/configs/fixed subsets, stable layout and FP32
log-softmax. Every recorded frozen-policy token-log difference and sequence-ratio
deviation is exactly zero across both runs/all three steps. Each epoch update
uses 16 examples; gradients are finite and both server exits are clean (0).
The previously observed frozen-scoring mismatch is resolved in this smoke.
Final accuracy is GRPO 1/32 and GSPO 2/32; these are execution checks, not
performance conclusions. A tighter-clipping smoke can now be attempted before
any optional 20-step experiment. Historical artifacts and README are unchanged.

Tighter-clipping cloud smoke review: `tight_smoke_e4DHuvQ2` contains matched
three-step runs except explicitly declared importance weighting and bounds.
Stable layout/FP32 log-softmax are confirmed; all frozen-policy differences
are exactly zero, gradients finite, and both server exits clean (0). GRPO uses
0.2/0.2; GSPO uses 0.0003/0.0004. Both final accuracies are 3/32. GSPO mean
sequence-outside fractions across both epochs are 0.46875, 0.40625, 0.46875;
mean active-clipping fractions are 0.125, 0.125, 0.09375. Second-epoch gradient
norms are 0, 0, 0.0693, consistent with tight clipping and sparse rewards.
Bounds violations include zero-advantage responses and are not the same as
active surrogate clipping. This passes execution validation, not hyperparameter
validation or algorithm superiority. An optional new 20-step exploratory run
can use --pilot --gspo-tight; it is not comparable as a numerical before/after
against the historical reshuffled/BF16-normalization pilot. README unchanged.

Corrected 20-step pilot artifact review: `tight_pilot_FRqpow9W` contains both
complete 20-step runs with 160 updates of 32 examples each, matching sources,
config/environment and fixed 128-example evaluation subset except the declared
importance weighting and clipping bounds. All frozen-policy differences are
exactly zero, gradients finite, and server exits clean (0). GRPO 0.2/0.2 final
accuracy is 54/128 (42.1875%); GSPO 0.0003/0.0004 is 49/128 (38.28125%).
GSPO has 67/160 zero-gradient updates versus GRPO 0/160; mean active clipping
is 19.75% versus 0.97%. Last-step entropy is 0.616 versus 0.481 nats. Overall
response truncation is 36.09% versus 30.86%. Different clipping bounds, one
seed and capped generations prevent an algorithm-superiority conclusion or
a causal comparison to the historical pilot. This validates the corrected
implementation and produces the planned bounded exploratory comparison;
further clipping/reuse tuning is optional, not required to fix the resolved
frozen-scoring mismatch. No README or historical artifact changes in this review.


Scope decision: Stage 3 is complete with the corrected implementation, CPU
regressions, cloud scoring agreement, and bounded exploratory pilot. Further
GSPO tuning is deferred. Stage 4 can proceed using corrected GRPO-Clip as the
reference; framework estimator/default mapping still requires validation.
README now presents the corrected measured results and concise limitations.
