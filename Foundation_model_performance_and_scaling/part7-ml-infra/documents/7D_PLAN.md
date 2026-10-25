# 7D — RayService model serving

Updated: 2026-10-25. Implementation authorized; no instance remains provisioned.

## Current status

The old vLLM 0.13 serving image passed cloud build, model-free imports and Docker
Hub publication; all five evidence files were verified locally before VM release.
No model inference/HTTP/E2E pass has been recorded. That image remains a fallback.

The new latest-compatible candidate uses Ray 2.59 / vLLM 0.27 / Torch 2.13 and an
independent official vLLM base. Metadata/source review and real CPU schema/data/
filesystem/Hub checks passed. New GPU libraries and the derived image have not been
executed; first cloud build/import, then E2E remains required after hardware/budget
agreement. See [dependency review](7D_DEPENDENCY_REVIEW.md) and
[current commands](7D_RUNBOOK.md). 7D is not complete. 7A–7C remain preserved.

## Serving target and platform

- Use the existing 7A merged BF16 weights export, not the 7C FSDP-state dataset:
  `sclion/part7-7a-smoke-execution-20261005T045405Z-dwaxLH` at
  `40c33b6ad0a7a2879e7b5a8f5f6a52baaa94a584`. It was previously reloaded in 7A.
- Keep repo/revision, served ID, context length, sequence capacity and memory
  settings configurable. Default: Qwen2.5-Math-1.5B, one fixed replica on one
  whole GPU, TP=1, BF16, context 4096, max sequences 16, GPU memory fraction 0.5.
  Larger models/topologies need new compatibility/memory checks.
- Reuse the K3s/NVIDIA/KubeRay platform and pod security/storage conventions from
  7C. Start on one selected node with a CPU head and one 16-CPU/64-GiB one-GPU
  worker. The existing 7C bootstrap requires a two-GPU VM; one GPU is used by the
  initial service. No concurrent training or autoscaling is required.
- Pin Ray 2.59.0 and its exact vLLM 0.27.0 dependency, with Torch 2.13.0.
  A separate complete serving lock and official vLLM GPU base replace inheritance
  of the training ABI. The standalone newest vLLM 0.32 does not satisfy Ray's pin.
  Four OpenTelemetry beta transitive packages are documented exceptions. Never
  upgrade accepted 7A–7C environments/images in place.
- Both pod init containers stage the same exact Hub revision to PVC using a lock,
  resumable partial download and atomic rename. Record file hashes, tokenizer
  assets and complete safetensors inventory. Existing snapshots must pass identity
  and integrity checks. Main serving containers mount the workspace read-only
  and run with HF/Transformers offline; credentials are read only by staging.

## Implementation increments and acceptance

1. **CPU-prepared:** `config.py`, `render.py`, `prepare_model.py`, `part7D.sh` and
   snapshot/security/schema tests. Native `ray.serve.llm:build_openai_app` is the
   entry point; no copied inference loop. RayService `serveConfigV2` contains the
   model/engine settings, while `rayClusterConfig` contains Ray/Kubernetes settings.
2. **Next:** resolve serving extras, build an isolated digest-pinned image and run
   nonroot tool/API/import checks before publication. Include real user identity,
   writable cache/home and wget checks learned in 7C. Do not substitute a training
   image merely because its vLLM imports worked for verl.
3. Implement one normal cloud E2E entry point: install/check platform, stage pinned
   export, create RayService, wait for actual Serve/HTTP readiness, run the bounded
   load/restart test, retain evidence and verify owned-resource cleanup. Manual
   stage commands should be troubleshooting tools, not the normal workflow.
4. Benchmark native raw `/v1/completions` with prepared MATH/r1_zero prompts so no
   new chat-template semantics are introduced. Verify endpoint support against
   the selected Serve stack. Record warmup, concurrency, seeds, prompt/generated
   token counts, caps, stop strings, streaming and client placement.
5. Measure successful requests/s, generated tokens/s, streaming TTFT and per-token
   timing, p50/p95 end-to-end latency, errors and replica status. Use engine token
   accounting, not an assumed one-token-per-SSE-chunk rule. Capture native metrics/
   placement and distinguish GPU compute utilization from reserved cache memory.
6. Delete/restart only an owned serving worker pod, observe readiness loss and new
   pod/replica identity, then require successful post-recovery requests. Measure
   disruption/readiness recovery and disclose request failures; this is not VM or
   multi-node failover. Retain logs, model/image identities, manifest, metrics,
   benchmark report and cleanup evidence before marking 7D complete.

No measured serving throughput/latency/reliability/scalability claims yet. Full
controller-evidence backup is still absent; plan retention before node release.
Paid hardware/spending cap must be agreed before provisioning. No Git staging,
commits, pushes or registry uploads without the applicable user authorization.

## References

- [Pinned Ray LLM builder API](https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/serve/llm/__init__.py)
- [Pinned Ray dependency metadata](https://github.com/ray-project/ray/blob/ray-2.53.0/python/setup.py)
- [Pinned model-loading/engine config](https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/serve/core/configs/llm_config.py)
- [KubeRay RayService example](https://github.com/ray-project/kuberay/blob/v1.4.2/ray-operator/config/samples/ray-service.sample.yaml)
- [Pinned RayService schema](https://github.com/ray-project/kuberay/blob/v1.4.2/ray-operator/config/crd/bases/ray.io_rayservices.yaml)

See [runbook](7D_RUNBOOK.md) and [reading example](examples/7D_RAYSERVICE_EXAMPLE.yaml).


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

Validation for this increment: 80 combined 7C/7D CPU tests ran, 78 passed and
two optional CRD-schema tests skipped because their external schema fixtures
were not supplied. Shell syntax checks passed. No local GPU/model execution,
cloud deployment, publication or changes to accepted 7A–7C code.


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

Verification for E2E composition: 103 combined 7C/7D CPU tests ran; 101 passed
and two optional external CRD-schema checks skipped. CPU E2E preview, shell syntax
and diff-whitespace checks passed. No cloud, registry publication or model execution
was performed; Parts 1–6 and accepted 7A–7C implementation were preserved.


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
See the current stack and exact sequence in the [7D runbook](7D_RUNBOOK.md).
No provisioning, Docker build, image push, inference or Git commit/push occurred.

Validation for this increment: 107 combined 7C/7D CPU tests ran, 105 passed and
two optional CRD-schema checks skipped. Shell syntax and diff-whitespace checks
passed. The derived image syncs to the serving-only lock to avoid inherited
training-only dependency conflicts; actual image installation/import checks remain
cloud-pending. No local environment installation or model execution occurred.


## Cloud data-library failure and correction — 2026-10-24

The user-built cloud image completed but its pre-publication native import gate
failed: PyArrow 26.0.0 required NumPy >=2 while the candidate pinned 1.26.4.
No image push or inference followed. Added direct serving pins PyArrow 19.0.1 /
pandas 2.2.3 (the accepted 7C pair), regenerated the 220-package serving lock and
added a native in-memory NumPy/pandas/Arrow filesystem/Parquet check before image
export and inside the runtime receipt.

A separate temporary Python 3.12.3 CPU environment with these data packages passed
the real round trip without importing Torch/Ray or loading models. With that
fixture, 110 combined CPU tests ran: 108 passed and two optional CRD schema checks
skipped. This verifies this data-library pair, not the full cloud image/native
Ray/vLLM stack. Rerun build/import/publication on the existing VM after source
sync; preserve caches and stop/review any subsequent failure. Accepted 7A–7C code
and environments remain unchanged. No Git staging/commit/push performed.


## Cloud filesystem dependency correction — 2026-10-24

The next cloud image built and passed the data-library check but Ray Serve import
failed on missing boto3. Pinned filesystem source eagerly imports the S3 adapter;
Azure/GCS delegate to PyArrow without separate eager provider SDK imports. Added
boto3 and its resolved dependencies (boto3/botocore 1.43.117); the serving lock has
222 distinct packages / 223 requirement entries. Hugging Face/PVC storage and
credentials are unchanged; no AWS credentials or S3 bucket is required.

Moved the entire model-free native gate into the final Dockerfile RUN before
export, retaining the post-build/pre-push check. New helper imports the filesystem
registry without creating clients. In an isolated CPU environment, real Ray 2.53
filesystem imports passed with the exact boto3/botocore pins, mocked client/session
constructors and socket connections blocked, and no Torch import. With both native
CPU fixtures enabled, 111 tests ran: 109 passed, two optional schema checks skipped.
Shell syntax and diff checks passed. The full cloud Ray/vLLM image still needs its
corrected rerun; no native serving/inference success is claimed. No Git operations.


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


## Published image/import evidence — 2026-10-24

User-pasted cloud output records successful image creation, complete model-free
native import/config/constructor gate, Docker Hub push and digest pull verification:
`sclion/part7-training@sha256:172c13e7065a7414ea65e58347c01b858b51c612ec6b2686ddc4d69bdcec614a`.
Evidence directory: `results/7d/image_20261024T024858Z` (publication receipt, push log,
build log, runtime log and local image ID). Folder transfer/review is still pending.
The image is retained in the registry; no image/weights need laptop copying.

Observed versions: Ray 2.53.0, vLLM **0.13.0**, Torch 2.9.0+cu130, Transformers
4.57.6, NumPy 1.26.4, pandas 2.2.3, PyArrow 19.0.1, boto3/botocore 1.43.117.
The log explicitly reports no model loaded, GPU execution unverified and HTTP
unverified. This completes the observed build/import/publication step, not 7D.

Version discrepancy: the current serving lock selects `vllm[audio]==0.13.0`;
the GPU constraint file omitted a vLLM pin, contrary to earlier documentation's
0.12.0+cu130 expectation. This is not evidence of a stale user checkout. Reconcile
serving version/CUDA-wheel identity and documentation before GPU/E2E execution;
do not silently claim this image uses the training vLLM version. Preserve lightweight
evidence, then the build-only VM can be released; no further VM work is needed
solely to review this discrepancy. No implementation changes made in this review.


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
