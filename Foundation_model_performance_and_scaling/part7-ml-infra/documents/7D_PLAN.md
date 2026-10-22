# 7D — RayService model serving

Updated: 2026-10-17. Implementation authorized; no instance remains provisioned.

## Current status

First source increment is CPU-prepared: immutable model configuration, native Ray
Serve LLM configuration, RayService renderer and locked/verified PVC snapshot
staging. Eight new CPU tests pass, including the real KubeRay 1.4.2 RayService CRD.
No model checkpoint was downloaded/loaded locally. No serving image, Kubernetes
service or GPU inference has been executed for 7D. **7D is not cloud-ready yet.**
7A–7C and their environments/artifacts remain preserved.

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
- Candidate Ray 2.49.2 retains the verified cluster pin. Its declared LLM extra
  accepts vLLM >=0.10, but that lower bound does not prove API compatibility with
  our vLLM 0.12.0+cu130. Serving extras must be resolved and import-tested in a
  separate serving image before deployment. `environment/requirements.in` is only
  candidate input, not a lock. Choose a separately pinned Ray version if needed;
  never upgrade the accepted 7A–7C images/environments in place.
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

- [Pinned Ray LLM builder API](https://github.com/ray-project/ray/blob/ray-2.49.2/python/ray/serve/llm/__init__.py)
- [Pinned Ray dependency metadata](https://github.com/ray-project/ray/blob/ray-2.49.2/python/setup.py)
- [Pinned model-loading/engine config](https://github.com/ray-project/ray/blob/ray-2.49.2/python/ray/llm/_internal/serve/configs/server_models.py)
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
