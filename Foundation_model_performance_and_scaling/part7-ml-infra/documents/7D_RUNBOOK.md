# 7D runbook — serving workflow preparation

The source workflow and a separately resolved serving candidate are ready for a
first bounded cloud validation. **No 7D native import, inference or cloud E2E pass
has been recorded.** Agree VM hardware and total spending cap before rental.
Use the staged verification below; stop at the first failure and retain evidence.
See [7D plan](7D_PLAN.md) for acceptance limits.

## Current serving stack and first cloud validation

The serving image derives from the immutable accepted 7C image, but upgrades Ray
only in the derived image. Its environment is synced to the serving lock, removing
unused training distributions so they cannot conflict with serving dependencies.
Matching GPU packages are reused. 7A–7C code, dependencies and published image stay intact.

| Component | Serving candidate |
| --- | --- |
| Ray Serve LLM | 2.53.0 |
| Torch / vLLM | 2.9.0+cu130 / 0.12.0+cu130 |
| Transformers / NumPy | 4.57.6 / 1.26.4 |
| Flash Attention / CUTLASS DSL | 2.8.3 / 4.2.1 |
| PyArrow / pandas | 19.0.1 / 2.2.3 (NumPy 1.26-compatible candidate) |
| Dependency resolution | 220 packages; Python 3.12/Linux x86_64; metadata only |

Ray 2.53's official LLM extra targets vLLM >=0.12.0 and Transformers >=4.57.3.
Its public API uses common Ray deprecation utilities rather than importing RLlib;
its vLLM adapter handles different `init_app_state` signatures. These source checks
motivate the candidate; they do not prove runtime compatibility. References:
[dependency metadata](https://github.com/ray-project/ray/blob/ray-2.53.0/python/setup.py),
[public API](https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/serve/llm/__init__.py),
[vLLM adapter](https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py).

Local previews need no model, Docker or network:

```bash
bash infrastructure/7d_serve/part7D.sh image --check
bash infrastructure/7d_serve/part7D.sh e2e --check \
  --image registry.example/serve@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

After agreeing hardware/budget, clone/pull the approved source on a full cloud VM
and enter Part 7. The existing installer requires exactly two GPUs, at least 32
CPUs, 120 GiB host RAM and 200 GiB free disk; serving itself uses one GPU. Use a
full Ubuntu VM with systemd, passwordless sudo, Docker and NVIDIA container runtime,
not a restricted container rental. The previously reviewed 2 x H100 80 GB VM is
a reference hardware profile, not a current price/availability recommendation.

Git push/clone/pull remains the preferred source delivery method, performed only
with your approval. For uncommitted changes, first clone the full repository on
the VM (so Part 5 sources exist), then transfer the 7D directory from local Part 7:

```bash
rsync -az --exclude=__pycache__ --exclude='.env*' --exclude='.cache' \
  infrastructure/7d_serve/ \
  ubuntu@CLOUD_HOST:~/Foundation_model_performance_and_scaling/Foundation_model_performance_and_scaling/part7-ml-infra/infrastructure/7d_serve/
```

Replace the host/path to match the VM. This copies code/configuration only, without
committing, pushing, or transferring model files. Source hashes are checked again
when the resulting image is used.

First build/check the derived image and publish only if native import checks pass:

```bash
bash infrastructure/7d_serve/part7D.sh image --cloud --publish
```

Default destination is the existing `sclion/part7-training` public repository,
using a new `7d-<UTC timestamp>` tag. It does not overwrite a 7C tag or digest.
Use `--repository USER/REPO` to change the destination. Authentication reads
`DOCKERHUB_TOKEN` or prompts invisibly, using an isolated temporary Docker config
removed afterwards (including sudo-created files). No model or Hub token enters
the build context. Omit `--publish` for a build/import diagnostic without login.
Fresh evidence goes under `results/7d/image_*`; publication records the registry
digest and verifies it can be pulled. A passing import gate is not an inference pass.

If build/import checks pass, use the printed **serving** digest in the full run:

```bash
bash infrastructure/7d_serve/part7D.sh e2e --cloud --fresh-setup \
  --image sclion/part7-training@sha256:SERVING_DIGEST_FROM_PUBLICATION
```

The uppercase digest placeholder must be replaced; do not use a training digest.
For an existing accepted cluster, omit `--fresh-setup`. Fresh setup prompts for a
Hugging Face write token or reads `HF_TOKEN`, then mounts a Secret for model staging.
No manual model download, laptop data copy or per-phase commands are needed.
If native imports fail, stop before E2E and inspect `build/runtime_check.log`.
Do not release the VM until lightweight evidence is retained and reviewed.

## End-to-end workflow and code map

This is the main workflow reference for 7D. Like 7C, `e2e.py` connects existing
components on the cloud VM. A serving-image digest that has passed the native import gate is a
prerequisite; E2E does not build or publish the image.

```text
On the cloud VM (outside the serving pods)
part7D.sh e2e
  -> e2e.py: check serving image imports and source hashes in a Docker container
  -> install/check Kubernetes, KubeRay, PVC, Hub Secret and RayService API
  -> prepare the fixed 32 raw MATH prompts from cloned Part 5 sources
  -> render.py: generate the RayService manifest
  -> service_control.py: create it through the Kubernetes API
       -> KubeRay creates Ray head/worker pods from the serving image
       -> init containers run prepare_model.py: pinned Hub export -> shared PVC
       -> Ray Serve starts the native LLM application inside the Ray cluster
  -> wait for native Serve readiness and HTTP model catalog
  -> open VM-loopback port-forward; benchmark.py runs warmup + concurrent load
  -> retain pre-restart logs, metrics and placement
  -> delete the owned worker pod; KubeRay creates its replacement
  -> require a new pod UID, same node and successful inference
  -> benchmark again; retain recovery observations and post-restart evidence
  -> save passing review; delete owned RayService and verify descendant cleanup
  -> save overall e2e_review.json
```

| Responsibility | Main code |
| --- | --- |
| Shell entry point, image gate, prompts and overall composition | [part7D.sh](../infrastructure/7d_serve/part7D.sh), [e2e.py](../infrastructure/7d_serve/e2e.py) |
| Separate serving-image recipe and import diagnostics | [Dockerfile](../infrastructure/7d_serve/Dockerfile), [build_image.sh](../infrastructure/7d_serve/build_image.sh), [runtime_check.py](../infrastructure/7d_serve/runtime_check.py) |
| Configurable model identity/settings and RayService manifest | [config.py](../infrastructure/7d_serve/config.py), [render.py](../infrastructure/7d_serve/render.py) |
| Model download/integrity checks inside init containers | [prepare_model.py](../infrastructure/7d_serve/prepare_model.py) |
| VM-side readiness, transport, restart, evidence and cleanup | [service_control.py](../infrastructure/7d_serve/service_control.py) |
| VM-side HTTP load and measurements | [benchmark.py](../infrastructure/7d_serve/benchmark.py) |

Kubernetes/KubeRay/Serve status checks and actual HTTP generation checks serve
different purposes; a Running pod or a model-catalog response alone is insufficient.
The worker deletion is an intentional recovery test. KubeRay replaces the pod;
the controller observes readiness and inference, recording failures rather than
assuming there was an HTTP outage. Failures stop the workflow and retain resources
and diagnostic evidence. Successful E2E runs verify owned-resource cleanup;
PVCs, Secrets and the Kubernetes cluster remain.

**Current status:** this workflow is implemented and CPU-tested as source, but
native imports/inference and actual cloud E2E execution remain unverified.
The next step is bounded cloud validation after hardware/budget agreement. See [7D plan](7D_PLAN.md) for remaining gates and the
[7C workflow](7C_RUNBOOK.md#end-to-end-workflow-and-code-map) for the verified
training/recovery counterpart.

## CPU preview

From Part 7, generate a reading manifest with a placeholder image digest:

```bash
bash infrastructure/7d_serve/part7D.sh render \
  --image registry.example/part7-serving@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --node-name part7-gpu --output /tmp/part7-7d-rayservice.json
```

This writes JSON (valid Kubernetes YAML) and performs no Kubernetes/Hub/model
operations. The image is a placeholder, not the published 7C training image and
not deployable. Existing output is refused. The committed
[RayService example](examples/7D_RAYSERVICE_EXAMPLE.yaml) is for reading only.

Model/settings overrides: `--model-repo`, `--model-revision` (full 40-hex commit),
`--model-id`, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`.
Resource identities: `--namespace`, `--name`, `--pvc`, `--secret`, `--pull-secret`.
TP remains one for initial validation. Inspect `render --help` for all options.

## What deployment will do

The native RayService will use `ray.serve.llm:build_openai_app`, a local pinned
export path on PVC and an OpenAI-compatible endpoint. CPU head and GPU worker
init containers download/verify the same snapshot under a shared file lock;
weights are downloaded once, with later init pods checking hashes. A partial
download is retained and reused only for the same repo/revision. Staging reads
the mounted Hub Secret without CLI login or exported token values. Main serving
containers read the PVC offline and expose port 8000 via the internal head proxy.
No public ingress/LoadBalancer is configured. The controller/benchmark transport is implemented in
`service_control.py`; native runtime compatibility and cloud execution remain
unverified.

`prepare_model.py --cloud` is an init-container helper, not a local model command.
It requires Kubernetes pod identity, mounted Secret and a `/workspace` destination.
It validates config/tokenizer assets and complete safetensors shards, records
file hashes and atomically publishes the ready snapshot. It does not load tensors
or run inference. This path is CPU-tested with synthetic files only; actual
private export download/init scheduling still needs cloud verification.

## Next verification gates

- Resolve and lock the serving image, validate native Serve LLM configuration,
  imports, nonroot identity/cache paths, wget and endpoint requirements on cloud.
- Run the complete one-command setup/service/readiness/load/restart/cleanup test,
  retaining phase evidence and failure diagnostics. Do not claim a sequence of
  successful CPU mocks is an end-to-end cloud serving result.
- Review model/tokenizer identity, actual one-GPU allocation, request counts,
  token accounting/streaming metrics, old/new worker identities and cleanup.
- Preserve the accepted model/export and prior milestones; never retrain 7A or
  use the full FSDP-state dataset directly as a serving snapshot.

CPU schema checks use the pinned KubeRay 1.4.2 CRD via `KUBERAY_SERVICE_CRD` and
lightweight jsonschema/PyYAML dependencies outside the project environment. Eight
7D tests passed, including this schema check. No local checkpoint/model loading,
Docker build, cluster mutation or GPU execution has occurred.


## Historical serving dependency/import gate — 2026-10-20

The Ray 2.49.2 candidate described here was superseded by the Ray 2.53.0 stack
at the top of this runbook. These notes record the earlier increment.

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
been built or validated on cloud. Lifecycle, HTTP/load/restart helpers and the
single-command E2E runner are now composed; their actual cloud validation remains
pending. Do not rent a VM for this partial
workflow yet. Local tests cover signature mismatch detection with synthetic
functions; they are not native runtime or cloud E2E tests.


## HTTP benchmark helper (called by the E2E runner)

`part7D.sh benchmark --cloud --prompts PATH --output PATH` requires a running
service through a VM-local port-forward. Each JSONL row must have a fully rendered
raw `prompt` string. The E2E runner prepares these from the accepted
Part 5 sources and records their provenance, so manually preparing this file is not part of
the intended normal setup. Options include `--requests`, `--concurrency`,
`--warmup`, `--max-tokens`, `--timeout`, `--model-id` and `--base-url`.

The helper refuses output overwrite, requires a GPU-visible cloud host and saves
measurement failures before exiting unsuccessfully. No automatic retry hides
errors. Successful warmup is required. Client TTFT includes HTTP, queue/prefill
and transport latency. Output-token timing is an estimate based on the first/last
text events and engine token usage, not an engine inter-token latency measurement.
A zero-token response has no TTFT/output-token estimate. An endpoint without valid
streamed usage fails validation rather than substituting text chunk counts.
No throughput/latency results have been measured yet.


## Lifecycle helper (runs on the VM, outside serving pods)

`service_control.py` is called by the E2E runner, rather than requiring more
manual cloud commands. Its application sequence is:

```text
VM-side E2E controller
  -> validate/create RayService through Kubernetes API
  -> wait for KubeRay and native Serve readiness
  -> open loopback port-forward to the owned head pod
  -> verify model catalog + completion warmup
  -> benchmark concurrent HTTP requests
  -> retain pre-restart logs/status/placement
  -> delete the owned worker pod using its UID
  -> observe replacement UID, same node and working inference
  -> benchmark again + retain post-restart evidence
  -> save successful review
  -> optionally delete owned service and verify descendant cleanup
```

KubeRay replaces the worker pod; this code requests the intentional deletion and
observes the result. It does not restart the VM or restore a training checkpoint.
The serving model remains the pinned 7A export on PVC. HTTP requests may continue
or fail during replacement; reports retain the observed probe results. A successful
model-catalog response alone does not establish working inference.

Failure leaves service resources available for diagnosis. Evidence includes init
logs, bounded current/previous container logs, pod status/placement, service status,
benchmark reports and recovery/cleanup receipts. Log capture is capped at 10,000
lines per container/version and does not archive model files, Secrets or pod
environments. Full evidence backup/retention remains part of final E2E setup.

Status field mapping follows the pinned
[KubeRay 1.4.2 RayService API](https://github.com/ray-project/kuberay/blob/v1.4.2/ray-operator/apis/ray/v1/rayservice_types.go).
The UID deletion body uses the
[kubectl 1.32 raw-delete path](https://github.com/kubernetes/kubectl/blob/v0.32.0/pkg/cmd/delete/delete.go).
These source checks and synthetic tests do not establish cloud completion.


## E2E composition preview

The complete source workflow is now composed in `e2e.py`. Check it locally with a
placeholder digest (this does not download the image or load a model):

```bash
bash infrastructure/7d_serve/part7D.sh e2e --check \
  --image registry.example/serve@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

Once a compatible serving image has been built/published and hardware/budget are
agreed, the intended normal VM command is `part7D.sh e2e --cloud --fresh-setup
--image SERVING_IMAGE_DIGEST`. This sequence requires prior hardware/budget agreement and a successful
image build/import gate; actual native imports and inference remain unverified.
The placeholder and the training image must not be substituted as verified serving
images. `--fresh-setup` refuses reuse; omit it for an existing accepted cluster.

The source workflow checks the image before installing Kubernetes, then uses the
accepted installer and hidden Hugging Face Secret setup. It prepares prompts from
the Git-cloned Part 5 files on the VM, without laptop rsync. Model init containers
stage the immutable private export onto PVC. The lifecycle controller measures
HTTP serving and same-node worker replacement, retains evidence and verifies
owned-resource cleanup. Source hashes prevent a stale image silently running a
different pod-used configuration/staging implementation. VM/controller code need
not be baked into the image to run, but model-staging code must match.

Retain the complete lightweight run directory and review `e2e_review.json`,
`service/review.json`, both benchmarks, recovery observations and cleanup receipt
before releasing the VM. No model weights are copied back in this workflow.
The 32 fixed prompts provide repeatable load inputs; the benchmark is not a
32-example accuracy evaluation. Current load settings use 16 initial measured
requests and four post-recovery requests, with request indices/prompt hashes saved.


## Build troubleshooting: uv sync command syntax

If the dependency layer exits with `unexpected argument '-r' found`, use the
fixed Dockerfile: `uv pip sync` takes the lockfile as a positional source argument,
whereas `uv pip install` supports `-r`. This was a command-generation mistake,
not a dependency resolution failure. The source now uses:

```bash
uv pip sync --python /opt/part7/.venv-7a/bin/python \
  infrastructure/7d_serve/environment/requirements.lock
```

Update the VM checkout (approved Git delivery or the rsync alternative above),
then rerun `part7D.sh image --cloud --publish` there. Do not clear Docker caches:
previously downloaded/extracted base layers may be reused on the same builder.
No serving image was uploaded by the failed command because upload follows a
successful build/import check. Actual dependency installation and native imports
remain unverified until the corrected cloud run succeeds.

A prompt such as `ubuntu@TGlenovo7pro:/mnt/d/Repos/...` indicates the command may
have run in local WSL rather than over SSH on the cloud VM. Both the Python host
guard and direct build helper now reject known WSL kernels before Docker work.
`--cloud` expresses execution intent; it cannot establish cloud provenance by itself.


## Native import failure: PyArrow requires NumPy 2

The first cloud build completed but failed its native import gate: PyArrow 26.0.0
required NumPy >=2 at runtime, while the candidate retained NumPy 1.26.4. Package
metadata resolution and `uv pip check` did not catch that binary/import requirement.
This is a serving dependency issue, unrelated to whether execution is on cloud.
The failed invocation stopped before image publication; no inference was attempted.

The serving input/lock now pin PyArrow 19.0.1 and pandas 2.2.3, reusing the native
data-library pair from accepted 7C. The image runs a three-row NumPy/pandas/Arrow
filesystem/Parquet interoperability check before exporting layers. The full native
Ray/vLLM import gate still runs afterwards, before any push. A passing data check
alone does not establish serving compatibility or inference success.

Update the source on the existing cloud VM and rerun:

```bash
bash infrastructure/7d_serve/part7D.sh image --cloud --publish
```

Keep Docker/BuildKit caches. The same builder can reuse the base and unchanged
packages; changed dependency layers still need rebuilding. Review the new result
before starting E2E. No Kubernetes reinstall is needed for this image-import failure.
