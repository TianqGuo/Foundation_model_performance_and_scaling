# 7D runbook — serving workflow preparation

The latest-compatible serving candidate is prepared and CPU-tested. The earlier
vLLM 0.13 image passed cloud build/import/publication and is retained as a fallback;
**the new candidate has not been built, imported with GPU libraries or served on
cloud yet**. No user-run local dependency review is needed. The agent completed it;
see [dependency review](7D_DEPENDENCY_REVIEW.md) for versions, sources and limits.

## Current candidate and cloud sequence

| Component | Candidate |
| --- | --- |
| Ray Serve LLM | 2.59.0 (latest stable Ray at review time) |
| vLLM | 0.27.0 (Ray's exact supported dependency; standalone latest is 0.32.0) |
| Torch / Transformers | 2.13.0 / 5.20.2 |
| NumPy / pandas / PyArrow | 2.3.5 / 3.1.0 / 26.0.0 |
| Pydantic / Hub client | 2.14.1 / 2.3.0 |
| Base | Official vLLM v0.27.0 amd64 image, pinned by digest in Dockerfile/config.py |

This serving image is independent of the training image/GPU ABI. The official
base's Python 3.12 and prebuilt Torch/vLLM must match before dependency sync.
Only the derived image is changed. New core-version/CUDA checks, schema/protocol
checks and source hashes prevent silent drift. Parts 1–6 and accepted 7A–7C remain
preserved. The 7A BF16 export/revision is unchanged.

Agree hardware and total spending cap before renting. The VM must be full Linux
x86_64 with systemd, passwordless sudo, Docker/NVIDIA container runtime, visible
GPUs and NVIDIA driver R580 or newer (CUDA 13 baseline). Actual PTX/JIT/GPU support
still needs testing; the driver baseline alone is not a compatibility pass.
One GPU is enough for image build/import checks. The current reused Kubernetes
installer requires exactly two GPUs, 32 CPUs, 120 GiB RAM and 200 GiB free disk;
serving uses one GPU. A two-GPU VM can run both phases sequentially. No host CUDA
toolkit download or manual model download is part of the normal workflow.

After source publication/clone/pull under your control, enter Part 7 on the VM.
For the new candidate, first run:

```bash
bash infrastructure/7d_serve/part7D.sh image --cloud --publish
```

This builds, runs complete model-free native checks before export and again before
push, then publishes only on success. Default repository: `sclion/part7-training`,
new `7d-<UTC timestamp>` tag. A Docker Hub write token is read from DOCKERHUB_TOKEN
or prompted invisibly; temporary authentication is removed afterwards. No model
or Hub token enters the image/context. Omit --publish for a build/import diagnostic
without login. Record the printed digest and results/7d/image_* evidence.

Only if those checks pass, run the full E2E workflow using that **new** digest:

```bash
bash infrastructure/7d_serve/part7D.sh e2e --cloud --fresh-setup \
  --image sclion/part7-training@sha256:NEW_SERVING_DIGEST
```

Replace the uppercase digest placeholder. Do not substitute the retained 0.13 or
7C training digest; the version/source gate will reject it. Omit --fresh-setup on
an already accepted cluster; fresh mode refuses reuse and never resets a cluster.
The E2E runner handles setup/reuse, hidden Hub authentication, prompt preparation,
model staging, serving/HTTP load, restart/recovery, evidence and owned cleanup.
Stop at the first failure and retain its logs. Keep build caches.

Local previews are optional maintenance checks, not required user setup:

```bash
bash infrastructure/7d_serve/part7D.sh image --check
bash infrastructure/7d_serve/part7D.sh e2e --check \
  --image registry.example/serve@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

For uncommitted source, clone the full repository on the VM first so Part 5 data
exists, then transfer Part 7 source from local (replace host/path as needed):

```bash
rsync -az --exclude=__pycache__ --exclude='.env*' --exclude='.cache' \
  infrastructure workloads \
  ubuntu@CLOUD_HOST:~/Foundation_model_performance_and_scaling/Foundation_model_performance_and_scaling/part7-ml-infra/
```

This merges both source trees into the existing checkout without copying model
files. Git push/clone/pull remains the preferred delivery route, performed only
with your approval. Do not transfer credentials, models, environments or results.

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
  -> require old pod absence, a new pod UID, same node and successful inference
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
  -> require old pod absence, replacement UID, same node and working inference
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
data-library pair from accepted 7C. The image includes a three-row NumPy/pandas/Arrow
filesystem/Parquet interoperability check in the full native gate, which now runs
before image export and again before any push. A passing data check
alone does not establish serving compatibility or inference success.

Update the source on the existing cloud VM and rerun:

```bash
bash infrastructure/7d_serve/part7D.sh image --cloud --publish
```

Keep Docker/BuildKit caches. The same builder can reuse the base and unchanged
packages; changed dependency layers still need rebuilding. Review the new result
before starting E2E. No Kubernetes reinstall is needed for this image-import failure.


## Native import failure: missing boto3

The next cloud build passed the data-library check but failed importing Ray Serve:
Ray 2.53's cloud filesystem registry eagerly imports its S3 adapter, which imports
boto3/botocore even when our model source is Hugging Face and the serving path is
local PVC. This import requirement was absent from the resolved serving extra.
The serving input now explicitly includes boto3; its SDK dependencies are pinned
in the regenerated lock. No AWS credentials, S3 bucket or storage migration is needed.
Azure/GCS providers delegate to PyArrow in this pinned release; separate Azure/
Google SDKs are not eagerly imported on this path. Reviewed pinned source:
[registry](https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/common/utils/cloud_filesystem/__init__.py),
[S3 adapter](https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/common/utils/cloud_filesystem/s3_filesystem.py).

The Dockerfile now executes the **complete model-free native runtime check before
image export**, including data/SDK/provider imports, Ray Serve configuration and
vLLM imports/signatures. It does not start Ray, create cloud clients, load a model
or run inference. The post-build/pre-push check remains in place. Further import
failures should now stop at the build step without spending time exporting layers.
The serving lock has 222 distinct package names (223 requirement entries).

After syncing the corrected 7D source on the same VM, rerun the normal image
command. Keep the builder cache; no Kubernetes reinstall is needed. This build
also stopped before publication. Full native Ray/vLLM and inference verification
remain pending until the corrected cloud gates pass.


## Native check failure: ServeDeploySchema has no model_validate

This is a checker API error, not a missing package: Ray 2.53's Serve deployment
schema uses its Pydantic-v1 compatibility layer. The fixed helper validates using
`ServeDeploySchema(**configuration)`, with the native schema constructor handling
validation. There is no Pydantic downgrade or dependency-lock change. Real native
CPU coverage now accepts the generated Serve payload and rejects invalid apps.

The preceding Triton/no-CUDA notices occur inside Docker build, where this build
command does not expose GPUs. They are not the fatal exception in this log. GPU
execution remains a separate cloud serving/E2E check.

Sync the corrected source to the same VM and rerun the normal image command.
Keep the build cache; the unchanged dependency layer can be reused. The checker
still runs before export, and publication requires it to pass.


## Dependency metadata reproduction (maintainer reference)

The agent completed the review; no user-run local resolution is required.
Top-level framework versions are stable and pinned. Ray requires beta-only
OpenTelemetry components, so use the documented exception instead of a blanket
prerelease ban that silently selects older frameworks:

```bash
uv pip compile --prerelease if-necessary-or-explicit \
  --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
  infrastructure/7d_serve/environment/requirements.in \
  --output-file /tmp/part7-serving-review.lock
```

This is metadata only; it does not install packages or prove native/GPU support.
See [dependency review](7D_DEPENDENCY_REVIEW.md). Keep historical failure notes
below as history; the candidate table at the top states current versions.
