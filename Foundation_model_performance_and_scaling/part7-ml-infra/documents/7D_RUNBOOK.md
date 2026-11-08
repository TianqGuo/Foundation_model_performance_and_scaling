# 7D runbook — serving and recovery

The bounded cloud run is complete; see [accepted results and limits](7D_PLAN.md).
This document owns reproduction, the workflow/code map and troubleshooting.
No new cloud run is required for closeout. The accepted final E2E reused Kubernetes;
a combined fresh-install run with the final image remains unverified.

## Reproduce full cloud E2E

Agree hardware and spending cap, then use a fresh full x86_64 Ubuntu GPU VM with
systemd, passwordless sudo, Docker/NVIDIA runtime, driver R580+, exactly two GPUs,
at least 32 CPUs, 120 GiB RAM and 200 GiB free disk. Serving uses one GPU; the
reused Kubernetes bootstrap requires two. Clone/pull the current approved source,
enter Part 7, and run:

```bash
bash infrastructure/7d_serve/part7D.sh e2e --cloud --fresh-setup \
  --image sclion/part7-training@sha256:7ccbadd1212957b1342cf3b8054dd134cb66d5103e71dfcfb7c646d7475ab737
```

This reuses the published image. No Docker Hub token, image build, local GPU run,
manual Kubernetes installation or manual model download is required. Setup prompts
invisibly for the Hub token if HF_TOKEN is absent. The runner prepares/checks the
platform and data, stages the pinned model, verifies serving/HTTP, benchmarks,
restarts the owned worker, checks recovery and retains evidence/cleanup receipts.
Keep the VM until evidence is copied and reviewed; stop on any failure.

## Accepted stack and reproduction sequence

| Component | Accepted version (reviewed 2026-10-25) |
| --- | --- |
| Ray Serve LLM | 2.59.0 (latest stable Ray at review time) |
| vLLM | 0.27.0 (Ray's exact supported dependency; standalone latest at that review was 0.32.0) |
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
For a future image rebuild only (not needed for the E2E command above), run:

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
Fresh setup creates the Hub Secret from a hidden prompt (or HF_TOKEN). Reuse
requires the existing Secret; it does not prompt again. The E2E runner handles
setup/reuse, prompt preparation,
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

## CPU preview and workload overrides

From Part 7, render a reading manifest without API/model access:

```bash
bash infrastructure/7d_serve/part7D.sh render \
  --image registry.example/serve@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --node-name part7-gpu --output /tmp/part7-7d-rayservice.json
```

The placeholder image is not deployable. The [reading example](examples/7D_RAYSERVICE_EXAMPLE.yaml)
is also illustrative. Model options: `--model-repo`, `--model-revision` (full commit),
`--model-id`, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`.
Renderer resource options: `--namespace`, `--name`, `--pvc`, `--secret`, `--pull-secret`.
Run the relevant subcommand with `--help`; initial validation fixes TP=1.

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
See [reviewed measurements and settings](7D_PLAN.md).


## Evidence retention and cleanup

E2E prints its fresh `results/7d/<run-name>` directory. Review `e2e_review.json`,
`service/review.json`, `service/recovery.json`, both benchmark reports and
`service/cleanup.json`. Check identities, request errors, old/new worker UIDs,
old-pod absence, same-node placement and log-capture error lists. A Running pod
or model-catalog response alone does not prove inference recovery.

Success deletes only the owned RayService and verifies descendant cleanup;
Kubernetes, PVC/model cache and Hub Secret remain. Failure preserves resources
and diagnostic evidence. Logs are capped at 10,000 lines per container/version;
models and Secret values are not archived. Preserve both E2E and image-build
folders before releasing the VM. From the **local Part 7 directory**, substitute
host/run names and copy lightweight evidence:

```bash
rsync -az --info=progress2 \
  ubuntu@CLOUD_HOST:~/Foundation_model_performance_and_scaling/Foundation_model_performance_and_scaling/part7-ml-infra/results/7d/RUN_NAME/ \
  ./results/7d/RUN_NAME/
```

Repeat for the `image_<timestamp>` folder. The loopback port-forward is managed
by the controller. Its TTFT/latency includes transport and queue/prefill; SSE text
events are not individual tokens. Retain recovery failures rather than claiming
uninterrupted availability. Training checkpoint state is not restored by serving;
the replacement loads the same immutable model snapshot.

## Troubleshooting

| Symptom | Cause and maintained fix |
| --- | --- |
| Model-init HTTP 401 | Token invalid or missing access to the private model. Replace the mounted `part7-huggingface` Secret; do not print tokens. Fresh setup reads HF_TOKEN before prompting. Reuse requires a pre-existing Secret. |
| Pending cluster lookup NotFound | KubeRay may report its name before creation. Controller retries only NotFound; other API and ownership errors still fail. |
| Worker unready with `Readiness probe failed: success` | `HeadOnly` has no worker HTTP proxy. Renderer explicitly checks Raylet health; native Serve/HTTP checks independently gate application readiness. |
| `uv pip sync` rejects `-r` | Sync takes the requirements file as a positional argument; use the maintained Dockerfile. |
| PyArrow/NumPy import error | Use the serving lock's compatible NumPy/pandas/PyArrow pair and real round-trip gate; do not mix training pins. |
| Missing boto3 during Ray import | Ray imports its cloud filesystem registry eagerly. Keep the locked SDK even when weights come from HF; no AWS credentials are required. |
| ServeDeploySchema lacks `model_validate` | Use its public constructor, compatible with Ray's Pydantic-v1 schema under Pydantic 2. |
| No CUDA driver/Triton during build import gate | That model-free gate hides GPUs. Only actual cloud E2E proves inference. |
| Source/version gate rejects an image | Match image to pod-used source and lock. Rebuild/publish when checked source changes; use the newly printed digest. |

For live diagnosis in a second **cloud terminal**:

```bash
export KUBECONFIG="$HOME/.kube/part7-config"
kubectl get rayservices,rayclusters,pods -n part7
kubectl get events -n part7 --sort-by=.metadata.creationTimestamp
kubectl logs -n part7 POD_NAME -c stage-model --previous --tail=100
kubectl logs -n part7 POD_NAME -c ray-worker --tail=100
```

Use `--previous` only when that container restarted. Keep the failed run's evidence.
Before rerunning, remove only the identified failed test RayService and wait for
its pods to disappear; do not accumulate competing services or reinstall K3s for
an application failure. Reuse mode omits `--fresh-setup`; fresh mode refuses an
existing installation and never resets it automatically.

For dependency metadata reproduction, see the dated
[dependency review](7D_DEPENDENCY_REVIEW.md). Review compatibility again before
changing dependencies, GPU type, model or topology.
