# 7D runbook — serving workflow preparation

**Do not rent a GPU VM for 7D yet.** The candidate serving image/native API
compatibility is still under review. The E2E source workflow is now composed,
but has not been run on cloud. The lifecycle/load/restart helpers now have CPU tests, but have
not run against a real service.
The renderer, candidate image diagnostic, benchmark helper and composed E2E runner
are exposed by `part7D.sh`. Native image compatibility and cloud E2E remain unverified. See [7D plan](7D_PLAN.md).

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
No public ingress/LoadBalancer is configured. The controller/benchmark transport
must be implemented before the normal cloud command is runnable.

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


## HTTP benchmark helper (for the pending E2E runner)

`part7D.sh benchmark --cloud --prompts PATH --output PATH` requires a running
service through a VM-local port-forward. Each JSONL row must have a fully rendered
raw `prompt` string. The future E2E runner will prepare these from the accepted
bundle and record its provenance, so manually preparing this file is not part of
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

`service_control.py` will be called by the E2E runner, rather than requiring more
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
--image SERVING_IMAGE_DIGEST`. This is not a recommendation to rent/run yet: the
current candidate dependency stack has not passed native imports or inference.
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
