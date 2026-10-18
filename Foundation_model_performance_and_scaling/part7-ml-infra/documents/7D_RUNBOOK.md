# 7D runbook — first source increment

**Do not rent a GPU VM for 7D yet.** The serving image/extras, lifecycle controller,
load/restart runner and one-command cloud E2E workflow are still being prepared.
Only the renderer is exposed by `part7D.sh` in this increment. See [7D plan](7D_PLAN.md).

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
