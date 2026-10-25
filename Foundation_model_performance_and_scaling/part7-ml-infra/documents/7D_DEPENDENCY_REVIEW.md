# 7D serving dependency review — 2026-10-25

## Decision and scope

Use a separate latest-compatible stable serving candidate. Keep Parts 1–6 and the
accepted 7A–7C code, environments, images and artifacts unchanged. Keep the existing
published 7D vLLM 0.13 image as a retained fallback; the new candidate is not yet
built or GPU-verified. The same immutable 7A BF16 model export remains selected.

| Component | Reviewed candidate | Reason |
| --- | --- | --- |
| Ray Serve LLM | 2.59.0 | Latest stable Ray metadata at review time. |
| vLLM | 0.27.0 | Exact dependency of Ray 2.59's LLM extra. Standalone latest is 0.32.0, which does not satisfy that pin. |
| PyTorch | 2.13.0 | Required by vLLM 0.27. Standalone latest 2.14.1 is not selected. |
| Transformers | 5.20.2 | Resolved stable candidate satisfying vLLM's requirements; model execution remains unverified. |
| NumPy / pandas / PyArrow | 2.3.5 / 3.1.0 / 26.0.0 | Resolved together; real CPU filesystem/Parquet round trip passed. |
| Pydantic / Hub client | 2.14.1 / 2.3.0 | Real CPU Serve schema and model-staging API checks passed. |
| boto3 / botocore | 1.43.118 / 1.43.118 | Explicit filesystem-import support; no AWS clients/credentials required. |
| FlashInfer / CUTLASS DSL | 0.6.16.post3 / 4.6.0 | Resolved GPU dependencies, not locally installed or GPU-tested. |

Primary metadata: [Ray 2.59](https://pypi.org/pypi/ray/2.59.0/json),
[vLLM 0.27](https://pypi.org/pypi/vllm/0.27.0/json),
[PyTorch 2.13](https://pypi.org/pypi/torch/2.13.0/json).

The lock contains 236 distinct package names. The top-level framework versions are
stable releases. Four OpenTelemetry transitive packages are beta releases
(0.66b1): exporter-http-transport, exporter-otlp-common, exporter-prometheus and
semantic-conventions. These are recorded exceptions required by the integration.
A blanket `--prerelease disallow` made uv silently select Ray 2.46/vLLM 0.10;
explicit core pins plus `if-necessary-or-explicit` prevent that false upgrade.
Metadata resolution is not proof of native or GPU compatibility.

## Independent GPU base

Use the official vLLM v0.27.0 Linux/amd64 image, rather than layering new Torch
libraries over the training image:

```text
vllm/vllm-openai@sha256:5b03c7de2951cda2d5fd087457902625eabbe9188478e439561ef34f325b0fd5
```

Registry config was inspected without pulling layers or running Docker. It reports
CUDA_VERSION=13.0.2, an NVIDIA CUDA >=13.0 requirement and a root/default Python
installation. The derived image requires the existing Python 3.12 and its
prebuilt Torch/vLLM versions to match before syncing, installs only into the derived
image and provides a real UID/GID 1000/writable HOME for Ray. It clears the upstream
NCCL override and follows this project's complete serving lock; optional upstream
extras are trimmed. No DeepEP/MoE/TP compatibility claim is made. No recursive
ownership change is applied to large inherited library directories.

Sources: [official image recipe](https://github.com/vllm-project/vllm/blob/v0.27.0/docker/Dockerfile),
[official image tag](https://hub.docker.com/r/vllm/vllm-openai/tags?name=v0.27.0).
The recipe's source defaults and the already-built image's config need not be
identical; the digest/config above identify the actual selected artifact.

Cloud host baseline is Linux x86_64, systemd, Docker/NVIDIA runtime and driver
R580 or newer. NVIDIA documents R580 as the CUDA 13 family baseline, with PTX/JIT
and library-feature limitations; it does not guarantee every newer component works
on every R580 driver. The lock includes some CUDA 13.4 components, so actual GPU
execution remains required. Host toolkit version need not match the image toolkit.
See [NVIDIA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The imported Torch build must report CUDA major 13; full identities are retained.

## Adapter/check changes

- Explicit core pins and native version checks reject unexpected Ray/vLLM/Torch/
  Transformers versions; CUDA local-version suffixes remain visible in receipts.
- Ray 2.59 calls `engine_cls` after assigning `AsyncLLM`. Signature checking now
  recognizes this explicit alias and still rejects unknown/dynamic calls. Reviewed
  upstream source matches the vLLM constructor's argument shape, without importing
  or creating the engine locally. Conditional KV-aware wrappers are not a tested
  runtime claim.
- The real native Serve schema constructor validates the rendered configuration
  and rejects malformed applications in a CPU-only environment.
- Hub staging checks cover model-info/snapshot signatures, revision and token
  properties without network/model access. The init command now uses the serving
  image's system Python; it does not reference a training virtual environment.
- Benchmark and native request validation share one payload constructor. Round-trip
  checks require all requested fields while allowing additional framework defaults,
  so stream-usage/stop semantics cannot silently disappear.
- Complete model-free native checks still run before image export and publication.
  The completion schema and full Ray/vLLM imports require the cloud image; they
  were not executed locally with GPU libraries.

Reviewed sources: [Ray engine](https://github.com/ray-project/ray/blob/ray-2.59.0/python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py),
[vLLM engine](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/engine/async_llm.py).
Small metadata/source-hash audit evidence is retained at
`results/7d/dependency_review/latest_20261025`. No checkpoints or image layers were
loaded locally. Latest production lock was generated from requirements.in;
latest-review.in is a maintainer reproduction input, not a user setup step.

## Remaining cloud evidence

After hardware/spending agreement and source delivery, run the normal image build/
import/publication command. Only if it passes, run the complete RayService E2E
workflow using the newly published digest. Native imports, dense-Qwen inference,
HTTP/load metrics, same-node worker recovery and cleanup must be reviewed before
accepting this candidate. Model quality, multi-node and larger-model claims remain
outside this test. See [7D runbook](7D_RUNBOOK.md).


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
