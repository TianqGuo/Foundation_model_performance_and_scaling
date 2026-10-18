# 7C — Kubernetes training

Updated: 2026-10-17 (PDT). Accepted run: 2026-10-18 UTC.

## Status: complete at the bounded same-node scope

Reviewed `results/7c/part7-e2e-20261018-031219`: fresh Kubernetes installation, GPU
preflight, actual two-rank FSDP2 training, controlled interruption, pinned Hub
checkpoint recovery and verified per-job cleanup all passed. The evidence was
synced locally and the dependency-light recovery reviewer was rerun successfully.
This completes 7C; 7D serving is next and has not been implemented or verified.

## Accepted configuration and results

| Item | Reviewed result |
| --- | --- |
| Hardware | One Lambda KVM Ubuntu 22.04 GPU VM; two H100 80 GB GPUs, 52 CPUs, about 442 GiB RAM. |
| Platform | K3s v1.32.13+k3s1, NVIDIA device plugin 0.17.1, KubeRay 1.4.2, Ray 2.49.2. |
| Workload | Qwen2.5-Math-1.5B/MATH; unchanged 7A pins, prompt/reward semantics, 1,023 training and 32 fixed evaluation prompts. |
| Placement | CPU head and one two-GPU worker on the same node; distinct worker pods and Ray runtimes across the three training phases. |
| Storage | Same-node 100 GiB RWO PVC; mounted Hub Secret; complete checkpoint and manifest retained in a private Hub dataset repository. |
| Baseline/interruption/resume | Three steps / controlled stop after step 2 / resume at step 3. Optimizer calls per rank: 6 / 4 / 2. |
| State audit | Model, Adam, scheduler, saved worker RNG and trainer step/data restoration passed; next source indices matched `[78, 536, 279, 461]`. |
| Continuation timing | About 13.39 s per rank from post-audit to first optimizer update; not total recovery time. |
| Lifecycle | Staging pod cleanup and all four RayJob/RayCluster/Ray-pod cleanup checks passed; PVC/Secret/node are outside job cleanup. |
| Local retention | 87 files, 1,292,168 bytes; no checkpoint tensor files. |

Recorded phase wall times: preflight 79.1 s, baseline 218.0 s, interruption 307.1 s,
resume 279.0 s. They include controller waiting/retention/cleanup and phase-specific
work; they are not isolated training or network benchmarks.

Training image:

```text
sclion/part7-training@sha256:b1d8240b348d2a036d311f19fe75784825613607129c1d93526cfa7f530fb8e8
```

Complete checkpoint dataset repository: `sclion/part7-7c-part7-e2e-20261018-031219-interrupt`.
Immutable revision: `122e784fddc0a315a65e5ad9b2ed932dcc41b117`. Upload receipts say Hub listing
verification alone is insufficient; the subsequent download hashes and native
state audits passed in this accepted run.

## Implementation map

| Files under `infrastructure/7c_kuberay/` | Responsibility |
| --- | --- |
| `e2e.py`, `part7C.sh` | One cloud entry point: setup/readiness, automatic data preparation, staging and complete recovery experiment. |
| `setup_cluster.py`, `cluster_check.py` | Guarded fresh-VM bootstrap and read-only cluster prerequisites. |
| `Dockerfile`, `Dockerfile.dockerignore`, `build_image.sh`, `image_publish.py` | Isolated dependencies/source image, cached builds, nonroot startup/import checks, explicit public Docker Hub publication and digest receipts. |
| `render.py` | Generate RayJob manifests containing Kubernetes pod settings, Ray configuration and launcher arguments. |
| `preflight.py` | Two concurrent GPU probes, CUDA/import and Hub authentication checks; no model load. |
| `run_7c.py` | Compose native verl phases, attach to existing Ray, prepare model and reuse 7B restore audits. |
| `checkpoint_transfer.py` | Private Hub full-state upload and pinned fresh-path download with manifest verification. |
| `data_staging.py`, `job_control.py` | Verified bundle staging, RayJob submission/status/log/evidence retention and bounded owned-resource cleanup. |
| `experiment.py` | Baseline/interruption/resume orchestration and cross-phase recovery review. |

The prepared bundle is generated on the VM from Git-tracked Part 5 sources;
laptop transfer is optional. Part 5 is mounted read-only and its five accepted
input hashes are enforced. Ray Train does not wrap verl. Native verl workers
provide training coordination and synchronization.

## Limits and 7D handoff

- Same Kubernetes GPU node only; no replacement-node, multi-node or automatic
  crash/reboot recovery claim. The interruption is at a complete checkpoint boundary.
- vLLM server RNG/cache/in-flight requests and driver-process RNG are not restored.
  No bitwise rollout/final-weight parity, accuracy improvement or scalability claim.
- RWO host-backed storage survives pods but not VM deletion. Full checkpoints and
  manifests are on HF; the controller receipt/lightweight logs need separate retention.
  Automatic controller-loss discovery and external backup of all evidence are absent.
- Image publication and CPU regression tests are separate from `e2e`. Latest local
  suite: 69 passed, one optional pinned-schema check skipped because its temporary
  dependencies/fixture were unavailable; the pinned schema passed earlier checks.
- For 7D, review the existing exported 7A model's pinned repository/revision,
  select a compatible serving stack and prepare a RayService/readiness/benchmark/
  restart-recovery plan. A full FSDP training-state dataset is not a serving export.
  Do not assume the training image supports the serving stack unchanged.

Core roadmap remains 7A–7D; optional 7E, TP/PP, Kueue and broader failure testing
remain later decisions. Preserve Parts 1–6. No commits/pushes without approval;
agree hardware/spending cap before paid provisioning.

See [runbook](7C_RUNBOOK.md), [manifest guide](MANIFEST_GUIDE.md),
[Q&A](QA.md) and [reading example](examples/7C_RAYJOB_EXAMPLE.yaml).

## Primary references

- [KubeRay 1.4.2](https://github.com/ray-project/kuberay/releases/tag/v1.4.2)
- [GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [K3s NVIDIA runtime](https://docs.k3s.io/advanced#nvidia-container-runtime)
- [uv image caching](https://docs.astral.sh/uv/guides/integration/docker/)
