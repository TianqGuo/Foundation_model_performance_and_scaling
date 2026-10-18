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
