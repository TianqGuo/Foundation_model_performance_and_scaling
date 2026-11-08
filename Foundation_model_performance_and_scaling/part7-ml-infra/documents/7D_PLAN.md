# 7D — RayService model serving

Updated: 2026-11-08. Bounded cloud scope complete; instance released.

## Current status

7D is complete at its bounded single-replica/GPU, same-node worker-recovery
scope. Accepted cloud run `part7-7d-20261025-044105` served the pinned 7A BF16
export, passed 16/16 initial and 4/4 post-recovery HTTP requests, replaced the
worker with a new UID, restored inference in 225.6 s and verified owned cleanup.
All 35 E2E evidence files and the image-publication receipt were reviewed locally;
the user subsequently released the VM. The final run reused Kubernetes. Fresh
installation passed earlier, but a single fresh-install-to-finish run with the
final corrected image remains unverified and does not block this bounded scope.
Core 7A–7D is complete; 7E, shared-pool/Kueue, larger-model/multi-node work and
broader recovery are optional follow-ups requiring a separate decision.

## Measured 7D serving and recovery

Qwen2.5-Math-1.5B, pinned 7A BF16 export; one GPU, one replica, TP=1,
context limit 4096, maximum 16 sequences and GPU memory fraction 0.5.
Ray 2.59.0 / vLLM 0.27.0 / Torch 2.13.0+cu130 served raw MATH prompts.
The initial streaming test used 16 requests, concurrency 4, two warmup requests
and at most 128 output tokens: 16/16 succeeded, 354.8 generated tokens/s,
60.3 ms median client TTFT and 1.34/1.36 s p50/p95 request latency.
After intentional worker deletion, a new worker on the same node restored
inference in 225.6 s; all four post-recovery requests succeeded. Owned cleanup
passed and lightweight evidence was retained before VM release.

These are small client measurements through a VM-local port-forward; they do not
establish model quality, engine inter-token latency or multi-node performance.
The successful final run reused Kubernetes; a single fresh-install-to-finish run
with the final image remains unverified. See [7D runbook](7D_RUNBOOK.md).

Accepted image: `sclion/part7-training@sha256:7ccbadd1212957b1342cf3b8054dd134cb66d5103e71dfcfb7c646d7475ab737`.

Evidence: `results/7d/part7-7d-20261025-044105` and
`results/7d/image_20261025T043048Z` (ignored artifacts retained locally).
The run receipts preserve their original pre-review wording; the reviewed status
is recorded here. The development summary below explains the main changes.

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
  of the training ABI. At the October 25 review, standalone vLLM 0.32 did not satisfy Ray's pin.
  Four OpenTelemetry beta transitive packages are documented exceptions. Never
  upgrade accepted 7A–7C environments/images in place.
- Both pod init containers stage the same exact Hub revision to PVC using a lock,
  resumable partial download and atomic rename. Record file hashes, tokenizer
  assets and complete safetensors inventory. Existing snapshots must pass identity
  and integrity checks. Main serving containers mount the workspace read-only
  and run with HF/Transformers offline; credentials are read only by staging.

## Acceptance evidence

| Check | Reviewed evidence |
| --- | --- |
| Image and native integration | Published digest above; version/lock/source checks, native imports and request-schema validation passed. |
| Model identity | Pinned merged 7A BF16 export; shared-PVC staging/integrity checks and offline serving passed. |
| HTTP/load | 16 requests at concurrency 4, two warmups, streaming raw completions, cap 128; zero benchmark errors. |
| Worker recovery | Old UID absent; replacement UID on the same node; inference restored in 225.6 s. Fourteen failed HTTP recovery probes retained. |
| Post-recovery load | Four requests at concurrency 1, zero warmups; all succeeded. Not directly comparable to initial throughput. |
| Evidence/cleanup | 35 local E2E files reviewed; before/after log-error lists empty; service/cluster ownership and cleanup UIDs match. |

The 225.6 s recovery interval starts with the restart test, not just model load.
Controller evidence is retained locally; there is no automatic off-VM backup of
all controller logs. The run's original receipts retain their pre-review wording.
No quality, engine inter-token latency, autoscaling, multi-node, VM recovery or
production availability claim follows from this bounded run. Fresh installation
and final-image E2E were verified separately, rather than in one final fresh run.

## How we reached the accepted run

| Stage | What happened and why it matters |
| --- | --- |
| Initial implementation | Added native Serve configuration, pinned-model staging, HTTP measurements and VM-side lifecycle control. The E2E runner connected these into one command. |
| October 22–24 image checks | Cloud builds exposed invalid uv sync syntax, incompatible data-library versions, a missing eager-import SDK and a Pydantic schema API mismatch. Fixed the recipe/lock/checker and moved native checks before large image export. See runbook troubleshooting. |
| October 25 dependency decision | Selected Ray 2.59/vLLM 0.27/Torch 2.13 in a separate image. Explicit compatible pins and native version checks prevent silent resolver downgrades or changes to training environments. See the dated dependency review for alternatives and reasoning. |
| First cloud E2E attempts | Wrong Hub credentials blocked model staging. After correction, a cluster-creation timing race required waiting for NotFound. Native inference/load then passed, but the default worker proxy probe conflicted with HeadOnly. |
| Final corrected image | Explicit worker Raylet readiness retained independent Serve/HTTP checks. Rebuilt/published the accepted digest and reused the installed Kubernetes cluster; load, worker replacement, recovery and cleanup all passed. |
| Review and closeout | Synced and reviewed results locally before releasing the VM. Fresh installation passed earlier, but the final-image success was in reuse mode; preserve that distinction in future claims. |

This is a decision/history summary, not a transcript of every attempt. The runbook
retains actionable fixes, the dependency review records compatibility rationale,
and retained run artifacts/source history provide detailed diagnostics.

## Implementation and operating references

The [runbook](7D_RUNBOOK.md#end-to-end-workflow-and-code-map) owns the invocation
graph, code map, cloud commands, evidence retention and troubleshooting. The
[dependency review](7D_DEPENDENCY_REVIEW.md) owns stack decisions and source links.
The [RayService example](examples/7D_RAYSERVICE_EXAMPLE.yaml) is a reading example;
use the renderer with an actual compatible image digest for execution.

Resolved integration decisions: native Serve builder rather than a custom serving
loop; separate training/serving images; raw completions retain Part 5 prompt
semantics; explicit Raylet-only worker readiness with `HeadOnly`; HTTP/Serve status
independently gates serving; native UID checks restrict intentional deletion and
cleanup. The controller tolerates cluster-creation NotFound while retaining API
and ownership failures. Detailed command fixes belong in the runbook.

Core completion requires no further implementation. Choose optional experiments
in [PLAN.md](../PLAN.md#optional-milestone-and-deferred-follow-ups) separately.
