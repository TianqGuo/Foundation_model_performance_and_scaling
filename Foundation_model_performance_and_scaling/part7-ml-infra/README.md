# Part 7 — ML Infrastructure

Distributed post-training and model serving on a shared Ray platform, carrying
forward Qwen2.5-Math-1.5B and MATH from Part 5.

**Status:** planning documentation is ready. Implementation, environment setup
and cloud provisioning have not started; no Part 7 results are available yet.

## Roadmap

| Milestone | Planned capability | Completion evidence |
| --- | --- | --- |
| 7A | verl + FSDP2 + vLLM post-training | Cloud generation, updates, evaluation and checkpointing with verified sharding across at least two training ranks. |
| 7B | Ray placement, observability and recovery | Recorded worker/GPU topology, durable artifacts and tested checkpoint resumption with explicit state-restoration limits. |
| 7C | Kubernetes training through KubeRay RayJob | Reproducible GPU deployment, retained operational evidence and validated recovery. |
| 7D | Ray Serve LLM + vLLM through RayService | Serve an exported 7A checkpoint; measure throughput, TTFT, latency and errors, and demonstrate service recovery. |
| 7E — optional | Ray Train + FSDP2 supervised training | Demonstrate platform reuse with a small SFT workload, two-rank sharding and recovery. |

The shared platform is **Kubernetes → KubeRay → Ray**. Training runs as batch
RayJobs; serving runs as a long-lived RayService. verl manages the RL workload;
Ray Train is reserved for optional SFT. Training and serving may use separate
pinned environments and run sequentially on a small GPU allocation.

## Reference and validation

The initial reference is Part 5's corrected GRPO-Clip implementation. Framework
integration will explicitly map reward, prompt, estimator, batching and evaluation
semantics. Completion requires retained execution and sharding evidence; matching
Part 5 accuracy or bitwise outputs is not an acceptance criterion.

All model loading, inference, serving and training run on **cloud GPUs**. Local
verification uses static checks and lightweight CPU tests without checkpoints.
Results will report tested configurations and measurement limits; multi-node
scalability and production reliability remain unclaimed.

Parts 1–6 remain preserved. Part 5 modernization is complete, and deferred GSPO
tuning does not block this roadmap. Implementation favors thin adapters and
framework configuration over additional training code.


