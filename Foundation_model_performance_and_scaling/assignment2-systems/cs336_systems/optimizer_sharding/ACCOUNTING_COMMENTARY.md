# Section 3: Optimizer State Sharding - Accounting Commentary

This document provides analysis for the three accounting questions in Section 3.

---

## Part (a): Memory Profiling (2-3 sentences)

**Configuration**: 1 node, 2 GPUs, XL model (d_model=1600, num_layers=48)

**Question**: Create a script to profile the peak memory usage when training language models with and without optimizer state sharding. Using the standard configuration (1 node, 2 GPUs, XL model size), report the peak memory usage after model initialization, directly before the optimizer step, and directly after the optimizer step. Do the results align with your expectations? Break down the memory usage in each setting.

### Results

**Peak Memory Usage:**

| Point | Without Sharding | With Sharding | Savings |
|-------|------------------|---------------|---------|
| After model init | [TO BE FILLED] GB | [TO BE FILLED] GB | [TO BE FILLED] GB |
| Before optimizer step | [TO BE FILLED] GB | [TO BE FILLED] GB | [TO BE FILLED] GB |
| After optimizer step | [TO BE FILLED] GB | [TO BE FILLED] GB | [TO BE FILLED] GB |

**Memory Breakdown (After Optimizer Step):**

*Without Sharding:*
- Master weights (FP32): ~6.8 GB
- Gradients (FP32): ~6.8 GB
- Optimizer states (AdamW, 2×FP32): ~13.6 GB
- Activations + overhead: ~X GB
- **Total**: ~Y GB

*With Sharding (world_size=2):*
- Master weights (FP32): ~6.8 GB
- Gradients (FP32): ~6.8 GB
- Optimizer states (AdamW, 2×FP32, sharded): ~6.8 GB (50% reduction!)
- Activations + overhead: ~X GB
- **Total**: ~Z GB

### Commentary

[TO BE FILLED AFTER RUNNING BENCHMARK]

**Template**: The optimizer state sharding reduces peak memory usage by approximately X GB (~Y%) as expected, with the reduction coming entirely from the optimizer states (AdamW's momentum and variance buffers). With world_size=2, each rank stores optimizer states for only half the parameters (~6.8 GB instead of ~13.6 GB), which aligns with the theoretical prediction of (world_size - 1) / world_size reduction in optimizer memory. The memory breakdown confirms that gradients and master weights remain fully replicated across ranks (as expected in this simplified ZeRO Stage 1 implementation), while only the optimizer states are sharded, demonstrating that the primary memory bottleneck for large model training with AdamW is indeed the optimizer state rather than the model parameters themselves.

---

## Part (b): Training Speed Impact (2-3 sentences)

**Configuration**: 1 node, 2 GPUs, XL model (d_model=1600, num_layers=48)

**Question**: How does our implementation of optimizer state sharding affect training speed? Measure the time taken per iteration with and without optimizer state sharding for the standard configuration (1 node, 2 GPUs, XL model size).

### Results

| Configuration | Time per Iteration | Overhead |
|---------------|-------------------|----------|
| Without sharding | [TO BE FILLED] ms | - |
| With sharding | [TO BE FILLED] ms | +[TO BE FILLED] ms (~X%) |

### Commentary

[TO BE FILLED AFTER RUNNING BENCHMARK]

**Template**: The optimizer state sharding introduces a modest overhead of approximately X ms per iteration (~Y% slowdown) compared to non-sharded training, which is acceptable given the Z GB (~W%) memory savings. This overhead comes from the additional broadcast operations needed to synchronize updated parameters after each optimizer step (broadcasting ~6.8 GB of FP32 weights across 2 GPUs via NVLink), which adds minimal latency compared to the total iteration time dominated by forward/backward computation (~800-1000 ms for XL model). The trade-off is favorable for memory-constrained scenarios where the X% slowdown enables training models that would otherwise not fit in GPU memory, and the overhead could be further reduced by overlapping the broadcast with other operations or using faster interconnects.

---

## Part (c): Comparison with ZeRO Stage 1 (2-3 sentences)

**Question**: How does our approach to optimizer state sharding differ from ZeRO stage 1 (described as ZeRO-DP Pos in Rajbhandari et al., 2020)?

### Analysis

**Our Implementation**:
- **What is sharded**: Optimizer states only
- **Gradient handling**: All-reduce (full gradient replicas on each rank)
- **Parameter synchronization**: Broadcast each parameter from owner rank after optimizer step
- **Communication volume per step**: ~1× model size (broadcast)
- **Memory savings**: Optimizer states only (~25% for AdamW with world_size=2)

**ZeRO Stage 1 (ZeRO-DP Pos)**:
- **What is sharded**: Optimizer states only (same!)
- **Gradient handling**: Reduce-scatter (sharded gradients)
- **Parameter synchronization**: All-gather parameters before forward/backward OR broadcast after update
- **Communication volume per step**: ~2× model size (reduce-scatter + all-gather)
- **Memory savings**: Optimizer states + gradients (~37.5% for AdamW with world_size=2)

**Key Differences**:

| Aspect | Our Implementation | ZeRO Stage 1 |
|--------|-------------------|--------------|
| Gradient storage | **Full** (not sharded) | **Sharded** (1/world_size) |
| Gradient communication | All-reduce → full replicas | Reduce-scatter → sharded |
| Memory savings | Optimizer only (~25%) | **Optimizer + gradients (~37.5%)** |
| Communication pattern | Simpler (all-reduce + broadcast) | More complex (reduce-scatter + all-gather) |

### Commentary

Our implementation differs from ZeRO Stage 1 (ZeRO-DP Pos) primarily in gradient handling: we maintain full gradient replicas on each rank using standard all-reduce, whereas ZeRO Stage 1 shards gradients using reduce-scatter, which provides additional memory savings (~6.8 GB for gradients with world_size=2) at the cost of a more complex communication pattern requiring all-gather operations. While both approaches shard optimizer states identically (achieving ~13.6 GB → ~6.8 GB reduction for AdamW), ZeRO Stage 1's gradient sharding enables ~37.5% total memory reduction versus our ~25% reduction, making it more memory-efficient for large-scale training where gradient memory becomes significant. Our simpler implementation is easier to understand and implement but leaves optimization opportunities on the table; extending to full ZeRO Stage 1 would require replacing all-reduce with reduce-scatter during backward pass and adding all-gather before using gradients in the optimizer step.

---

## Summary Table

| Part | Key Result |
|------|------------|
| (a) Memory Profiling | [TO BE FILLED] GB savings (~X%), optimizer states reduced by 50% |
| (b) Training Speed | ~X% overhead ([TO BE FILLED] ms), acceptable for Y GB memory savings |
| (c) vs ZeRO Stage 1 | Our impl: optimizer sharding only (~25% savings); ZeRO: optimizer + gradients (~37.5% savings) |

---

## Instructions for Filling Results

After running the benchmarks on H100 instance:

1. **Memory Profiling**: Run `bash benchmark.sh` and fill in the table in Part (a) with actual measurements from `results/optimizer_sharding/memory_profile.txt`

2. **Speed Comparison**: Fill in Part (b) table with actual measurements from `results/optimizer_sharding/speed_comparison.csv`

3. **Update Commentary**: Replace the "[TO BE FILLED]" placeholders in the commentary sections with actual numbers

4. **Validate Results**: Ensure the measured memory savings align with theoretical expectations (~50% reduction in optimizer states, ~25% total reduction)