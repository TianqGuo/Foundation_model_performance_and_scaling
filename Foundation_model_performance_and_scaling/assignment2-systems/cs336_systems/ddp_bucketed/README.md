# DDP with Bucketed Gradients and Overlapping Computation (Section 2.3.3)

Implementation of Section 2.3.3: DDP that groups parameters into buckets and overlaps backward pass computation with bucketed gradient communication.

## Overview

This module implements **DDP with bucketed gradients and computation/communication overlap**, combining the best of both worlds:

**Key innovation**: Group parameters into buckets AND overlap communication with computation!

**How it works**:
1. Group parameters into buckets (reverse order, max `bucket_size_mb` each)
2. Register backward hooks on each parameter
3. When all parameters in a bucket are ready, flatten and launch async all-reduce
4. Continue backward pass while bucket communication happens
5. Wait for all buckets to finish before optimizer step

**Improvements over previous implementations**:
- ✅ Overlaps communication with computation (like individual overlap)
- ✅ Batches communication into buckets (like flat DDP)
- ✅ **Best of both**: Fewer communication calls AND overlapping!
- ✅ Configurable bucket size for performance tuning

**Still not as good as PyTorch production DDP** (which has additional optimizations), but close!

## Files

- `ddp.py` - DDP wrapper class with bucketing, hooks, and async all-reduce
- `benchmark_bucketed.py` - Benchmark with various bucket sizes
- `benchmark_part_a.sh` - Main benchmark script for Section 2.3.3 Part (a)
- `run_benchmark.sh` - Alias for benchmark_part_a.sh
- `BENCHMARKING_COMMENTARY.md` - Part (a) analysis: benchmark results and commentary
- `README.md` - This file

## Quick Start

### Run Tests (Local - Single RTX 4090)

Test the bucketed DDP implementation for correctness:

```bash
cd ~/MachineLearningNotes/stanford-cs336/assignment2-systems

# Run tests once
uv run pytest tests/test_ddp.py -v

# Run multiple times (5) to ensure reliability
for i in {1..5}; do
    echo "Test run $i/5"
    uv run pytest tests/test_ddp.py -v || break
done
```

**Can run on**: Local laptop with single GPU (RTX 4090)
**Purpose**: Verify implementation correctness

### Run Benchmark (Remote - Multi-GPU H100s)

⚠️ **IMPORTANT**: Performance benchmarks require 2+ GPUs and should be run on remote instance (vast.ai/lambda.ai), NOT on local laptop!

Compare performance across different bucket sizes:

```bash
# On H100 instance with 2+ GPUs:
cd cs336_systems/ddp_bucketed
bash benchmark_part_a.sh
```

This will:
- Benchmark bucket sizes: 1, 10, 100, 1000 MB
- Show performance vs number of buckets
- Save results to `../../results/ddp_bucketed/bucket_size_comparison.csv`

**Requirements**:
- 2+ GPUs (H100s recommended)
- XL model requires ~40GB GPU memory per GPU
- ~10-15 minutes runtime

### Expected Output

```
================================================================================
Bucketed DDP Benchmark
================================================================================
Configuration:
  Model size: xl
  World size: 2 GPUs
  Bucket sizes: [1.0, 10.0, 100.0, 1000.0] MB

Benchmarking with bucket size: 1.0 MB...
  Time: XXX.XX ms, Buckets: ~350
Benchmarking with bucket size: 10.0 MB...
  Time: XXX.XX ms, Buckets: ~40
Benchmarking with bucket size: 100.0 MB...
  Time: XXX.XX ms, Buckets: ~4
Benchmarking with bucket size: 1000.0 MB...
  Time: XXX.XX ms, Buckets: 1

================================================================================
Results
================================================================================

Bucket Size (MB)     Num Buckets     Time/Iter (ms)
--------------------------------------------------------------------------------
1.0                  ~350            XXX.XX
10.0                 ~40             XXX.XX
100.0                ~4              XXX.XX
1000.0               1               XXX.XX
```

## Implementation Details

### Key Concept: Bucketing + Hooks + Async All-Reduce

The innovation is combining bucketing with overlap:

```python
class DDP(nn.Module):
    def __init__(self, module, bucket_size_mb):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        # Group parameters into buckets (reverse order)
        self._create_buckets()

        # Register hooks on each parameter
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_allreduce_hook(param)
                )

    def _create_buckets(self):
        # Get params in reverse order (gradients ready in reverse)
        params_reversed = list(reversed([p for p in self.module.parameters() if p.requires_grad]))

        # Group into buckets of max bucket_size_mb MB
        bucket_size_bytes = self.bucket_size_mb * 1024 * 1024
        current_bucket = []
        current_size = 0

        for param in params_reversed:
            param_size = param.numel() * param.element_size()

            if current_bucket and (current_size + param_size) > bucket_size_bytes:
                # Finalize current bucket, start new one
                self.buckets.append(current_bucket)
                current_bucket = [param]
                current_size = param_size
            else:
                current_bucket.append(param)
                current_size += param_size

        if current_bucket:
            self.buckets.append(current_bucket)

    def _make_allreduce_hook(self, param):
        def hook(grad):
            # Mark param as ready in its bucket
            bucket_idx = self.param_to_bucket[param]
            self.bucket_ready_params[bucket_idx].add(param)

            # If all params in bucket ready, launch async all-reduce
            if len(self.bucket_ready_params[bucket_idx]) == len(self.buckets[bucket_idx]):
                self._allreduce_bucket(bucket_idx)

            return None
        return hook

    def _allreduce_bucket(self, bucket_idx):
        # Flatten bucket gradients
        grads = [p.grad.data for p in self.buckets[bucket_idx] if p.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        # Launch async all-reduce (returns immediately!)
        handle = dist.all_reduce(flat_grads, async_op=True)

        # Store for later
        self.bucket_handles.append((handle, flat_grads, bucket_idx))

    def finish_gradient_synchronization(self):
        # Wait for all buckets and unflatten
        for handle, flat_grads, bucket_idx in self.bucket_handles:
            handle.wait()
            flat_grads /= self.world_size

            # Unflatten and copy back
            grads = [p.grad.data for p in self.buckets[bucket_idx] if p.grad is not None]
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)

            for param_grad, unflat_grad in zip(grads, unflat_grads):
                param_grad.copy_(unflat_grad)

        # Clear state
        self.bucket_handles.clear()
        for ready_set in self.bucket_ready_params:
            ready_set.clear()
```

### Timeline Comparison

**Individual Overlap DDP** (Section 2.3.2):
```
Backward Pass:
  Grad[param 435] ready ──► Async all-reduce[435] (0.05ms each)
  Grad[param 434] ready ──► Async all-reduce[434]
  ...
  Grad[param 1] ready ────► Async all-reduce[1]

Total: ~435 communication operations (small overhead)
```

**Bucketed Overlap DDP** (Section 2.3.3):
```
Backward Pass:
  Bucket 4 ready ──► Async all-reduce[bucket 4] (~100 params, 5ms)
  Bucket 3 ready ──► Async all-reduce[bucket 3]
  Bucket 2 ready ──► Async all-reduce[bucket 2]
  Bucket 1 ready ──► Async all-reduce[bucket 1]

Total: ~4 communication operations (lower overhead!)
```

**Key benefit**: Fewer communication calls + overlapping = Best performance!

### Usage Pattern

```python
from ddp_bucketed.ddp import DDP

# Setup distributed
setup_distributed(rank, world_size, backend="nccl")

# Create model and wrap with DDP (25 MB buckets - PyTorch default)
model = create_model().to(device)
ddp_model = DDP(model, bucket_size_mb=25.0)

# Create optimizer
optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    inputs, targets = batch

    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)

    # Backward pass (async all-reduce triggered automatically per bucket!)
    loss.backward()

    # Wait for all buckets to finish
    ddp_model.finish_gradient_synchronization()

    # Optimizer step (gradients are now synchronized)
    optimizer.step()
```

### Bucket Size Trade-offs

**Small buckets (e.g., 1 MB)**:
- ✅ More overlap (buckets ready sooner)
- ❌ More communication calls (higher overhead)
- ❌ Less efficient use of bandwidth

**Large buckets (e.g., 1000 MB)**:
- ✅ Fewer communication calls (lower overhead)
- ✅ More efficient use of bandwidth
- ❌ Less overlap (must wait for whole bucket)

**Optimal bucket size** depends on:
- Model size
- Number of GPUs
- Network bandwidth
- Backward pass computation time

PyTorch uses **25 MB** as default (empirically good for most cases).

## Expected Performance

### Speedup Analysis

For XL model on 2 H100 GPUs:

| Implementation | Communication Ops | Expected Time |
|----------------|-------------------|---------------|
| Naive DDP | ~435 individual | ~1060 ms |
| Flat DDP | 1 batched | ~1010 ms |
| Overlap Individual | ~435 async | ~860 ms |
| **Bucketed (optimal)** | **~4-10 async** | **~850-870 ms** |

**Expected results**:
- Bucket size too small (1 MB): ~860-870 ms (like individual overlap)
- Bucket size optimal (10-100 MB): ~850-860 ms (best!)
- Bucket size too large (1000 MB): ~860-880 ms (like flat DDP)

The optimal bucket size balances:
1. Few enough buckets to reduce overhead
2. Small enough buckets to maximize overlap

## Output Files

Results saved to `../../results/ddp_bucketed/`:

- `bucket_size_comparison.csv` - Performance vs bucket size

CSV columns:
- `bucket_size_mb`: Bucket size in megabytes
- `num_buckets`: Number of buckets created
- `model_size`: Model configuration name
- `d_model`, `num_layers`: Model dimensions
- `world_size`: Number of GPUs used
- `avg_step_time_ms`: Average time per training step (ms)

## Mathematical Analysis (Part b)

### Communication Overhead Model

Given:
- `s` = total size of model parameters (bytes)
- `w` = all-reduce bandwidth (bytes/second)
- `o` = overhead per communication call (seconds)
- `n_b` = number of buckets

**Assumptions**:
- Gradient computation time = gradient communication time (per bucket)
- Buckets are evenly sized: each bucket has `s / n_b` bytes

**Model**:

```
Communication overhead = Time spent waiting after backward completes

Without overlap:
  Total_time = Backward_time + Communication_time
  Overhead = Communication_time = (s / w) + (n_b * o)

With overlap (ideal):
  Backward_time = sum(bucket_backward_times)
  Communication_time_per_bucket = (s / n_b) / w + o

  Since computation time = communication time (by assumption):
    bucket_backward_time = (s / n_b) / w + o

  After backward completes, no buckets remain!
  Overhead = 0  (ideal case)

With overlap (realistic):
  Last bucket might not finish before backward completes
  Overhead ≈ max(0, last_bucket_comm_time - last_bucket_compute_time)

  In worst case (last bucket finishes last):
    Overhead ≈ (s / n_b) / w + o
```

**Simplified model** (assuming backward slightly faster than communication):
```
Overhead = (s / n_b) / w + o
```

This shows:
- More buckets (larger `n_b`) → smaller per-bucket time → less overhead
- But more buckets → more overhead calls (`o`)
- Trade-off!

### Optimal Bucket Size

To minimize overhead, take derivative with respect to `n_b`:

```
Overhead(n_b) = s / (n_b * w) + n_b * o

d(Overhead)/d(n_b) = -s / (n_b^2 * w) + o = 0

Solving for n_b:
  s / (n_b^2 * w) = o
  n_b^2 = s / (o * w)
  n_b_optimal = sqrt(s / (o * w))

Optimal bucket size:
  bucket_size_optimal = s / n_b_optimal = sqrt(s * o * w)
```

**Key insights**:
- Optimal bucket size ∝ √(model size)
- Optimal bucket size ∝ √(overhead per call)
- Optimal bucket size ∝ √(bandwidth)

For typical values:
- s = 1.7 GB (XL model)
- o = 0.01 ms (H100 launch overhead)
- w = 900 GB/s (H100 NVLink)

```
bucket_size_optimal = sqrt(1.7e9 * 1e-5 * 9e11) ≈ 124 MB
```

This suggests **~100 MB buckets** are optimal for XL model on H100s!

## Custom Benchmarks

### Different Bucket Sizes

```bash
# Test specific bucket sizes
uv run python benchmark_bucketed.py --bucket-sizes 5 25 50 100

# Fine-grained search around optimal
uv run python benchmark_bucketed.py --bucket-sizes 75 100 125 150
```

### Different Model Sizes

```bash
# Test with large model
uv run python benchmark_bucketed.py --model-size large --bucket-sizes 10 25 50

# Test with medium model
uv run python benchmark_bucketed.py --model-size medium --bucket-sizes 10 25 50
```

### Different GPU Counts

```bash
# Test with 4 GPUs
uv run python benchmark_bucketed.py --world-size 4 --bucket-sizes 10 25 100
```

## Comparison: All DDP Implementations

| Aspect | Naive | Flat | Overlap Individual | **Bucketed** |
|--------|-------|------|-------------------|-------------|
| **Comm calls** | ~435/step | 1/step | ~435/step | **~4-10/step** |
| **Comm type** | Sync | Sync | Async | **Async** |
| **Overlap** | No | No | Yes | **Yes** |
| **Bucketing** | No | Yes | No | **Yes** |
| **Speedup** | 1.0x | ~1.02x | ~1.02x | **~1.02-1.05x** |
| **Tunable** | No | No | No | **Yes** (bucket size) |

## Troubleshooting

### Import errors

```bash
cd cs336_systems/ddp_bucketed
uv run python benchmark_bucketed.py
```

### Port already in use

```bash
export MASTER_PORT=29502
./run_benchmark.sh
```

### Out of memory

```bash
uv run python benchmark_bucketed.py --model-size medium
```

## Requirements

**Section 2.3.3 (11 pts total)**:

**ddp_overlap_bucketed (8 pts)**:
- Implement DDP container class with bucketed gradient communication
- Overlap gradient communication with backward computation
- Use `bucket_size_mb` parameter
- Bucket parameters in reverse order
- Use `register_post_accumulate_grad_hook()` for automatic triggering
- Use asynchronous all-reduce (`async_op=True`) per bucket
- Flatten/unflatten gradients within buckets
- Implement adapters: `adapters.get_ddp_bucketed`, `adapters.ddp_bucketed_on_after_backward`, `adapters.ddp_bucketed_on_train_batch_start`
- Test: `uv run pytest tests/test_ddp.py`
- Run tests multiple times (e.g., 5) to ensure reliable passing

**ddp_bucketed_benchmarking (3 pts)**:
- **(a)** Benchmark with bucket sizes: 1, 10, 100, 1000 MB on 1 node, 2 GPUs, XL model
  - Compare with previous implementations
  - Deliverable: Measured time per iteration for each bucket size, 3-4 sentence commentary
- **(b)** Mathematical modeling
  - Equation for communication overhead as function of s, w, o, n_b
  - Equation for optimal bucket size
  - Deliverable: Both equations with brief derivation

## Next Steps

Section 2.3.3 achieves near-optimal DDP performance by combining bucketing and overlapping. Future optimizations:

- **Later sections**: ZeRO-2, ZeRO-3, FSDP for memory-efficient training
- **Production DDP**: PyTorch's implementation adds more optimizations (DDP reducer, gradient compression, etc.)

This bucketed overlap DDP is very close to PyTorch's production DDP!