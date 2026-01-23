# Flat DDP - Batched Gradient Communication (Section 2.3.1)

Implementation of Section 2.3.1: Improving DDP by batching all gradient communication into a single all-reduce operation.

## Overview

This module implements **Flat DDP**, which improves upon naive DDP by flattening all parameter gradients into a single tensor before communication. This reduces communication overhead by minimizing the number of collective operations.

**Key improvements over naive DDP**:
- ✅ Single batched all-reduce (1 call vs. ~1000+ calls)
- ✅ Better bandwidth utilization (larger messages)
- ✅ Lower latency (fewer kernel launches)
- ✅ Typically 1.2-2x speedup in communication time

**Still has limitations**:
- ❌ No overlap of computation and communication
- ❌ Must wait for entire backward pass before communication
- ❌ No parameter/optimizer sharding

This serves as an intermediate step toward fully optimized DDP implementations that use bucketing and overlap communication with computation.

## Files

- `flat_ddp_trainer.py` - Flat DDP trainer with batched gradient all-reduce
- `benchmark_comparison.py` - Compare naive vs. flat DDP implementations
- `run_comparison.sh` - Run comparison benchmark
- `README.md` - This file

## Quick Start

### Run Comparison Benchmark

Compare naive DDP (individual parameter all-reduces) vs. flat DDP (single batched all-reduce):

```bash
cd cs336_systems/flat_ddp
./run_comparison.sh
```

This will:
- Benchmark both implementations on XL model with 2 GPUs
- Measure total time per step and communication time
- Show speedup from batching communication calls
- Display number of all-reduce operations for each approach
- Save results to `../../results/flat_ddp/comparison_results.csv`

**Requirements**: 2+ GPUs (40GB+ recommended for XL), ~10-15 minutes runtime

### Expected Output

```
================================================================================
DDP Implementation Comparison Benchmark
================================================================================
Configuration:
  Model size: xl
  World size: 2 GPUs
  Batch size per GPU: 2
  Warm-up steps: 5
  Measured steps: 10

Running benchmarks...
--------------------------------------------------------------------------------
Benchmarking naive DDP (individual parameter all-reduces)...
✓ Naive DDP complete
Benchmarking flat DDP (single batched all-reduce)...
✓ Flat DDP complete

================================================================================
Results
================================================================================

Implementation       Avg Time/Iter   Avg Comm Time   Num Comm Ops
--------------------------------------------------------------------------------
Naive DDP                 XXX.XX ms        XX.XX ms             1152
Flat DDP                  YYY.YY ms        YY.YY ms                1

--------------------------------------------------------------------------------

Speedup (overall): 1.XXx
Speedup (communication): X.XXx

Analysis:
--------------------------------------------------------------------------------
Naive DDP: Communication overhead = XX.X% of total time
           Number of all-reduce calls = 1152

Flat DDP:  Communication overhead = YY.Y% of total time
           Number of all-reduce calls = 1

Batching gradients provides 1.XXx speedup by reducing from
1152 communication calls to 1 call(s).
```

## Implementation Details

### Flat DDP Algorithm

**Key difference from naive DDP**:

**Naive DDP** (many small operations):
```python
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad)    # ~1000+ calls
        param.grad /= world_size
```

**Flat DDP** (one batched operation):
```python
# Collect all gradients
gradients = [p.grad for p in model.parameters() if p.grad is not None]

# Flatten into single tensor
flat_grads = torch._utils._flatten_dense_tensors(gradients)

# Single all-reduce
dist.all_reduce(flat_grads)            # 1 call
flat_grads /= world_size

# Unflatten back to parameters
unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, gradients)

# Copy back
for param_grad, unflat_grad in zip(gradients, unflat_grads):
    param_grad.copy_(unflat_grad)
```

### Why Batching Helps

1. **Fewer kernel launches**: Reduces overhead from ~1000+ launches to 1
2. **Better bandwidth utilization**: Large messages use network more efficiently
3. **Lower per-operation latency**: Amortizes fixed communication costs

### Why Batching Alone Isn't Enough

While batching reduces communication overhead, it still:
- Waits for entire backward pass before communicating
- Cannot overlap computation with communication
- Leaves potential speedup on the table

The next optimization (Section 2.3.2) addresses this by overlapping communication with backward computation.

## Code Example

```python
from flat_ddp.flat_ddp_trainer import FlatDDPTrainer
from naive_ddp.naive_ddp_trainer import setup_distributed, shard_batch

# Setup distributed
rank = 0  # or 1, 2, ...
world_size = 2
setup_distributed(rank, world_size, backend="nccl")

# Create model and optimizer
model = create_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Use flat DDP trainer
trainer = FlatDDPTrainer(model, optimizer, rank, world_size, device=f"cuda:{rank}")

# Training loop (identical API to naive DDP)
for batch in dataloader:
    local_inputs, local_targets = shard_batch(batch, rank, world_size)
    step_info = trainer.train_step(local_inputs, local_targets)

    print(f"Loss={step_info['loss']:.4f}, "
          f"Comm={step_info['comm_time']*1000:.1f}ms")
```

## Custom Benchmarks

### Different Model Sizes

```bash
# Test with large model (if GPU memory limited)
uv run python benchmark_comparison.py --model-size large

# Test with medium model
uv run python benchmark_comparison.py --model-size medium
```

### Different GPU Counts

```bash
# Test with 4 GPUs
uv run python benchmark_comparison.py --world-size 4
```

### More Iterations

```bash
# More measured steps for stable results
uv run python benchmark_comparison.py --num-steps 20
```

## Output Files

Results saved to `../../results/flat_ddp/`:

- `comparison_results.csv` - Comparison benchmark results

CSV columns:
- `implementation`: "naive" or "flat"
- `model_size`: Model configuration name
- `d_model`, `num_layers`: Model dimensions
- `world_size`: Number of GPUs used
- `avg_step_time_ms`: Average time per training step (ms)
- `avg_comm_time_ms`: Average gradient communication time (ms)
- `comm_overhead_pct`: Communication overhead (% of total time)
- `num_comm_ops`: Number of all-reduce operations
- `speedup_vs_naive`: Overall speedup compared to naive DDP
- `comm_speedup_vs_naive`: Communication speedup compared to naive DDP

## Comparison: Naive vs. Flat DDP

| Aspect | Naive DDP | Flat DDP |
|--------|-----------|----------|
| **All-reduce calls** | ~1000+ | 1 |
| **Communication overhead** | High (20-40%) | Lower (10-25%) |
| **Bandwidth utilization** | Poor (small messages) | Good (large message) |
| **Computation/comm overlap** | No | No |
| **Implementation complexity** | Simple | Simple |
| **Speedup over naive** | 1.0x | ~1.2-2.0x |

## Troubleshooting

### Import errors

If you get import errors, make sure you're running from the correct directory:

```bash
cd cs336_systems/flat_ddp
uv run python benchmark_comparison.py
```

Or use the shell script which handles paths correctly:

```bash
./run_comparison.sh
```

### Port already in use

```bash
export MASTER_PORT=29501
./run_comparison.sh
```

### Out of memory

```bash
# Use smaller model
uv run python benchmark_comparison.py --model-size medium
```

## Requirements

**Section 2.3.1 (2 pts)**:
- **minimal_ddp_flat_benchmarking (2 pts)**:
  - Implement flat DDP with single batched all-reduce
  - Benchmark and compare with naive implementation
  - Report measured time per training iteration
  - Report measured time spent communicating gradients
  - Provide 1-2 sentences comparing results when batching vs. individually communicating gradients

## Next Steps

Section 2.3.1 addresses communication overhead by batching, but still doesn't overlap computation with communication. Future sections cover:

- **Section 2.3.2**: Overlap backward pass with individual parameter communication
- **Section 2.3.3**: Combine bucketing with overlap (PyTorch DDP approach)
- **Later sections**: ZeRO-2, ZeRO-3, FSDP for memory-efficient training