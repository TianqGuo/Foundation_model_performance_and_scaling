# Naïve Distributed Data Parallel (DDP) Training

Implementation of Section 2.2: A naïve implementation of vanilla distributed data parallel training.

## Overview

This module implements **vanilla DDP** by all-reducing individual parameter gradients after the backward pass. This approach is "naïve" because it causes high communication overhead by:
- All-reducing each gradient tensor separately (many small operations)
- Preventing overlap between communication and computation
- Higher latency from multiple kernel launches

**Key characteristics**:
- ✅ Full model replication (each GPU has complete copy)
- ✅ Full optimizer replication (each GPU has complete optimizer states)
- ❌ Individual gradient all-reduce (inefficient communication)
- ❌ No parameter/optimizer sharding (memory inefficient)

This implementation serves as a baseline to understand communication bottlenecks and motivate optimized DDP approaches (gradient bucketing, ZeRO, FSDP).

## Files

- `naive_ddp_trainer.py` - Core naïve DDP implementation
- `verify_correctness.py` - Verify DDP matches single-process training
- `benchmark_naive_ddp.py` - Benchmark communication overhead
- `run_verification.sh` - Run correctness verification
- `run_benchmark.sh` - Run benchmark on XL model

## Quick Start

### Step 1: Verify Correctness

Test that naïve DDP produces identical results to single-process training:

```bash
cd cs336_systems/naive_ddp
./run_verification.sh
```

This will:
- Train a toy model using both single-process and DDP (2 GPUs)
- Compare final model weights
- Report whether they match within tolerance

**Requirements**: 2+ GPUs, ~1-2 minutes runtime

### Step 2: Benchmark Communication Overhead

Measure communication overhead on XL model:

```bash
./run_benchmark.sh
```

This will:
- Benchmark XL model (d_model=1600, 48 layers) on 2 GPUs
- Measure total time per step and communication time
- Save results to `../../results/naive_ddp/benchmark_results.csv`
- Auto-adjust model size based on GPU memory

**Requirements**: 2+ GPUs (40GB+ recommended for XL), ~5-10 minutes runtime

## Implementation Details

### Vanilla DDP Algorithm

**Initialization**:
1. Each device constructs identical model (random init)
2. Broadcast parameters from rank 0 to all ranks
3. Each device holds full copy of model + optimizer states

**Training Iteration**:
1. **Shard batch**: Given `n` examples, each of `d` devices gets `n/d` examples
2. **Compute gradients**: Each device runs forward + backward on local data
3. **All-reduce gradients** (NAÏVE): All-reduce each parameter gradient individually
4. **Optimizer step**: Each device updates parameters with synchronized gradients
5. **Repeat**: Parameters stay synchronized across devices

### Code Example

```python
from naive_ddp_trainer import NaiveDDPTrainer, setup_distributed, shard_batch

# Setup distributed (rank 0 on GPU 0, rank 1 on GPU 1, etc.)
rank = 0  # or 1, 2, ...
world_size = 2
setup_distributed(rank, world_size, backend="nccl")

# Create model and optimizer
model = create_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create naïve DDP trainer
trainer = NaiveDDPTrainer(model, optimizer, rank, world_size, device=f"cuda:{rank}")

# Training loop
for epoch in range(num_epochs):
    for global_batch in dataloader:
        # Shard batch across devices
        local_inputs, local_targets = shard_batch(
            global_batch,
            rank,
            world_size,
        )

        # Training step (includes naïve gradient all-reduce)
        step_info = trainer.train_step(local_inputs, local_targets)

        print(f"Rank {rank}: Loss={step_info['loss']:.4f}, "
              f"Comm={step_info['comm_time']*1000:.1f}ms "
              f"({step_info['comm_fraction']*100:.1f}%)")
```

### Communication Pattern

**Naïve Approach** (this implementation):
```
For each parameter p in model:
    all_reduce(p.grad)  # Many small operations
```

**Problems**:
- High latency: Each all-reduce has kernel launch overhead
- No overlap: Must wait for each all-reduce to complete
- Poor bandwidth utilization: Small messages underutilize network

**Optimized Approach** (PyTorch DDP):
```
Bucket gradients into larger chunks
For each bucket:
    all_reduce(bucket)  # Fewer large operations
Overlap communication with backward pass
```

## Verification Details

The verification script (`verify_correctness.py`) tests correctness by:

1. Creating identical initial model weights
2. Training with single-process on full batch
3. Training with DDP (2 GPUs) on sharded batch
4. Comparing final weights (should be identical)

**Why they match**:
- Same initial weights (broadcast from rank 0)
- Same gradients (all-reduced across ranks)
- Same optimizer updates (deterministic)
- No randomness in training loop

## Benchmark Details

The benchmark script (`benchmark_naive_ddp.py`) measures:

**Metrics**:
- Total time per training step
- Compute time (forward + backward)
- Communication time (gradient all-reduce)
- Communication fraction (comm_time / total_time)
- Number of communication operations

**Expected Results** (XL model, 2 GPUs):
```
Total time:      ~XXX ms per step
Compute time:    ~XXX ms (YY%)
Communication:   ~XXX ms (ZZ%)

Communication overhead: ZZ% of total time
Number of all-reduce ops: ~NNNN (one per parameter)
```

The high communication overhead (typically 20-40%) motivates:
- Gradient bucketing (PyTorch DDP)
- Optimizer state sharding (ZeRO-2)
- Parameter sharding (ZeRO-3 / FSDP)

## Custom Benchmarks

### Test Different Model Sizes

```bash
# Test with large model (if GPU memory limited)
uv run python benchmark_naive_ddp.py --model-size large

# Test with medium model
uv run python benchmark_naive_ddp.py --model-size medium

# Test with more GPUs
uv run python benchmark_naive_ddp.py --world-size 4
```

### Adjust Benchmark Parameters

```bash
# More iterations for stable measurements
uv run python benchmark_naive_ddp.py --num-steps 20 --num-warmup 5

# Different sequence length
uv run python benchmark_naive_ddp.py --context-length 1024

# Larger batch size
uv run python benchmark_naive_ddp.py --batch-size 8
```

### Verify with Different Configurations

```bash
# More training steps
uv run python verify_correctness.py --num-steps 50

# More GPUs
uv run python verify_correctness.py --world-size 4

# Larger toy model
uv run python verify_correctness.py --num-samples 5000
```

## Output Files

Results saved to `../../results/naive_ddp/`:

- `benchmark_results.csv` - Benchmark timing data

CSV columns:
- `model_size`: Model configuration name
- `d_model`, `num_layers`: Model dimensions
- `total_time_ms`: Total time per step (ms)
- `compute_time_ms`: Forward + backward time (ms)
- `comm_time_ms`: Gradient communication time (ms)
- `comm_fraction`: Communication overhead (0-1)
- `num_comm_ops`: Number of all-reduce operations

## Platform Notes

### Local Testing (WSL2 / Multi-GPU Workstation)

- ✅ Verification works on any 2+ GPU system
- ⚠️ Benchmark may OOM on GPUs with <16GB memory
  - Use `--model-size medium` or `large` instead of `xl`
- ⚠️ WSL2 may have networking issues with distributed training
  - If errors occur, try running on native Linux

### H100 Cloud / Production

- ✅ Full XL model benchmark works well
- ✅ Fast iteration times with H100 performance
- ✅ Can test with more GPUs (4, 8) for scaling analysis

## Comparison: DDP vs Single-Process

| Aspect | Single-Process | Naïve DDP (2 GPUs) |
|--------|----------------|-------------------|
| **Batch size** | 4 | 4 (2 per GPU) |
| **Model replication** | 1 copy | 2 copies (full) |
| **Memory usage** | 1x | 2x (no savings) |
| **Computation** | 1x speed | ~2x speed |
| **Communication** | None | High overhead |
| **Effective speedup** | 1x | ~1.3-1.6x (due to comm) |

**Key insight**: Naïve DDP has significant communication overhead (20-40%), reducing effective speedup.

## Troubleshooting

### Port already in use

```bash
# Change master port in code or use different port
export MASTER_PORT=29501
```

### Out of memory

```bash
# Use smaller model
./run_benchmark.sh  # Auto-selects based on GPU memory

# Or manually specify
uv run python benchmark_naive_ddp.py --model-size medium
```

### DDP hangs or freezes

- Ensure all ranks reach collective operations (all-reduce, broadcast)
- Check that batch size is divisible by world size
- Verify all GPUs are accessible (`nvidia-smi`)

### Verification fails (weights don't match)

- Check that same random seed is used
- Verify all ranks use same learning rate
- Ensure no non-deterministic operations (dropout should be 0)

## Requirements

See Assignment 2 Section 2.2:
- **naive_ddp (5 pts)**: Implement and verify correctness
- **naive_ddp_benchmarking (3 pts)**: Benchmark XL model on 2 GPUs, measure communication overhead

## Next Steps

After understanding naïve DDP overhead, future sections will cover:
- Gradient bucketing (PyTorch DDP optimization)
- ZeRO-2 (optimizer state sharding)
- ZeRO-3 / FSDP (full parameter sharding)
- Pipeline parallelism
- Tensor parallelism

These optimizations reduce communication overhead and enable training much larger models.