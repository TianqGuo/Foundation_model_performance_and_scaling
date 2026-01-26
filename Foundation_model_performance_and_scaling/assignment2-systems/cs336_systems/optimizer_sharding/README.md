# Optimizer State Sharding (Section 3)

Implementation of optimizer state sharding for memory-efficient distributed training (similar to ZeRO Stage 1).

## Overview

**Key Idea**: Each rank maintains optimizer states for only ~1/world_size of parameters, broadcasting updated parameters after each step.

**Memory Savings**: ~(world_size - 1) / world_size reduction in optimizer memory (e.g., 50% with 2 GPUs)

**Trade-off**: Slight communication overhead (~1-2%) from parameter broadcasts

## Files

- `optimizer.py` - ShardedOptimizer implementation
- `benchmark_memory.py` - Part (a): Memory profiling script
- `benchmark_speed.py` - Part (b): Training speed script
- `benchmark.sh` - Main benchmark runner
- `ACCOUNTING_COMMENTARY.md` - Parts (a), (b), (c) analysis

## Quick Start

### Run Tests (Local)

```bash
# Run once
uv run pytest tests/test_sharded_optimizer.py -v

# Run 5 times for reliability
for i in {1..5}; do
    echo "Test run $i/5"
    uv run pytest tests/test_sharded_optimizer.py -v || break
done
```

**Status**: ✅ All tests pass (verified 5 times)

### Run Benchmarks (Remote H100 Instance)

⚠️ **IMPORTANT**: Run on remote instance with 2+ GPUs!

```bash
cd cs336_systems/optimizer_sharding
bash benchmark.sh
```

**Output**:
- `../../results/optimizer_sharding/memory_profile.txt`
- `../../results/optimizer_sharding/speed_comparison.csv`

**Then**: Fill results into `ACCOUNTING_COMMENTARY.md`

## Implementation

### ShardedOptimizer Class

```python
class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        # Partition parameters across ranks (round-robin)
        # Create wrapped optimizer with only this rank's parameters

    def step(self, closure=None, **kwargs):
        # 1. Update this rank's parameters via wrapped optimizer
        # 2. Broadcast each parameter from its owner rank
```

**Key Methods**:
- `__init__()` - Partition parameters, create wrapped optimizer
- `step()` - Update + broadcast synchronization
- `add_param_group()` - Support for parameter groups

### Memory Breakdown (XL Model, 2 GPUs)

**Without Sharding** (per GPU):
```
Weights:           6.8 GB
Gradients:         6.8 GB
Optimizer states: 13.6 GB  (AdamW: momentum + variance)
Total:            27.2 GB
```

**With Sharding** (per GPU):
```
Weights:           6.8 GB
Gradients:         6.8 GB
Optimizer states:  6.8 GB  (sharded: 50% reduction!)
Total:            20.4 GB
Savings:           6.8 GB (25%)
```

### Usage Example

```python
from cs336_systems.optimizer_sharding.optimizer import ShardedOptimizer

# Create model
model = TransformerLM(...).to(device)

# Create sharded optimizer (wraps AdamW)
optimizer = ShardedOptimizer(
    model.parameters(),
    torch.optim.AdamW,
    lr=1e-4,
    weight_decay=0.01
)

# Training loop (identical to standard training!)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Updates this rank's params + broadcasts
```

## Expected Performance

### Memory (XL Model, 2 GPUs)

| Component | No Sharding | Sharded | Savings |
|-----------|-------------|---------|---------|
| Optimizer states | 13.6 GB | 6.8 GB | **50%** |
| Total memory | ~27 GB | ~20 GB | **25%** |

### Speed (XL Model, 2 GPUs)

Expected overhead: ~1-2% (~10-20 ms per iteration)
- Broadcast: ~6.8 GB FP32 weights
- NVLink (~900 GB/s): ~7.6 ms
- Total iteration: ~800-1000 ms
- Overhead: <1%

## Comparison: Our Implementation vs ZeRO Stage 1

| Aspect | Our Implementation | ZeRO Stage 1 |
|--------|-------------------|--------------|
| Optimizer states | Sharded ✓ | Sharded ✓ |
| Gradients | **Full** (replicated) | **Sharded** |
| Communication | All-reduce + broadcast | Reduce-scatter + all-gather |
| Memory savings | ~25% (optimizer only) | **~37.5%** (optimizer + gradients) |

**Key Difference**: ZeRO Stage 1 also shards gradients, saving additional ~6.8 GB.

## Benchmarking Tasks

### Part (a): Memory Profiling (2 pts)

Profile peak memory at 3 points:
1. After model initialization
2. Before optimizer step (after backward)
3. After optimizer step

Compare with/without sharding, provide component breakdown.

### Part (b): Training Speed (2 pts)

Measure time per iteration with/without sharding on standard config (1 node, 2 GPUs, XL model).

### Part (c): ZeRO Stage 1 Comparison (1 pt)

Compare our approach with ZeRO Stage 1 (Rajbhandari et al., 2020).

**See `ACCOUNTING_COMMENTARY.md` for detailed analysis.**

## Requirements

- PyTorch with distributed support
- 2+ GPUs for benchmarks (H100s recommended)
- XL model: ~40-50 GB GPU memory per GPU
- Runtime: ~10-15 minutes

## Troubleshooting

**Import errors**: Use `uv run` from project root

**Port conflicts**: `export MASTER_PORT=29503`

**Out of memory**: Use `--model-size large` or `--batch-size 1`

## References

- Rajbhandari et al. (2020): "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"