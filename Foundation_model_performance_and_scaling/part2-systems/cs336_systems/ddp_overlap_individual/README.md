# DDP with Overlapping Computation and Communication (Section 2.3.2)

Implementation of Section 2.3.2: DDP that overlaps backward pass computation with gradient communication using backward hooks and asynchronous all-reduce.

## Overview

This module implements **DDP with computation/communication overlap**, which improves upon both naive and flat DDP by:

**Key innovation**: While the backward pass is still computing gradients for later layers, we can **already start communicating** gradients for earlier layers that are ready!

**How it works**:
1. Register backward hooks on each parameter using `register_post_accumulate_grad_hook()`
2. When a gradient becomes ready during backward(), immediately launch **asynchronous** all-reduce
3. Continue backward pass computation while communication happens in the background
4. Wait for all communication to complete before optimizer step

**Improvements over previous implementations**:
- ✅ Overlaps communication with computation (unlike naive and flat DDP)
- ✅ Reduces effective communication overhead by parallelizing it with backward pass
- ✅ Individual parameter communication (unlike flat DDP's single batched call)
- ✅ Simple implementation using PyTorch's built-in hooks

**Still has limitations**:
- ❌ Many small communication operations (like naive DDP)
- ❌ Not as efficient as bucketed DDP (Section 2.3.3)

This serves as a stepping stone toward PyTorch's production DDP which combines bucketing AND overlapping.

## Files

- `ddp.py` - DDP wrapper class with backward hooks and async all-reduce
- `benchmark_overlap.py` - Benchmark and compare with naive/flat DDP
- `run_benchmark.sh` - Run benchmark script
- `profile_overlap.py` - Profiling script for Nsight Systems
- `run_profiling.sh` - Run profiling for naive vs overlap comparison
- `README.md` - This file

## Quick Start

### Run Benchmark

Compare DDP with overlap against naive and flat implementations:

```bash
cd cs336_systems/ddp_overlap_individual
./run_benchmark.sh
```

This will:
- Benchmark overlap DDP on XL model with 2 GPUs
- Load previous naive/flat results for comparison
- Show speedup from overlapping computation with communication
- Save results to `../../results/ddp_overlap_individual/benchmark_results.csv`

**Requirements**: 2+ GPUs (40GB+ recommended for XL), ~5-10 minutes runtime

### Expected Output

```
================================================================================
DDP with Overlapping Computation and Communication Benchmark
================================================================================
Configuration:
  Model size: xl
  World size: 2 GPUs
  Batch size per GPU: 2
  Warm-up steps: 5
  Measured steps: 10

Running benchmark...
--------------------------------------------------------------------------------
Benchmarking overlap DDP (individual params, async all-reduce)...
✓ Overlap DDP complete

================================================================================
Results
================================================================================

Overlap DDP: XXX.XX ms per step

Comparison with previous implementations:
--------------------------------------------------------------------------------
Implementation                 Avg Time/Iter   Speedup
--------------------------------------------------------------------------------
Naive DDP (no overlap)              1061.42 ms           1.00x
Flat DDP (no overlap)               1010.45 ms           1.05x
Overlap DDP (with overlap)           9XX.XX ms           1.XXx

================================================================================
✓ Results saved to: ../../results/ddp_overlap_individual/benchmark_results.csv
================================================================================
```

## Implementation Details

### Key Concept: Backward Hooks + Async All-Reduce

The core innovation is using PyTorch's `register_post_accumulate_grad_hook()`:

```python
class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.comm_handles = []

        # Register hook on each parameter
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_allreduce_hook(param)
                )

    def _make_allreduce_hook(self, param):
        """Create hook that launches async all-reduce when gradient is ready."""
        def hook(grad):
            # Launch async all-reduce (returns immediately!)
            handle = dist.all_reduce(grad.data, async_op=True)
            self.comm_handles.append((handle, grad))
            return None
        return hook

    def finish_gradient_synchronization(self):
        """Wait for all async operations to complete."""
        for handle, grad in self.comm_handles:
            handle.wait()  # Block until communication is queued
            grad.data /= self.world_size
        self.comm_handles.clear()
```

### Timeline Comparison

**Naive/Flat DDP (NO overlap)**:
```
Forward Pass
    ↓
Backward Pass (compute all gradients)
    ↓
────────────────────── All gradients ready ──────────────────────
    ↓
All-Reduce (communication)  ←── Computation idle during this time
    ↓
Optimizer Step
```

**Overlap DDP (WITH overlap)**:
```
Forward Pass
    ↓
Backward Pass starts
    ↓
Gradient[layer 48] ready ──► Async all-reduce[48] starts │
    ↓                                                      │
Gradient[layer 47] ready ──► Async all-reduce[47] starts │ Communication
    ↓                                                      │ happens in
Gradient[layer 46] ready ──► Async all-reduce[46] starts │ parallel with
    ↓                                                      │ backward pass
    ...                            ...                     │ computation!
    ↓                                                      │
Gradient[layer 0] ready ───► Async all-reduce[0] starts  │
    ↓                                                      │
────────────────────── All gradients ready ───────────────────────
    ↓
finish_gradient_synchronization()  ←── Wait for remaining communication
    ↓
Optimizer Step
```

**Key benefit**: Communication time is **hidden** behind backward pass computation!

### Usage Pattern

```python
from ddp_overlap_individual.ddp import DDP

# Setup distributed
setup_distributed(rank, world_size, backend="nccl")

# Create model and wrap with DDP
model = create_model().to(device)
ddp_model = DDP(model)  # Registers hooks automatically

# Create optimizer (operates on ddp_model.parameters())
optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    inputs, targets = batch

    # Forward pass
    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)

    # Backward pass (async all-reduce triggered automatically!)
    loss.backward()

    # Wait for all communication to finish
    ddp_model.finish_gradient_synchronization()

    # Optimizer step (gradients are now synchronized)
    optimizer.step()
```

### Why Async All-Reduce?

**Synchronous all-reduce** (`async_op=False`):
```python
dist.all_reduce(grad)  # Blocks until operation is queued
# Cannot continue backward pass until this completes
```

**Asynchronous all-reduce** (`async_op=True`):
```python
handle = dist.all_reduce(grad, async_op=True)  # Returns immediately!
# Backward pass continues while communication happens
# ...
handle.wait()  # Wait later when we need the result
```

The async version allows backward computation to continue while communication happens in the background.

## Expected Performance

### Speedup Analysis

For a model with N layers and backward time T:
- **Without overlap**: Total time = T (compute) + C (communication)
- **With overlap**: Total time ≈ max(T, C) + residual

If communication time C < backward time T, overlap can **hide most communication**!

**Expected results** (XL model, 2 GPUs, H100s):
- Backward pass: ~900-950ms
- Communication: ~100-110ms for individual params
- **Without overlap**: ~1050-1060ms total
- **With overlap**: ~950-1000ms total (communication mostly hidden)
- **Speedup**: ~1.05-1.1x

The speedup is limited because:
1. Gradients become ready in sequence (not all at once)
2. Last few layers must wait for communication to finish
3. H100 NVLink is so fast that compute still dominates

## Output Files

Results saved to `../../results/ddp_overlap_individual/`:

- `benchmark_results.csv` - Comparison with naive and flat DDP

CSV columns:
- `implementation`: "naive", "flat", or "overlap_individual"
- `model_size`: Model configuration name
- `d_model`, `num_layers`: Model dimensions
- `world_size`: Number of GPUs used
- `avg_step_time_ms`: Average time per training step (ms)
- `speedup_vs_naive`: Overall speedup compared to naive DDP

## Custom Benchmarks

### Different Model Sizes

```bash
# Test with large model
uv run python benchmark_overlap.py --model-size large

# Test with medium model
uv run python benchmark_overlap.py --model-size medium
```

### Different GPU Counts

```bash
# Test with 4 GPUs
uv run python benchmark_overlap.py --world-size 4
```

### More Iterations

```bash
# More measured steps for stable results
uv run python benchmark_overlap.py --num-steps 20
```

## Profiling with Nsight Systems

To visualize the overlap between computation and communication, use NVIDIA Nsight Systems profiler.

### Quick Start

Profile both naive and overlap DDP implementations:

```bash
cd cs336_systems/ddp_overlap_individual
./run_profiling.sh
```

This will generate:
- `naive_ddp.nsys-rep` - Profiling report for naive DDP (no overlap)
- `overlap_ddp.nsys-rep` - Profiling report for overlap DDP (with overlap)

### View Results

Open the profiling reports in Nsight Systems GUI:

```bash
nsys-ui naive_ddp.nsys-rep
nsys-ui overlap_ddp.nsys-rep
```

### What to Look For

**Naive DDP timeline (no overlap)**:
```
Forward Pass
    ↓
Backward Pass (all gradients computed)
    ↓
────────────────────── Backward completes ──────────────────────
    ↓
All-Reduce (NCCL operations) ←── GPU compute idle during this
    ↓
Optimizer Step
```

**Overlap DDP timeline (with overlap)**:
```
Forward Pass
    ↓
Backward Pass starts
    ↓
│ Backward compute │ ← NCCL all-reduce │  ← Overlap!
│  (CUDA kernels)  │ ← starts during   │
│                  │ ← backward pass    │
    ↓
────────────────────── Backward completes ──────────────────────
    ↓
Wait for remaining NCCL operations
    ↓
Optimizer Step
```

### Key Observations

In the Nsight Systems timeline, look for:

1. **Naive DDP**:
   - NCCL operations appear AFTER backward pass completes
   - Sequential execution: Backward → All-Reduce → Optimizer
   - GPU compute is idle during NCCL communication

2. **Overlap DDP**:
   - NCCL operations start DURING backward pass
   - Parallel execution: Backward kernels overlap with NCCL calls
   - Communication is hidden behind computation

### Manual Profiling

Profile individual implementations:

```bash
# Profile naive DDP
nsys profile -o naive_ddp --trace=cuda,nvtx \
    uv run python profile_overlap.py --implementation naive

# Profile overlap DDP
nsys profile -o overlap_ddp --trace=cuda,nvtx \
    uv run python profile_overlap.py --implementation overlap
```

### Taking Screenshots for Report

1. Open both `.nsys-rep` files in Nsight Systems GUI
2. Zoom in on one training iteration
3. Look for the "Training Step" NVTX range
4. Take screenshots showing:
   - **Naive**: All-Reduce happening after Backward
   - **Overlap**: NCCL operations during Backward

These screenshots demonstrate the overlap optimization visually.

## Comparison: Three DDP Implementations

| Aspect | Naive DDP | Flat DDP | Overlap DDP |
|--------|-----------|----------|-------------|
| **Communication calls** | ~435/step | 1/step | ~435/step |
| **Communication type** | Sync all-reduce | Sync all-reduce | **Async** all-reduce |
| **Overlap with compute** | No | No | **Yes** |
| **Communication overhead** | High (~10%) | Medium (~6%) | **Low (~2-5%)** |
| **Speedup over naive** | 1.0x | ~1.05x | **~1.05-1.1x** |

## Troubleshooting

### Import errors

```bash
cd cs336_systems/ddp_overlap_individual
uv run python benchmark_overlap.py
```

### Port already in use

```bash
export MASTER_PORT=29501
./run_benchmark.sh
```

### Out of memory

```bash
uv run python benchmark_overlap.py --model-size medium
```

## Requirements

**Section 2.3.2 (6 pts total)**:

**ddp_overlap_individual_parameters (5 pts)**:
- Implement DDP container class overlapping gradient communication with backward computation
- Use `register_post_accumulate_grad_hook()` for automatic triggering
- Use asynchronous all-reduce (`async_op=True`)
- Implement adapters: `adapters.get_ddp_individual_parameters`, `adapters.ddp_individual_parameters_on_after_backward` (optional)
- Test: `uv run pytest tests/test_ddp_individual_parameters.py`
- Run tests multiple times (e.g., 5) to ensure reliable passing

**ddp_overlap_individual_parameters_benchmarking (1 pt)**:
- **(a)** Benchmark overlap DDP on 1 node, 2 GPUs, XL model
  - Compare with naive and flat DDP implementations
  - Deliverable: Measured time per training iteration, 1-2 sentences comparing results
- **(b)** Profile with Nsight Systems to visualize overlap
  - Run `./run_profiling.sh` to generate profiling reports
  - Compare naive vs overlap DDP traces
  - Deliverable: 2 screenshots (naive showing no overlap, overlap showing overlap) demonstrating compute/communication overlap difference

## Next Steps

Section 2.3.2 addresses communication overhead by overlapping, but still uses many small operations. Future sections:

- **Section 2.3.3**: Combine bucketing with overlap (PyTorch production DDP)
- **Later sections**: ZeRO-2, ZeRO-3, FSDP for memory-efficient training

The next optimization combines the best of flat DDP (few large operations) with overlap DDP (hiding communication behind computation).