# FlashAttention-2 Implementation

This module implements FlashAttention-2 [Dao, 2023] using both pure PyTorch and Triton kernels.

## Overview

FlashAttention-2 is an optimized attention mechanism that uses tiling, recomputation, and operator fusion to:
- Reduce memory usage from O(seq_len²) to O(seq_len)
- Minimize data transfers between HBM and SRAM
- Improve computational efficiency through kernel fusion

See `../../FLASHATTENTION2_BACKGROUND.md` for detailed algorithm explanations.

## Files

- `flash_attention_pytorch.py` - Pure PyTorch tiled implementation (forward + backward)
- `flash_attention_triton.py` - Baseline Triton kernel implementation (forward + backward in Triton)
- `flash_attention_triton_optimized.py` - Leaderboard-oriented Triton kernels (autotuned tiles, two-pass backward)
- `benchmark_flash_attention.py` - Benchmarking script comparing FlashAttention-2 vs PyTorch attention
- `benchmark_leaderboard.py` - Benchmarks the leaderboard configuration (seq_len=16k, d_model=1024)
- `run_benchmarks.sh` - Shell wrapper for running the full benchmark sweep
- `__init__.py` - Module exports
- `IMPLEMENTATION_NOTES.md` / `LEADERBOARD_OPTIMIZATION.md` - Implementation details and optimization notes

## Usage

### PyTorch Implementation

```python
from cs336_systems.flash_attention import FlashAttentionPyTorchFunc

# Inputs: (batch_size, seq_len, d_model)
Q = torch.randn(2, 512, 64, device='cuda')
K = torch.randn(2, 512, 64, device='cuda')
V = torch.randn(2, 512, 64, device='cuda')

# Forward pass
O = FlashAttentionPyTorchFunc.apply(Q, K, V, False)
```

### Triton Implementation

```python
from cs336_systems.flash_attention import FlashAttentionTritonFunc

# Same interface as PyTorch version
O = FlashAttentionTritonFunc.apply(Q, K, V, False)

# With causal masking
O = FlashAttentionTritonFunc.apply(Q, K, V, True)
```

## Testing

Run the official tests:

```bash
# From repository root
cd /path/to/assignment2-systems

# Test PyTorch forward pass
uv run pytest -k test_flash_forward_pass_pytorch -v

# Test Triton forward pass
uv run pytest -k test_flash_forward_pass_triton -v

# Test PyTorch backward pass
uv run pytest -k test_flash_backward_pytorch -v

# Test Triton backward pass
uv run pytest -k test_flash_backward_triton -v

# Test all FlashAttention tests
uv run pytest tests/test_attention.py -k "flash" -v
```

## Benchmarking

Compare FlashAttention-2 performance against standard PyTorch attention:

```bash
# From this directory
cd cs336_systems/flash_attention

# Run full benchmark suite (30-60 minutes on H100)
./run_benchmarks.sh

# Or run directly with custom output path
uv run python benchmark_flash_attention.py --output /path/to/output.csv
```

**Benchmark Configuration**:
- Sequence lengths: 128 to 65536 (powers of 2)
- Embedding dimensions: 16 to 128 (powers of 2)
- Precisions: bfloat16 and float32
- Batch size: 1, causal masking enabled
- Total: 80 configurations

**Output**: CSV file with forward, backward, and end-to-end latencies, plus speedup ratios

### Leaderboard Configuration (Optional §1.3.3/1.3.4)

To reproduce the leaderboard timings or compare the optimized kernels against the baseline implementation:

```bash
# From this directory
uv run python benchmark_leaderboard.py
```

This script benchmarks both `FlashAttentionTritonFunc` (baseline, atomics-based backward) and
`FlashAttentionTritonOptimizedFunc` (autotuned forward + two-pass backward) on the official configuration
`(batch_size=1, seq_len=16384, d_model=1024, dtype=torch.bfloat16, is_causal=True)` and prints the measured
latencies and speedups.

## Implementation Details

### PyTorch Version
- Tile sizes: 16×16 (minimum required)
- Uses online softmax for numerically stable computation
- Maintains running statistics (max, sum, output) per query tile
- Slower than Triton but useful for debugging

### Triton Version
- Tile sizes: 64×64 for the baseline implementation; the optimized kernel autotunes tile sizes per shape
- Launch grid: (num_query_tiles, batch_size) with Triton automatically inferring tile size in the optimized kernel
- Single fused kernel for forward pass; backward is in Triton for both variants (atomics vs. two-pass)
- Uses float32 for on-chip accumulation buffers
- Supports causal masking via `is_causal` flag and early-termination logic in the optimized kernels

## Key Features

### Tiling
- Processes attention in small tiles to fit in SRAM
- Query tiles: size B_q
- Key tiles: size B_k
- No need to materialize full attention matrix

### Recomputation (Backward Pass)
- Backward pass uses `torch.compile` for efficient gradient computation
- Saves only Q, K, V, O, and logsumexp L (linear memory)
- Recomputes attention scores P in backward pass from Q, K, L
- Eliminates need to store full attention matrix P (quadratic memory)
- Implements equations 13-19 from FlashAttention-2 paper

### Operator Fusion
- Single kernel performs all attention operations
- Minimal data transfers between HBM and SRAM
- Improved memory bandwidth utilization

## Leaderboard Optimizations Summary

- **Forward autotuning & early termination** (`flash_attention_triton_optimized.py`): Triton autotune selects tile sizes
  based on `(seq_len, d_model)` and causal attention skips fully-masked tiles, reducing redundant work for the
  16k-token sequence.
- **Two-pass backward (no atomics)**: `flash_bwd_dq_kernel` computes `dQ` per query tile, `flash_bwd_dkdv_kernel`
  computes `dK`/`dV` per key tile, eliminating atomic contention at the cost of recomputation as suggested in §1.3.4.
- **Leaderboard benchmark**: `benchmark_leaderboard.py` warms up and times both implementations on an H100. On an
  NVIDIA H100 80GB the optimized kernels currently achieve ~2.4× faster forward and ~1.1× faster backward passes
  compared to the baseline.

## Performance Notes

- Triton implementation is significantly faster than PyTorch
- Memory usage scales linearly with sequence length
- Supports arbitrarily long sequences (limited by available memory)
- Causal masking adds minimal overhead

## References

- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
