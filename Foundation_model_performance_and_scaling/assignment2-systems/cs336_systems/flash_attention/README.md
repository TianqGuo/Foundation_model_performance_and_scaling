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
- `flash_attention_triton.py` - Triton kernel implementation (forward in Triton, backward in PyTorch)
- `benchmark_flash_attention.py` - Benchmarking script comparing FlashAttention-2 vs PyTorch attention
- `run_benchmarks.sh` - Shell wrapper for running benchmarks
- `__init__.py` - Module exports
- `IMPLEMENTATION_NOTES.md` - Detailed implementation notes and test results

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

## Implementation Details

### PyTorch Version
- Tile sizes: 16×16 (minimum required)
- Uses online softmax for numerically stable computation
- Maintains running statistics (max, sum, output) per query tile
- Slower than Triton but useful for debugging

### Triton Version
- Tile sizes: 64×64 (tunable)
- Launch grid: (num_query_tiles, batch_size)
- Single fused kernel for all operations
- Uses float32 for on-chip accumulation buffers
- Supports causal masking via `is_causal` flag

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

## Performance Notes

- Triton implementation is significantly faster than PyTorch
- Memory usage scales linearly with sequence length
- Supports arbitrarily long sequences (limited by available memory)
- Causal masking adds minimal overhead

## References

- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.