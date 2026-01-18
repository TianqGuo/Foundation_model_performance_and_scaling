# FlashAttention-2 Leaderboard Optimization

## Problem Identified

You were absolutely right! The original benchmark (`benchmark_flash_attention.py`) only tested up to `d_model=128`, but the **leaderboard configuration requires `d_model=1024`** (16 heads × 64 d_head).

## Leaderboard Configuration

```python
batch_size = 1
seq_len = 16384
d_model = 1024  # 16 heads × 64 d_head
dtype = bfloat16
is_causal = True
```

## Optimizations Applied

### 1. **Two-Pass Backward (NO ATOMICS!)**

**Problem**: The original backward kernel used `atomic_add` for dQ gradients, which causes:
- Thread contention and serialization
- Poor GPU utilization
- Slower performance

**Solution**: Split backward into two separate kernels:
- **Pass 1**: Compute dQ (launch grid over query tiles)
- **Pass 2**: Compute dK and dV (launch grid over key tiles)

**Benefit**: Each pass writes to independent memory locations → **no atomics needed!**

```python
# Pass 1: dQ (query-parallel)
flash_bwd_dq_kernel[grid_q](...)  # No atomics

# Pass 2: dK, dV (key-parallel)
flash_bwd_dkdv_kernel[grid_k](...)  # No atomics
```

### 2. **Autotune for Tile Sizes**

**Problem**: Fixed tile sizes (32×32) may not be optimal for all configurations.

**Solution**: Use Triton's `@triton.autotune` to automatically find best tile sizes:

```python
@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 128}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8),
    ],
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
```

**Benefit**: Triton benchmarks all configs and selects the fastest for each input shape.

### 3. **Early Termination for Causal Masking**

**Problem**: Processing tiles that are entirely masked (all zeros) wastes compute.

**Solution**:
- **Forward pass**: Only process key tiles up to current query position
- **Backward pass**: Skip query tiles that don't affect current key tile

```python
# Forward: limit key tiles based on query position
if is_causal:
    max_key_idx = query_start + Q_TILE_SIZE
    max_key_tiles = tl.cdiv(max_key_idx, K_TILE_SIZE)
    num_key_tiles = tl.minimum(num_key_tiles, max_key_tiles)

# Backward: skip irrelevant query tiles
if is_causal:
    query_start = i * Q_TILE_SIZE
    if query_start + Q_TILE_SIZE <= key_start:
        continue  # Skip this tile
```

**Benefit**: ~50% fewer tiles processed for causal attention!

### 4. **Larger Tile Sizes for d_model=1024**

**Problem**: Small tiles (32×32) underutilize GPU for large d_model.

**Solution**: Autotune tests up to 128×128 tiles, finding optimal size for d_model=1024.

**Benefit**: Better memory bandwidth utilization and compute efficiency.

## Files Created

1. **`flash_attention_triton_optimized.py`**
   - Optimized Triton kernels with two-pass backward
   - Autotune for tile sizes
   - Early termination for causal masking
   - Class: `FlashAttentionTritonOptimizedFunc`

2. **`benchmark_leaderboard.py`**
   - Benchmarks exact leaderboard configuration (seq_len=16384, d_model=1024)
   - Compares current vs optimized implementation
   - Shows speedup achieved

## How to Run

### Quick Test (Leaderboard Config Only)

```bash
cd cs336_systems/flash_attention
uv run python benchmark_leaderboard.py
```

This will:
1. Test current implementation (with atomics)
2. Test optimized implementation (two-pass, autotune, early termination)
3. Show comparison and speedup

### Expected Improvements

Based on the optimizations:

- **Forward pass**: ~1.5-2x faster (autotune + early termination)
- **Backward pass**: ~2-4x faster (no atomics + autotune + early termination)
- **Overall**: ~2-3x faster end-to-end

## Understanding the Results

From your CSV (d_model=128, seq_len=16384):
- Backward speedup: **0.43x** (PyTorch was faster!)
- This is because:
  1. Small d_model → small tiles → poor GPU utilization
  2. Atomic operations add significant overhead
  3. PyTorch can materialize full attention matrix in memory

With d_model=1024 and optimizations:
- Larger tiles → better GPU utilization
- No atomics → no contention
- Early termination → fewer tiles
- **Expected: 2-4x backward speedup**

## For Leaderboard Submission

After testing, you can use the optimized version for leaderboard submission:

```python
# In your submission code, replace:
from flash_attention_triton import FlashAttentionTritonFunc

# With:
from flash_attention_triton_optimized import FlashAttentionTritonOptimizedFunc as FlashAttentionTritonFunc
```

## Further Optimizations (Optional)

If needed for even better performance:

1. **Separate diagonal tiles** - Compute fully-masked and diagonal tiles separately
2. **TMA on H100** - Use Tensor Memory Accelerator for faster memory transfers
3. **Warp-level primitives** - Use warp shuffle for reductions
4. **Persistent kernels** - Keep kernels resident on GPU across launches

## Summary

You correctly identified that we were missing the crucial `d_model=1024` test! The optimized implementation should show significant improvements:

✅ **Two-pass backward** → No atomics, much faster
✅ **Autotune** → Best tile sizes automatically
✅ **Early termination** → Skip unnecessary work
✅ **Larger tiles** → Better GPU utilization for d_model=1024

Run `benchmark_leaderboard.py` on your H200 to see the actual speedup!