# Quick Start Guide - Benchmarking

## Setup Complete ✅

The benchmarking infrastructure is ready to use. Your earlier run showed:
- ✅ Small, medium, large models work perfectly
- ⚠️ XL and 2.7B models need reduced settings (OOM on 16GB VRAM)

## Quick Commands

### 1. Benchmark Forward and Backward Passes Separately (Part b) ⭐

**This is what you need for assignment part (b):**

```bash
uv run python -m cs336_systems.profiling_benchmarking.benchmark_separate \
    --model-sizes small medium large \
    --warmup-steps 5 \
    --measure-steps 10
```

This will:
- Measure **forward pass** time separately
- Measure **backward pass** time separately
- Show you the ratio between them
- Generate a table ready for your writeup

Results saved to `results/profiling_benchmarking_separate.csv` by default.

### 2. Benchmark All Models (Combined Forward+Backward)

```bash
uv run python -m cs336_systems.profiling_benchmarking.benchmark_direct \
    --model-sizes small medium large \
    --warmup-steps 5 \
    --measure-steps 10
```

This measures forward+backward together (not separately). Results saved to `results/profiling_benchmarking_direct.csv`.

### 3. Compare Warmup Settings (Part c)

```bash
# This runs warmup experiments: 0, 1, 2, 5, 10 steps
uv run python -m cs336_systems.profiling_benchmarking.warmup_comparison \
    --model-size small \
    --measure-steps 10
```

Results will be saved to `results/profiling_benchmarking_warmup_comparison.csv` by default.

### 4. Benchmark Forward Pass Only

```bash
uv run python -m cs336_systems.profiling_benchmarking.benchmark \
    --model-size small \
    --pass-type forward \
    --warmup-steps 5 \
    --measure-steps 10
```

### 5. Benchmark Large Models with Reduced Settings

```bash
# XL model - reduced context and batch size
uv run python -m cs336_systems.profiling_benchmarking.benchmark \
    --model-size xl \
    --context-length 256 \
    --batch-size 2 \
    --warmup-steps 5 \
    --measure-steps 10

# 2.7B model - even more reduced
uv run python -m cs336_systems.profiling_benchmarking.benchmark \
    --model-size 2.7B \
    --context-length 128 \
    --batch-size 1 \
    --warmup-steps 5 \
    --measure-steps 10
```

## Files in This Module

### Core Scripts
- `benchmark.py` - Main benchmarking script
- `benchmark_direct.py` - Direct sweep (no subprocess)
- `run_benchmarks.py` - Subprocess-based sweep
- `warmup_comparison.py` - Warmup effect analysis
- `benchmark_large_models.py` - Large model helper

### Documentation
- `BENCHMARKING.md` - Full documentation
- `BENCHMARKING_RESULTS.md` - Analysis of your results
- `QUICK_START.md` - This file
- `README.md` - Module overview

## Your Current Results

From your successful run:

| Model  | Params | Time      | Std Dev  | CV%   |
|--------|--------|-----------|----------|-------|
| small  | 128M   | 308 ms    | 17.9 ms  | 5.79% |
| medium | 423M   | 928 ms    | 9.4 ms   | 1.01% |
| large  | 969M   | 17.8 s    | 1.04 s   | 5.84% |

These are good results with acceptable variability!

## Performance Optimizations (Optional)

### Enable TF32 (Recommended for RTX 4090)

Add this to your benchmark scripts:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Expected speedup: 1.5-2x with minimal accuracy impact.

### Use Mixed Precision

For even faster benchmarks:

```python
with torch.cuda.amp.autocast():
    output = model(input_ids)
```

Expected speedup: 2-3x and 2x memory savings.

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce `--context-length` and `--batch-size`

```bash
# Example for large models
--context-length 256 --batch-size 2
```

### High Variability (CV > 10%)

**Solution**: Increase warmup and measurement steps

```bash
--warmup-steps 10 --measure-steps 20
```

### Slow Performance

**Possible causes**:
1. GPU thermal throttling (laptop GPUs)
2. Background processes using GPU
3. Not using TF32/mixed precision

**Check GPU status**:
```bash
nvidia-smi
```

## Assignment Deliverables

### Part (b): Forward/Backward Timing ⭐

**Use the separate benchmarking script:**

```bash
uv run python -m cs336_systems.profiling_benchmarking.benchmark_separate \
    --model-sizes small medium large \
    --warmup-steps 5 \
    --measure-steps 10
```

This will:
1. Measure forward pass separately
2. Measure backward pass separately
3. Show standard deviations for both
4. Generate a markdown table ready for your writeup

Results saved to `results/profiling_benchmarking_separate.csv`.

**Sample output for your writeup:**
```
small model (128.6M parameters):
  Forward pass:  152.30 ± 8.45 ms (CV: 5.55%)
  Backward pass: 305.20 ± 12.30 ms (CV: 4.03%)
  Ratio: Backward is 2.00x forward
```

### Part (c): Warmup Effect

Run:
```bash
uv run python -m cs336_systems.profiling_benchmarking.warmup_comparison \
    --model-size small
```

Results will be saved to `results/profiling_benchmarking_warmup_comparison.csv`.

Expected findings:
- **No warmup (0)**: Slower first runs due to CUDA kernel compilation
- **1-2 warmup**: Still some variance
- **5+ warmup**: Stable performance

Write 2-3 sentences explaining:
1. Without warmup, results include GPU initialization overhead
2. CUDA kernels need JIT compilation on first run
3. 5 warmup steps appear sufficient based on stable CV%

## Next Steps

1. ✅ Run benchmarks for your writeup
2. 📊 Generate tables using the CSV outputs
3. 📝 Answer the written questions
4. 🚀 (Optional) Try optimizations like TF32

Good luck with your assignment!
