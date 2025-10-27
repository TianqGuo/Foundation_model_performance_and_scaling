# Benchmarking Results and Analysis

## Your Hardware

- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU
- **VRAM**: ~16GB (typical for laptop RTX 4090)

## Results from Your Run

### Successfully Benchmarked Models

| Model Size | Parameters | Context | Batch | Mean Time | Std Dev | CV%  |
|------------|------------|---------|-------|-----------|---------|------|
| small      | 128.6M     | 512     | 4     | 308.03 ms | 17.85 ms| 5.79%|
| medium     | 423.2M     | 512     | 4     | 928.49 ms | 9.42 ms | 1.01%|
| large      | 969.4M     | 512     | 4     | 17.8 s    | 1.04 s  | 5.84%|

### Failed Models (Out of Memory)

| Model Size | Parameters | Status |
|------------|------------|--------|
| xl         | ~2.0B      | OOM    |
| 2.7B       | ~2.7B      | OOM    |

## Analysis

### 1. Why XL and 2.7B Failed

The XL and 2.7B models are too large for your GPU VRAM with the default settings:

**Memory Requirements (rough estimates):**
- **Model parameters**:
  - XL: 2.0B params × 4 bytes (fp32) = 8 GB
  - 2.7B: 2.7B params × 4 bytes = 10.8 GB
- **Activations** (forward pass):
  - batch_size × context_length × d_model × num_layers × multiplier
  - For XL with bs=4, ctx=512: ~6-8 GB
- **Gradients** (backward pass): Similar to parameters
- **Optimizer states**: Not used in benchmark, but would be 2-3x parameters

**Total for XL**: ~20-25 GB (exceeds your 16GB VRAM)

### 2. Options to Benchmark Large Models

#### Option A: Reduce Context Length and Batch Size (Recommended)

```bash
# For XL model
uv run python -m cs336_systems.benchmark \
    --model-size xl \
    --context-length 256 \
    --batch-size 2 \
    --warmup-steps 5 \
    --measure-steps 10

# For 2.7B model
uv run python -m cs336_systems.benchmark \
    --model-size 2.7B \
    --context-length 128 \
    --batch-size 1 \
    --warmup-steps 5 \
    --measure-steps 10
```

#### Option B: Use Forward Pass Only

```bash
uv run python -m cs336_systems.benchmark \
    --model-size xl \
    --pass-type forward \
    --context-length 512 \
    --batch-size 2
```

#### Option C: Use Direct Benchmarking Script

We created `benchmark_direct.py` which avoids subprocess overhead:

```bash
uv run python -m cs336_systems.benchmark_direct \
    --model-sizes small medium large \
    --output results.csv
```

### 3. Performance Optimizations (For Future)

Currently your benchmarks are running in **FP32 (full precision)**. Here are optimizations you could enable:

#### Enable TF32 (Tensor Float 32)

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

- **Expected speedup**: 1.5-2x on Ampere/Ada GPUs (RTX 4090)
- **Accuracy impact**: Minimal (same as FP32 for most applications)

#### Use Mixed Precision (FP16/BF16)

```python
with torch.cuda.amp.autocast():
    output = model(input_ids)
```

- **Expected speedup**: 2-3x
- **Memory savings**: ~2x (can fit larger models)

#### Torch Compile (PyTorch 2.0+)

```python
model = torch.compile(model)
```

- **Expected speedup**: 1.3-2x after warmup
- **Note**: First run will be slow (compilation)

### 4. Variability Analysis

Looking at your results:

- **Small model**: CV = 5.79% (moderate variability)
- **Medium model**: CV = 1.01% (very stable)
- **Large model**: CV = 5.84% (moderate variability)

**Analysis**:
- Medium model shows excellent stability (CV < 2%)
- Small and large models show acceptable variability (CV < 6%)
- The higher variability might be due to:
  - GPU frequency scaling
  - Background processes
  - Thermal throttling (laptop GPU)

**Recommendations**:
- Your 5 warmup steps appear adequate
- Consider 15-20 measurement steps for more stable statistics
- Monitor GPU temperature during benchmarks

## Using the Benchmark Scripts

### Method 1: Individual Benchmarks

```bash
# Small model (fits easily)
uv run python -m cs336_systems.benchmark --model-size small

# Large model with reduced settings
uv run python -m cs336_systems.benchmark \
    --model-size xl \
    --context-length 256 \
    --batch-size 1
```

### Method 2: Direct Sweep (Recommended)

```bash
# Benchmark models that fit in VRAM
uv run python -m cs336_systems.benchmark_direct \
    --model-sizes small medium large \
    --warmup-steps 5 \
    --measure-steps 10 \
    --output results.csv
```

### Method 3: Subprocess Sweep (if you have time)

```bash
# Note: This is slower due to subprocess overhead
uv run python -m cs336_systems.run_benchmarks \
    --model-sizes small medium \
    --output results.csv
```

## Answering Assignment Questions

### Part (b): Forward and Backward Pass Timings

Based on your results:

**Forward + Backward Pass Times:**
- Small (128M params): ~308 ms
- Medium (423M params): ~928 ms
- Large (969M params): ~17.8 seconds

**Variability**: Low to moderate (CV: 1-6%), indicating stable measurements.

### Part (c): Effect of Warmup Steps

To test this, run:

```bash
# No warmup
uv run python -m cs336_systems.benchmark \
    --model-size small --warmup-steps 0 --measure-steps 10

# 1 warmup
uv run python -m cs336_systems.benchmark \
    --model-size small --warmup-steps 1 --measure-steps 10

# 2 warmup
uv run python -m cs336_systems.benchmark \
    --model-size small --warmup-steps 2 --measure-steps 10

# 5 warmup (baseline)
uv run python -m cs336_systems.benchmark \
    --model-size small --warmup-steps 5 --measure-steps 10
```

**Expected findings:**
- Without warmup: First few runs will be slower due to:
  - CUDA kernel compilation (JIT)
  - GPU memory allocation and caching
  - GPU frequency ramping up
- With 1-2 warmup: Still some variance from incomplete warm-up
- With 5+ warmup: Stable, representative timings

## Next Steps

1. ✅ Use `benchmark_direct.py` for reliable benchmarking
2. 📊 Generate tables for writeup with `--output results.csv`
3. 🔬 Run warmup experiments for part (c)
4. 📝 Document findings in writeup

For XL/2.7B models, note in your writeup that they require reduced batch size/context length due to VRAM constraints on consumer hardware.
