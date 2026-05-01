# Mixed Precision Training (CS336 Assignment 2 Section 1.1.5)

This module implements mixed precision training experiments for comparing FP32, FP16, and BF16 performance.

## Quick Start

```bash
cd cs336_systems/mixed_precision

# Part (a): Accumulation comparison
./run_accumulation.sh

# Part (b) Question (a): ToyModel dtype analysis
./run_toy_model.sh

# Part (b) Question (c): Benchmark all models
./run_benchmark.sh
```

## Files

### Python Scripts

- **`accumulation_comparison.py`** - Demonstrates why FP32 accumulation is important
  - Runs 4 accumulation variants with different dtype combinations
  - Shows numerical precision issues with FP16 accumulation
  - Used for Part (a) deliverable

- **`toy_model_dtypes.py`** - Analyzes data types in a toy model with FP16 autocast
  - Shows dtypes of parameters, activations, loss, and gradients
  - Explains why layer norm is kept in FP32
  - Used for Part (b) Questions (a) and (b)

- **`benchmark_mixed_precision.py`** - Benchmarks Transformer models with/without mixed precision
  - Compares FP32 vs BF16 (or FP16) performance
  - Benchmarks all model sizes from Table 1
  - Outputs CSV with timings and speedups
  - Used for Part (b) Question (c)

### Shell Scripts

- **`run_accumulation.sh`** - Runs accumulation comparison (Part a)
- **`run_toy_model.sh`** - Runs toy model analysis (Part b Question a)
- **`run_benchmark.sh`** - Runs full benchmark (Part b Question c)

## Requirements

- CUDA-capable GPU (20+ GB VRAM for all models, smaller GPUs automatically skip xl/2.7B)
- PyTorch with CUDA support
- BF16 support recommended (H100, A100) - falls back to FP16 if not available

## Usage Examples

### Part (a): Accumulation Comparison

```bash
# Run from module directory
./run_accumulation.sh

# Or run Python directly
python -m cs336_systems.mixed_precision.accumulation_comparison
```

**Deliverable**: 2-3 sentence response explaining accuracy differences across the 4 variants.

### Part (b) Question (a): ToyModel Data Types

```bash
# Run from module directory
./run_toy_model.sh

# Or run Python directly
python -m cs336_systems.mixed_precision.toy_model_dtypes
```

**Deliverable**: List data types for model parameters, fc1 output, ln output, logits, loss, and gradients.

### Part (b) Question (b): Layer Norm Sensitivity

This is answered by the toy model script output. The script explains:
- Why layer norm is sensitive to mixed precision
- Whether BF16 requires special handling vs FP16
- Numerical stability concerns

**Deliverable**: 2-3 sentence response about layer norm sensitivity and BF16 handling.

### Part (b) Question (c): Full Benchmarking

```bash
# Run from module directory (auto-saves to results/mixed_precision/)
./run_benchmark.sh

# Or run Python directly with custom options
python -m cs336_systems.mixed_precision.benchmark_mixed_precision \
    --all-models \
    --context-length 512 \
    --dtype bf16 \
    --output results/mixed_precision/mixed_precision_benchmark.csv

# Benchmark single model
python -m cs336_systems.mixed_precision.benchmark_mixed_precision \
    --model-size small \
    --context-length 512 \
    --use-mixed-precision
```

**Deliverable**: 2-3 sentence response with timings and commentary on trends as model size changes.

## Output

All results are saved to `../../results/mixed_precision/`:
- `mixed_precision_benchmark.csv` - Full benchmark results (Part b Question c)

Terminal output provides:
- Detailed timing breakdowns
- Speedup calculations
- Numerical analysis explanations

## Design Notes

### Why BF16?

BF16 (bfloat16) is preferred over FP16 for mixed precision because:
- Same dynamic range as FP32 (8-bit exponent)
- More stable training (no loss scaling needed)
- Supported by modern GPUs (A100, H100)
- Falls back to FP16 automatically if not supported

### Autocast Usage

The benchmarking uses `torch.autocast` with `contextlib.nullcontext` for clean comparison:

```python
if use_mixed_precision:
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    ctx = nullcontext()  # No-op context

with ctx:
    output = model(input)
```

### Memory Considerations

- Small GPU (<20 GB): Automatically skips xl/2.7B models
- Large GPU (≥20 GB): Runs all model sizes
- OOM handling: Catches and logs OOM errors, continues to next model

## Expected Results

### Accumulation Comparison

- Variant 1 (FP32+FP32): Most accurate (~0.0)
- Variant 2 (FP16+FP16): Worst error (~10.0)
- Variants 3 & 4 (FP32+FP16): Moderate error (~0.00244)

### Mixed Precision Speedup

Typical speedups on H100/A100:
- Small models: 1.2-1.5x
- Medium models: 1.5-2.0x
- Large models: 2.0-2.5x
- XL/2.7B models: 2.5-3.0x

Speedup increases with model size because:
- Larger matmuls benefit more from Tensor Cores
- Fixed overhead becomes proportionally smaller
- Memory bandwidth bottlenecks reduced

## Troubleshooting

**"CUDA not available"**: This module requires a GPU. Cannot run on CPU.

**"BF16 not supported"**: Older GPUs (pre-Ampere) don't support BF16. Script automatically uses FP16 instead.

**OOM errors**: Reduce `--context-length`, `--batch-size`, or skip large models.

**Line ending errors**: Run `sed -i 's/\r$//' *.sh` to fix CRLF to LF.
