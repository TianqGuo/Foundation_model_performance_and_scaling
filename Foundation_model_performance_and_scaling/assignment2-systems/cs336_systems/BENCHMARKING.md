# Benchmarking Guide

This directory contains benchmarking tools for profiling Transformer models.

## Files

- `benchmark.py`: Core benchmarking script for timing forward and backward passes
- `run_benchmarks.py`: Helper script for running benchmark sweeps across multiple configurations

## Quick Start

### Single Benchmark

Run a single benchmark for a specific model size:

```bash
# Benchmark small model with default settings
uv run python -m cs336_systems.benchmark --model-size small

# Benchmark with custom settings
uv run python -m cs336_systems.benchmark \
    --model-size medium \
    --context-length 512 \
    --batch-size 4 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --pass-type forward_backward
```

### Benchmark Sweep

Run benchmarks across all model sizes:

```bash
# Run sweep for all models
uv run python -m cs336_systems.run_benchmarks

# Run sweep with custom settings
uv run python -m cs336_systems.run_benchmarks \
    --model-sizes small medium large \
    --context-length 512 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --output results.csv
```

## Model Configurations

The following model sizes are supported (from Table 1 in the assignment):

| Size   | d_model | d_ff  | num_layers | num_heads | Parameters |
|--------|---------|-------|------------|-----------|------------|
| small  | 768     | 3072  | 12         | 12        | ~128M      |
| medium | 1024    | 4096  | 24         | 16        | ~354M      |
| large  | 1280    | 5120  | 36         | 20        | ~772M      |
| xl     | 1600    | 6400  | 48         | 25        | ~1.5B      |
| 2.7B   | 2560    | 10240 | 32         | 32        | ~2.7B      |

All models use:
- Vocabulary size: 10,000
- Default batch size: 4
- Default context length: 512

## Command-Line Arguments

### benchmark.py

- `--model-size`: Model size to benchmark (small, medium, large, xl, 2.7B)
- `--context-length`: Sequence length (default: 512)
- `--batch-size`: Batch size (default: 4)
- `--warmup-steps`: Number of warmup iterations before timing (default: 5)
- `--measure-steps`: Number of iterations to measure (default: 10)
- `--pass-type`: Type of pass to benchmark (forward or forward_backward)
- `--no-sync`: Disable `torch.cuda.synchronize()` (for testing)
- `--device`: Device to use (cuda or cpu)

### run_benchmarks.py

- `--model-sizes`: List of model sizes to benchmark
- `--context-length`: Sequence length
- `--batch-size`: Batch size
- `--warmup-steps`: Number of warmup steps
- `--measure-steps`: Number of measurement steps
- `--pass-type`: Type of pass to benchmark
- `--output`: Output CSV file path
- `--device`: Device to use

## Usage Examples

### Benchmark Forward Pass Only

```bash
uv run python -m cs336_systems.benchmark \
    --model-size small \
    --pass-type forward \
    --warmup-steps 5 \
    --measure-steps 10
```

### Benchmark Without Warmup (for comparison)

```bash
uv run python -m cs336_systems.benchmark \
    --model-size small \
    --warmup-steps 0 \
    --measure-steps 10
```

### Benchmark Different Context Lengths

```bash
for ctx_len in 128 256 512 1024; do
    uv run python -m cs336_systems.benchmark \
        --model-size small \
        --context-length $ctx_len \
        --output "results_ctx${ctx_len}.csv"
done
```

### Run on CPU (for testing)

```bash
uv run python -m cs336_systems.benchmark \
    --model-size small \
    --device cpu
```

## Output Format

The benchmark script outputs:
- Model configuration details
- Number of parameters
- Mean execution time (in milliseconds and seconds)
- Standard deviation of execution time
- Coefficient of variation (CV%)

Example output:
```
Benchmarking small model
  d_model: 768
  d_ff: 3072
  num_layers: 12
  num_heads: 12
  vocab_size: 10000
  batch_size: 4
  context_length: 512
  ...

Model has 128,625,408 parameters (128.63M)

Results:
  Mean time: 172.94 ms (0.172940 s)
  Std dev: 2.41 ms (0.002405 s)
  Coefficient of variation: 1.39%
```

## Implementation Details

### CUDA Synchronization

The benchmarking script uses `torch.cuda.synchronize()` after each forward/backward pass to ensure accurate timing. CUDA operations are asynchronous by default, so without synchronization, we would only measure the time to launch kernels, not their actual execution time.

### Warmup Steps

Warmup steps are critical for accurate benchmarking because:
1. Initial runs may include CUDA kernel compilation and optimization
2. GPU may need to reach steady-state temperature
3. Memory allocations and cache effects need to stabilize

The default of 5 warmup steps is typically sufficient for most cases.

### Timing

We use `timeit.default_timer()` which provides the system's highest-resolution clock for accurate timing measurements.

## Tips for Accurate Benchmarking

1. **Always use warmup steps**: At least 5 warmup iterations
2. **Run multiple measurements**: At least 10 measurement iterations for stable statistics
3. **Check coefficient of variation**: CV < 5% indicates stable measurements
4. **Close other applications**: Minimize background GPU usage
5. **Use consistent power settings**: Ensure GPU is not throttling

## Troubleshooting

### CUDA Out of Memory

If you get OOM errors with larger models:
- Reduce `--context-length`
- Reduce `--batch-size`
- Use a smaller model size

### High Variability

If you see high standard deviation:
- Increase `--warmup-steps`
- Increase `--measure-steps`
- Close background applications
- Check GPU temperature and throttling
