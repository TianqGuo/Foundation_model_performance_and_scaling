# Profiling and Benchmarking Module

This module contains all tools for benchmarking and profiling Transformer models for CS336 Assignment 2.

## Files

### Core Scripts
- **`benchmark.py`** - Main benchmarking module with model configs and timing functions
- **`benchmark_direct.py`** - Direct benchmarking (recommended) - runs sweeps without subprocess
- **`run_benchmarks.py`** - Subprocess-based benchmarking - slower but more isolated
- **`warmup_comparison.py`** - Analyzes effect of warmup steps (for assignment part c)
- **`benchmark_large_models.py`** - Helper for XL/2.7B models with reduced settings

### Documentation
- **`QUICK_START.md`** - Quick reference for common commands
- **`BENCHMARKING.md`** - Full technical documentation
- **`BENCHMARKING_RESULTS.md`** - Analysis and interpretation guide

## Quick Start

### Benchmark Models That Fit in VRAM

```bash
uv run python -m cs336_systems.profiling_benchmarking.benchmark_direct \
    --model-sizes small medium large \
    --warmup-steps 5 \
    --measure-steps 10
```

Results will be saved to `results/profiling_benchmarking_direct.csv` by default.

### Compare Warmup Settings (Assignment Part c)

```bash
uv run python -m cs336_systems.profiling_benchmarking.warmup_comparison \
    --model-size small \
    --measure-steps 10
```

Results will be saved to `results/profiling_benchmarking_warmup_comparison.csv` by default.

### Single Model Benchmark

```bash
uv run python -m cs336_systems.profiling_benchmarking.benchmark \
    --model-size small \
    --warmup-steps 5 \
    --measure-steps 10
```

## Module Structure

```
profiling_benchmarking/
├── __init__.py                    # Module exports
├── benchmark.py                   # Core benchmarking functions
├── benchmark_direct.py            # Direct sweep (recommended)
├── run_benchmarks.py              # Subprocess-based sweep
├── warmup_comparison.py           # Warmup analysis
├── benchmark_large_models.py      # Large model helper
├── README.md                      # This file
├── QUICK_START.md                 # Quick reference
├── BENCHMARKING.md                # Full documentation
└── BENCHMARKING_RESULTS.md        # Results analysis
```

## Usage from Python

```python
from cs336_systems.profiling_benchmarking import (
    MODEL_CONFIGS,
    create_model,
    generate_random_batch,
    benchmark_forward,
    benchmark_forward_backward,
)

# Create a model
config = MODEL_CONFIGS["small"]
model = create_model(config, context_length=512, device="cuda")

# Generate test data
input_ids = generate_random_batch(4, 512, 10000, device="cuda")

# Run benchmark
mean_time, std_time = benchmark_forward_backward(
    model, input_ids, warmup_steps=5, measure_steps=10
)

print(f"Mean: {mean_time*1000:.2f} ms")
```

## See Also

- `QUICK_START.md` for common commands
- `BENCHMARKING_RESULTS.md` for interpreting your results
- `BENCHMARKING.md` for implementation details
