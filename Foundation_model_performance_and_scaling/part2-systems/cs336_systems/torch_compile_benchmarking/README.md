# Torch.compile Benchmarking Module

Benchmarking tools for analyzing torch.compile performance on attention and full Transformer models (Assignment 2, Section 1.3.1).

## Overview

This module benchmarks PyTorch's JIT compiler (`torch.compile`) introduced in PyTorch 2.0. It compares vanilla PyTorch implementations against torch.compile optimized versions to measure the impact of automatic kernel fusion and optimization.

## Quick Start

```bash
cd cs336_systems/torch_compile_benchmarking

# Run all benchmarks (Parts a + b)
./run_all.sh

# Or run individual parts:
./run_part_a.sh  # Attention benchmarking
./run_part_b.sh  # Transformer benchmarking
```

## Files

### Python Scripts

- **`benchmark_compiled_attention.py`**: Attention kernel benchmarking
  - Compares vanilla vs torch.compile attention
  - Tests all (d_model, seq_len) combinations from 1.2.1
  - Measures forward/backward pass timings for both versions
  - Outputs side-by-side comparison and speedup ratios

- **`benchmark_compiled_transformer.py`**: End-to-end Transformer benchmarking
  - Compares vanilla vs torch.compile Transformer models
  - Tests different model sizes and context lengths
  - Measures three scenarios:
    - Forward-only pass
    - Forward+backward pass
    - Full training step (forward+backward+optimizer)

### Shell Scripts

- **`run_part_a.sh`**: Automated attention benchmarking (Section 1.3.1a)
- **`run_part_b.sh`**: Automated Transformer benchmarking (Section 1.3.1b)
- **`run_all.sh`**: Master script to run all benchmarks

## Configuration

### Part (a): Attention Benchmarking

Same configuration as Section 1.2.1:
- **Batch size**: 8 (fixed)
- **d_model**: [16, 32, 64, 128]
- **seq_len**: [256, 1024, 4096, 8192, 16384]
- **Warmup iterations**: 10
- **Measurement iterations**: 100

Tests BOTH vanilla and compiled versions for each configuration.

### Part (b): Transformer Benchmarking

- **Model sizes**: small, medium, large, xl, 2.7B (all from §1.1.2)
- **Context lengths**: 128, 256, 512, 1024 (same as §1.1.3)
- **Batch size**: 4
- **Warmup iterations**: 5
- **Measurement iterations**: 10

Tests BOTH vanilla and compiled versions for each configuration.

### Output

Results are saved to:
- `../../results/torch_compile_benchmarking/compiled_attention_benchmark.csv`
- `../../results/torch_compile_benchmarking/compiled_transformer_benchmark.csv`

## Usage Examples

### 1. Run All Benchmarks

```bash
./run_all.sh
```

This will:
1. Benchmark compiled vs vanilla attention (all configs)
2. Benchmark compiled vs vanilla Transformer (multiple sizes/contexts)
3. Generate CSV files with results
4. Print comparison tables

### 2. Attention Benchmarking Only

```bash
# Run all attention configs
./run_part_a.sh

# Or run specific configuration
python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_attention \
    --batch-size 8 \
    --d-model 64 \
    --seq-len 4096 \
    --num-warmup 10 \
    --num-iterations 100
```

### 3. Transformer Benchmarking Only

```bash
# Run all Transformer configs
./run_part_b.sh

# Or run specific configuration
python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_transformer \
    --model-size small \
    --context-length 512 \
    --warmup-steps 5 \
    --measure-steps 10
```

## Expected Output

### Attention Benchmark CSV

Columns:
- `batch_size`, `seq_len`, `d_model`: Configuration parameters
- `compiled`: Boolean indicating vanilla (False) or compiled (True)
- `forward_mean_ms`, `forward_std_ms`: Forward pass timing
- `backward_mean_ms`, `backward_std_ms`: Backward pass timing
- `total_mean_ms`: Combined time
- `peak_memory_mb`: Peak memory usage
- `status`: "success" or "OOM"

The script automatically computes speedup ratios and prints comparison tables.

### Transformer Benchmark CSV

Columns:
- `model_size`, `context_length`: Configuration parameters
- `num_params`, `num_params_millions`: Model size
- `forward_vanilla_ms`, `forward_compiled_ms`: Forward pass timings
- `forward_speedup`: Speedup ratio
- `fwd_bwd_vanilla_ms`, `fwd_bwd_compiled_ms`: Forward+backward timings
- `fwd_bwd_speedup`: Speedup ratio
- `full_step_vanilla_ms`, `full_step_compiled_ms`: Full training step timings
- `full_step_speedup`: Speedup ratio

## Assignment Deliverables (Section 1.3.1)

This module helps answer:

### Part (a): Compiled vs Vanilla Attention

**Deliverable**: Table comparing forward/backward timings for compiled vs uncompiled attention

Run `./run_part_a.sh` to generate:
- Forward pass comparison table (vanilla vs compiled)
- Backward pass comparison table (vanilla vs compiled)
- Speedup ratios for each (d_model, seq_len) pair

### Part (b): Compiled vs Vanilla Transformer

**Deliverable**: Table comparing vanilla vs compiled Transformer performance

Run `./run_part_b.sh` to generate:
- Forward-only pass comparison
- Forward+backward pass comparison
- Full training step comparison (forward+backward+optimizer)

**Key questions to answer:**
- How does forward pass performance change?
- What about forward+backward passes?
- Does the optimizer step affect overall speedup?

## Implementation Notes

### torch.compile Usage

The key pattern is simple:

```python
# Vanilla
attention_fn = naive_attention

# Compiled
attention_fn = torch.compile(naive_attention)

# Or for models
model = TransformerLM(...)
model_compiled = torch.compile(model)
```

### Important Considerations

1. **Compilation overhead**: First iteration after wrapping with `torch.compile` is slower (JIT compilation)
2. **Warmup required**: More warmup iterations needed for compiled versions to trigger compilation
3. **Memory usage**: Should be similar between compiled and vanilla
4. **Speedup variation**: Varies by workload size, operation type, and hardware

### What torch.compile Does

From the PyTorch tutorial:
- Analyzes your computation graph dynamically
- Generates fused Triton kernels automatically
- Optimizes memory access patterns
- Reduces kernel launch overhead
- May not always provide speedups (depends on workload)

### Expected Speedups

Typical observations:
- **Larger workloads**: Better speedups (more to optimize)
- **Matmul-heavy**: Good speedups from kernel fusion
- **Memory-bound**: Less benefit (memory bottleneck remains)
- **Small operations**: May be slower (compilation overhead)

## Requirements

- PyTorch 2.0+ (for torch.compile support)
- CUDA-capable GPU
- pandas, numpy
- cs336_basics package (for TransformerLM)

## Troubleshooting

**Issue**: `torch.compile` not available
**Solution**: Upgrade to PyTorch 2.0 or later:
```bash
pip install --upgrade torch
```

**Issue**: First iteration very slow
**Solution**: This is expected - torch.compile analyzes and compiles on first run. Use adequate warmup.

**Issue**: No speedup or slower performance
**Solution**:
- torch.compile doesn't always help (especially for small workloads)
- Try larger batch sizes or sequence lengths
- Check if memory-bound (torch.compile can't fix memory bottlenecks)

**Issue**: Compilation errors
**Solution**:
- Some operations may not be supported by torch.compile
- Try with simpler models first
- Check PyTorch version and CUDA compatibility

## Comparison with FlashAttention-2

Key differences:
- **torch.compile**: Automatic optimization, general-purpose, easy to use
- **FlashAttention-2**: Manual kernel, attention-specific, better memory efficiency

torch.compile can improve compute performance through kernel fusion, but doesn't address the fundamental seq_len² memory bottleneck that FlashAttention-2 solves through tiled computation.

## Next Steps

After running benchmarks:

1. **Analyze results**: Review CSV files and comparison tables
2. **Create writeup tables**: Use pandas to format for LaTeX/Markdown
3. **Discuss observations**:
   - Where did torch.compile help most?
   - Where was it ineffective?
   - How do speedups scale with model/sequence size?
4. **Compare approaches**: Relate to FlashAttention-2 benefits from 1.2
