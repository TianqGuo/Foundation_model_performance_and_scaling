# Attention Benchmarking Module

Benchmarking tools for analyzing PyTorch attention implementations at different scales to understand memory and compute bottlenecks (Assignment 2, Section 1.2.1).

## Overview

This module benchmarks naive attention to identify the seq_len² memory bottleneck that FlashAttention-2 aims to solve. It measures forward/backward pass timings and memory usage across various configurations, and provides theoretical memory accounting.

## Quick Start

```bash
cd cs336_systems/attention_benchmarking

# Run full benchmark suite (all configurations)
./run_pytorch_benchmark.sh

# Run specific configuration
python -m cs336_systems.attention_benchmarking.benchmark_pytorch_attention \
    --batch-size 8 --d-model 64 --seq-len 4096

# Analyze memory for OOM configuration
python -m cs336_systems.attention_benchmarking.memory_accounting \
    --batch-size 8 --seq-len 8192 --d-model 16
```

## Files

### Python Scripts

- **`benchmark_pytorch_attention.py`**: Main benchmarking script
  - Tests attention at various (d_model, seq_len) combinations
  - Measures forward/backward pass timings
  - Tracks memory usage and identifies OOM cases
  - Outputs CSV with results

- **`memory_accounting.py`**: Theoretical memory analysis
  - Computes expected memory footprint for attention
  - Shows memory breakdown (inputs, scores, gradients)
  - Analyzes how memory scales with seq_len
  - Suggests mitigation strategies (FlashAttention-2)

### Shell Scripts

- **`run_pytorch_benchmark.sh`**: Automated benchmark runner
  - Runs all required configurations
  - Identifies smallest OOM case
  - Automatically runs memory accounting on OOM configuration
  - Provides next steps for writeup

## Configuration

### Benchmark Parameters (from REQUIREMENTS.md 1.2.1)

- **Batch size**: 8 (fixed)
- **d_model**: [16, 32, 64, 128]
- **seq_len**: [256, 1024, 4096, 8192, 16384]
- **Warmup iterations**: 10
- **Measurement iterations**: 100

### Output

Results are saved to:
- `../../results/attention_benchmarking/pytorch_attention_benchmark.csv`
- `../../results/attention_benchmarking/memory_analysis_d{D}_s{S}.txt`

## Usage Examples

### 1. Run Full Benchmark Suite

```bash
./run_pytorch_benchmark.sh
```

This will:
1. Test all 20 configurations (4 d_model × 5 seq_len)
2. Identify OOM cases
3. Run memory accounting on smallest OOM configuration
4. Save results to CSV

### 2. Test Specific Configuration

```bash
python -m cs336_systems.attention_benchmarking.benchmark_pytorch_attention \
    --batch-size 8 \
    --d-model 16 32 \
    --seq-len 256 1024 \
    --num-warmup 5 \
    --num-iterations 50 \
    --output results/attention_benchmarking/custom_test.csv
```

### 3. Memory Accounting Analysis

```bash
# Analyze specific configuration
python -m cs336_systems.attention_benchmarking.memory_accounting \
    --batch-size 8 \
    --seq-len 8192 \
    --d-model 16
```

This shows:
- Detailed memory breakdown (forward + backward)
- Memory scaling with seq_len
- Mitigation strategies

## Expected Output

### Benchmark Results CSV

Columns:
- `batch_size`, `seq_len`, `d_model`: Configuration parameters
- `forward_mean_ms`, `forward_std_ms`: Forward pass timing
- `backward_mean_ms`, `backward_std_ms`: Backward pass timing
- `total_mean_ms`: Combined time
- `memory_before_backward_mb`: Memory allocated before backward
- `peak_memory_mb`: Peak memory during execution
- `status`: "success" or "OOM"
- `error`: Error message if any

### Memory Analysis

Shows:
- Forward pass memory (inputs, scores, attention weights, output)
- Backward pass memory (gradients, saved tensors)
- Memory breakdown percentages
- Scaling analysis across seq_len values
- FlashAttention-2 mitigation strategies

## Assignment Deliverables (Section 1.2.1)

This module helps answer:

**(a) Benchmark timings table**: Run `./run_pytorch_benchmark.sh` to generate CSV with all configurations

**(b) OOM threshold**: Script automatically identifies smallest OOM case

**(c) Memory accounting**: Run `memory_accounting.py` for detailed analysis showing:
- Theoretical memory usage breakdown
- How backward memory scales with seq_len²
- Proposed mitigation (FlashAttention-2)

**(d) 1-2 paragraph response**: Use memory_accounting.py output to explain:
- What causes OOM (seq_len² attention scores)
- How backward memory changes with seq_len
- Elimination strategy (tiled attention, recomputation)

## Implementation Notes

### Key Features

1. **Proper GPU synchronization**: Calls `torch.cuda.synchronize()` after each iteration
2. **Memory tracking**: Uses `torch.cuda.max_memory_allocated()` to measure peak usage
3. **OOM handling**: Gracefully catches and reports out-of-memory errors
4. **Statistical rigor**: Multiple iterations with warmup for accurate measurements

### Attention Implementation

Uses naive attention following the formula:
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
```

This materialized the full seq_len × seq_len score matrix, causing the memory bottleneck.

### Memory Bottleneck

The key issue:
- **Forward**: Need to store attention scores (batch × seq_len × seq_len)
- **Backward**: Must save scores and weights for gradient computation
- **Total**: O(batch × seq_len²) memory

FlashAttention-2 solves this by:
- Computing attention in tiles
- Never materializing full score matrix
- Recomputing during backward (time vs memory tradeoff)
- Reducing to O(batch × seq_len) memory

## Requirements

- PyTorch with CUDA support
- pandas, numpy
- GPU with sufficient memory (16GB+ recommended)

## Troubleshooting

**Issue**: Import errors when running scripts
**Solution**: Install package in editable mode:
```bash
cd /path/to/assignment2-systems
pip install -e .
```

**Issue**: OOM on all configurations
**Solution**: Reduce batch size or test smaller seq_len values:
```bash
python -m cs336_systems.attention_benchmarking.benchmark_pytorch_attention \
    --batch-size 4 --seq-len 256 1024
```

**Issue**: Script fails with CUDA not available
**Solution**: Ensure GPU is accessible:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
