# Distributed Communication Benchmarking

Implementation of Part 2.1.3: Benchmarking single-node distributed communication in PyTorch.

## Overview

This module benchmarks the all-reduce collective operation to understand the performance characteristics of distributed communication across different backends, data sizes, and process counts.

## Files

- `benchmark_allreduce.py` - Main benchmarking script (supports both CPU and GPU)
- `visualize_results.py` - Generate plots and analysis from benchmark results
- `run_benchmark_cpu.sh` - Run Gloo + CPU benchmarks (local testing)
- `run_benchmark_gpu.sh` - Run NCCL + GPU benchmarks (H100 cloud)
- `run_benchmark.sh` - Run full benchmark suite (auto-detects GPUs)
- `test_distributed_setup.py` - Simple test to verify distributed setup works

## Quick Start

### Recommended Workflow

**Step 1: Local Testing (CPU)**
```bash
cd cs336_systems/distributed_communication
./run_benchmark_cpu.sh
```

This runs Gloo + CPU benchmarks:
- Tests 2, 4, 6 processes
- Data sizes: 1MB, 10MB, 100MB, 1GB
- Saves to `../../results/distributed_communication/gloo_cpu_benchmark.csv`
- Runtime: ~3-5 minutes
- No GPU required

**Step 2: H100 Cloud (GPU)**

Copy module to H100 and run:
```bash
cd cs336_systems/distributed_communication
./run_benchmark_gpu.sh
```

This runs NCCL + GPU benchmarks:
- Tests 2, 4, 6 processes (auto-adjusts based on GPU count)
- Data sizes: 1MB, 10MB, 100MB, 1GB
- Saves to `../../results/distributed_communication/nccl_gpu_benchmark.csv`
- Runtime: ~3-5 minutes
- Requires 6 GPUs for full benchmark

**Step 3: Combine and Visualize**

Download GPU results and combine:
```bash
# If both files are in results/distributed_communication/, just run:
uv run python visualize_results.py

# Or specify paths explicitly:
uv run python visualize_results.py \
    --cpu-results ../../results/distributed_communication/gloo_cpu_benchmark.csv \
    --gpu-results ../../results/distributed_communication/nccl_gpu_benchmark.csv
```

This will:
- Combine CPU and GPU results into `combined_benchmark.csv`
- Generate performance plots (PNG files)
- Create summary tables (Markdown files)
- Produce text analysis of results

## Configuration Options

### Backends and Devices

- **Gloo + CPU**: CPU-based communication (for local development and CPU-only systems)
- **NCCL + GPU**: GPU-based communication (requires CUDA-capable GPUs)

### Data Sizes

Default: 1MB, 10MB, 100MB, 1GB (float32 tensors)

### Process Counts

Default: 2, 4, 6 processes

Automatically adjusted based on available GPU count:
- 6+ GPUs: Full benchmark (2, 4, 6 processes)
- 4-5 GPUs: Limited benchmark (2, 4 processes)
- 2-3 GPUs: Minimal benchmark (2 processes)
- 0-1 GPUs: CPU-only benchmark

## Custom Benchmarks

### Test Specific Configuration

```bash
# Test only NCCL+GPU with 2 and 4 processes
uv run python benchmark_allreduce.py \
    --backend nccl \
    --device cuda \
    --num-processes 2 4 \
    --data-sizes 10 100 1000

# Test only small data sizes
uv run python benchmark_allreduce.py \
    --data-sizes 1 10 \
    --output results/distributed_communication/small_sizes.csv
```

### Adjust Benchmarking Parameters

```bash
# More iterations for better statistics
uv run python benchmark_allreduce.py \
    --num-warmup 10 \
    --num-iters 200

# Quick test with fewer iterations
uv run python benchmark_allreduce.py \
    --num-warmup 2 \
    --num-iters 20
```

## Output Files

All outputs are saved to `../../results/distributed_communication/`:

**Benchmark Results:**
- `gloo_cpu_benchmark.csv` - CPU (Gloo) benchmark results
- `nccl_gpu_benchmark.csv` - GPU (NCCL) benchmark results
- `combined_benchmark.csv` - Combined CPU + GPU results (auto-generated)

**Visualizations:**
- `allreduce_time_vs_datasize.png` - Performance plots by data size
- `allreduce_bandwidth_vs_datasize.png` - Bandwidth plots
- `allreduce_backend_comparison_100mb.png` - Backend comparison at 100MB

**Analysis:**
- `full_results.md` - Full results table (markdown)
- `summary_by_config.md` - Summary statistics (markdown)
- `analysis.txt` - Text analysis and key observations

## Platform Notes

### WSL2 / Local Development

The scripts auto-detect WSL2 and available GPU count. Limited GPU configurations will be automatically selected based on hardware.

### H100 / Production

For full benchmarks with 6 GPUs, run on a system with sufficient GPU resources:

```bash
# Ensure all 6 GPUs are visible
nvidia-smi

# Run full benchmark
./run_benchmark.sh
```

## Implementation Details

### Benchmarking Best Practices

The implementation follows PyTorch distributed benchmarking best practices:

1. **Warm-up iterations**: 5 iterations before timing (especially important for NCCL)
2. **Synchronization**: `torch.cuda.synchronize()` called before/after timing for accurate GPU measurements
3. **Result aggregation**: Statistics collected from all ranks using `dist.all_gather_object`
4. **Process spawning**: Uses `mp.spawn` for clean process isolation
5. **Error handling**: Gracefully handles OOM and missing GPU scenarios

### Process Group Setup

```python
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)  # Each rank uses different GPU
```

### All-Reduce Operation

```python
# Warm-up
for _ in range(num_warmup):
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for GPU operations

# Timed iterations
for _ in range(num_iters):
    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()
    end = time.perf_counter()
```

## Troubleshooting

### Port already in use

If you see `Address already in use` errors, change the master port:

```bash
uv run python benchmark_allreduce.py --master-port 29501
```

### Not enough GPUs

The benchmark automatically skips configurations requiring more GPUs than available. Check error messages in output for details.

### NCCL not available

If NCCL backend fails, ensure:
- CUDA is properly installed
- PyTorch is built with CUDA support
- GPUs are visible to PyTorch (`torch.cuda.is_available()`)

## Requirements

See Assignment 2 Part 2.1.3:
- Up to 6 GPUs for full benchmark
- Each benchmarking run: <5 minutes
- Deliverable: Plots/tables comparing settings + 2-3 sentence commentary