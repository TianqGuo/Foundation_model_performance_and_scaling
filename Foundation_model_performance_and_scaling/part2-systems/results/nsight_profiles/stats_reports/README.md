# Nsys Stats Reports

This directory contains exported statistics from nsys profiling runs.

## Directory Structure

Each subdirectory corresponds to a profiling task:
- `part_a/` - Forward pass timing validation
- `part_b_c/` - Kernel analysis (forward and forward+backward)
- `part_d/` - Training step analysis
- `part_e/` - Attention component analysis

## Report Types

For each `.nsys-rep` profile, we generate several report types:

### 1. `*_nvtx_sum.txt` - NVTX Range Summary
Shows timing for each NVTX annotated range (warmup, forward, backward, etc.)

**Use for**: Part (a) timing validation

### 2. `*_cuda_api_sum.txt` - CUDA API Call Summary
Shows all CUDA API calls (cudaLaunchKernel, cudaMalloc, etc.) with counts and times

**Use for**: Understanding CPU-side overhead

### 3. `*_cuda_gpu_kern_sum.txt` - GPU Kernel Summary ⭐ MOST IMPORTANT
Shows all GPU kernel executions with:
- Kernel name (e.g., `ampere_sgemm_128x128_nn`)
- Total time
- Number of calls
- Average time per call

**Use for**:
- Part (b): Finding most expensive kernel
- Part (c): Identifying non-matmul kernels
- Part (d): Comparing kernel distributions
- Part (e): Breaking down attention operations

### 4. `*_cuda_gpu_mem_time_sum.txt` - GPU Memory Time
Shows memory operations (copies, allocations) and their timing

### 5. `*_cuda_gpu_mem_size_sum.txt` - GPU Memory Size
Shows memory operation sizes

## How to Use These Reports

### Part (a): Forward Pass Timing
```bash
# Check NVTX summary for forward step timings
grep "forward_step" part_a/*_nvtx_sum.txt

# Compare with Python timing from:
cat ../part_b_results.csv
```

### Part (b): Most Expensive Kernel
```bash
# Look at kernel summary, sorted by total time
head -20 part_b_c/*forward_annotated_cuda_gpu_kern_sum.txt

# The first non-header line is usually the most expensive kernel
# Check "Instances" column for call count
```

### Part (c): Non-Matmul Kernels
```bash
# In kernel summary, look for kernels not containing "gemm" or "sgemm"
grep -v "gemm\|sgemm" part_b_c/*forward_annotated_cuda_gpu_kern_sum.txt | head -20

# Common non-matmul kernels:
# - *softmax*
# - *layernorm*
# - *dropout*
# - *reduce*
# - *elementwise*
```

### Part (d): Training vs Inference
```bash
# Compare kernel summaries:
# 1. Forward only: part_b_c/*forward_annotated_cuda_gpu_kern_sum.txt
# 2. Training: part_d/*training_step_cuda_gpu_kern_sum.txt

# Calculate % time in matmul vs other kernels for each
# Look for new kernels in training (AdamW optimizer kernels)
```

### Part (e): Attention Breakdown
```bash
# With annotated attention, filter by kernel regions:
cat part_e/*attention_analysis_cuda_gpu_kern_sum.txt

# Use Nsight GUI to correlate kernels with NVTX ranges:
# - "computing_attention_scores" → QK^T matmul kernels
# - "computing_softmax" → softmax kernels
# - "final_matmul" → attention @ V kernels
```

## Generating These Reports

Run on native Linux (H100/A100):
```bash
cd cs336_systems/nsight_systems_profiler
./export_stats_reports.sh
```

⚠️ Do NOT run on WSL2 - it won't have kernel data.

## File Sizes

Text reports are small (~5-50 KB each) and safe to commit to git,
unlike binary .nsys-rep files (10-50 MB each).
