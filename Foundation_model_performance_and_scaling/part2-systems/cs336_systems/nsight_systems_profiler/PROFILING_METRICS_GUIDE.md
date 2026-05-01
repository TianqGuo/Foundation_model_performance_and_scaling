# Nsight Profiling Metrics Guide

This document explains what metrics we need to answer assignment questions and how to extract them.

## Environment Note

**Profiles were generated on**: Lightning AI H100 instance (Ubuntu, native Linux)
**Local testing done on**: WSL2 with RTX 4090 (limited profiling capabilities)

All extraction scripts work on **both** environments, but H100 profiles contain complete kernel data while WSL2 profiles have limitations.

## Assignment Questions & Required Metrics

### Part (a): Forward Pass Timing Comparison
**Question**: What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

**Required Metrics**:
- Total forward pass time (milliseconds)
- Comparison with Python `timeit` results

**How to Extract**:
```bash
# Method 1: From NVTX ranges in sqlite
python analyze_wsl_profiles.py results/nsight_profiles/part_a/small_forward_ctx512.sqlite

# Method 2: From nsys CLI stats
nsys stats --report nvtx_sum results/nsight_profiles/part_a/small_forward_ctx512.nsys-rep
```

**Current Status**: ✅ **SUFFICIENT**
- `analyze_wsl_profiles.py` extracts NVTX forward step timings
- `ANALYSIS_SUMMARY.txt` contains average forward pass times
- Can compare with `part_b_results.csv` from Python benchmarking

---

### Part (b): Most Expensive CUDA Kernel
**Question**: What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked? Is it the same kernel for forward+backward?

**Required Metrics**:
- Kernel name (e.g., `ampere_sgemm_128x32_nn`, `volta_scudnn_winograd_128x128`)
- Total time spent in this kernel (milliseconds)
- Number of invocations (call count)
- Comparison between forward-only vs forward+backward

**How to Extract**:
```bash
# Method 1: nsys CLI (native Linux only - not WSL2)
nsys stats --report cuda_gpu_kern_sum results/nsight_profiles/part_b_c/small_forward_annotated.nsys-rep

# Method 2: Nsight Systems GUI
# 1. Open .nsys-rep file in GUI
# 2. Go to "Stats System View" → "CUDA GPU Kernel Summary"
# 3. Sort by "Total Time" (descending)
# 4. Filter by NVTX range "forward" to exclude warmup/backward

# Method 3: SQL query on .sqlite (limited in WSL2)
sqlite3 small_forward_annotated.sqlite "
SELECT name, COUNT(*) as calls, SUM(end-start)/1e6 as total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY name
ORDER BY total_ms DESC
LIMIT 10;
"
```

**Current Status**: ✅ **Available from H100 profiles**
- `analyze_wsl_profiles.py` cannot extract kernel names (limitation of the script)
- `ANALYSIS_SUMMARY.txt` shows CUDA API calls but not GPU kernel execution
- **Use `export_stats_reports.sh` on H100 to extract kernel data OR use Nsight GUI**

**What's Missing**:
```
Expected output (from H100/native Linux):
  1. ampere_sgemm_128x128_nn
     Total time: 450.23 ms
     Calls: 1,440
     Avg time: 0.31 ms
```

---

### Part (c): Non-Matmul Kernels
**Question**: What other kernels besides matrix multiplies account for non-trivial CUDA runtime in the forward pass?

**Required Metrics**:
- Kernel names for top 5-10 kernels (excluding *gemm*)
- Time spent in each kernel
- Percentage of total runtime

**How to Extract**:
```bash
# In Nsight GUI:
# 1. Open "CUDA GPU Kernel Summary"
# 2. Sort by "Total Time"
# 3. Look for kernels NOT containing "gemm" or "sgemm"
# 4. Common ones: *softmax*, *layernorm*, *dropout*, *add*, *reduce*

# Example expected kernels:
# - softmax_warp_forward
# - vectorized_layernorm_kernel
# - dropout_kernel
# - reduce_sum_kernel
```

**Current Status**: ✅ **Available from H100 profiles**
- Use `export_stats_reports.sh` or Nsight GUI

---

### Part (d): Training vs Inference Kernel Distribution
**Question**: How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

**Required Metrics**:
- Forward-only: % time in matmul kernels, % time in other kernels
- Training (forward+backward+optimizer): % time breakdown
- Comparison table

**How to Extract**:
```bash
# On H100: Generate kernel reports
./export_stats_reports.sh

# Compare:
# 1. part_b_c/small_forward_annotated_cuda_gpu_kern_sum.txt
# 2. part_d/small_training_step_cuda_gpu_kern_sum.txt

# Calculate % for matmul kernels (*gemm*):
# matmul_time / total_kernel_time * 100%
```

**Expected Format**:
```
Inference (forward only):
  - Matmul kernels: 78% of GPU time
  - Softmax/layernorm: 12%
  - Other: 10%

Training (forward+backward+optimizer):
  - Matmul kernels: 65% of GPU time  (lower %)
  - Optimizer kernels (AdamW): 15%
  - Softmax/layernorm: 11%
  - Other: 9%
```

**Current Status**: ✅ **Available from H100 profiles**

---

### Part (e): Softmax vs Matmul in Attention
**Question**: Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

**Required Metrics**:
- Attention QK^T matmul time
- Softmax time
- Attention @ V matmul time
- FLOP counts for each operation
- Runtime ratio vs FLOP ratio

**How to Extract**:
```bash
# Profiles already generated from H100
# Extract stats:
./export_stats_reports.sh

# Or use Nsight GUI:
# 1. Find NVTX range "scaled_dot_product_attention"
# 2. Inside it, find sub-ranges:
#    - "computing_attention_scores" → QK^T matmul
#    - "computing_softmax" → softmax
#    - "final_matmul" → attention @ V
# 3. Look at CUDA kernels under each range
# 4. Sum kernel times for each operation
```

**FLOP Calculations** (already in `profile_part_e.sh`):
```python
# For small model, ctx=512, batch=4, d_model=768, num_heads=12, d_head=64
QK_matmul_flops = 2 * batch * num_heads * seq_len^2 * d_head
                = 2 * 4 * 12 * 512^2 * 64
                = 804,257,792 FLOPs

softmax_flops = 5 * batch * num_heads * seq_len^2  (approx)
              = 5 * 4 * 12 * 512^2
              = 62,914,560 FLOPs

attention_V_flops = 2 * batch * num_heads * seq_len^2 * d_head
                  = 804,257,792 FLOPs
```

**Expected Analysis**:
```
Runtime:
  - QK matmul: 15 ms
  - Softmax: 8 ms
  - Attention @ V: 15 ms

FLOPs:
  - QK matmul: 804M FLOPs
  - Softmax: 63M FLOPs (12.8x less)
  - Attention @ V: 804M FLOPs

Observation:
  - Softmax has 12.8x fewer FLOPs but takes 0.53x the time of matmul
  - Softmax is memory-bound (low arithmetic intensity)
  - Matmul is compute-bound (high arithmetic intensity)
  - Ratio shows softmax underutilizes GPU compute
```

**Current Status**: ✅ **Available from H100 profiles**
- NVTX annotations captured correctly
- Kernel data available in H100 profiles
- Use `export_stats_reports.sh` or Nsight GUI

---

## Tools Available

### 1. `analyze_wsl_profiles.py` (✅ Portable)
**What it extracts**:
- NVTX range timings (warmup, forward steps, backward steps)
- CUDA API call statistics (cudaLaunchKernel, cudaMalloc, etc.)
- Average forward/backward times

**Limitations**:
- Does not extract GPU kernel names/times (by design - focused on NVTX)
- For kernel data, use `export_stats_reports.sh` or Nsight GUI

**Usage**:
```bash
# Works on both local and H100
python analyze_wsl_profiles.py <sqlite_file>
```

### 2. `extract_all_analyses.sh` (✅ Portable)
**What it does**:
- Runs `analyze_wsl_profiles.py` on all profiles
- Creates `ANALYSIS_SUMMARY.txt` with NVTX timing results
- Suitable for answering Part (a)

**Usage**:
```bash
# Works on both local and H100
cd cs336_systems/nsight_systems_profiler
./extract_all_analyses.sh
```

### 3. `export_stats_reports.sh` (✅ Best on H100)
**What it does**:
- Runs `nsys stats` commands to extract kernel data
- Creates text reports for each profile
- Extracts data for parts (b), (c), (d), (e)

**Environment notes**:
- ✅ **H100 profiles**: Full kernel data available
- ⚠️ **Local WSL2 profiles**: May show "SKIPPED" if kernel data not captured

**Usage** (on H100 after profiling):
```bash
cd cs336_systems/nsight_systems_profiler
./export_stats_reports.sh

# Creates: results/nsight_profiles/stats_reports/
# - *_cuda_gpu_kern_sum.txt  (kernel statistics)
# - *_nvtx_sum.txt  (NVTX timing)
# - *_cuda_api_sum.txt  (API calls)
```

### 4. Nsight Systems GUI (✅ Works - Recommended for detailed analysis)
**How to use**:
1. Download `.nsys-rep` files from H100/compute center
2. Open on Windows machine with Nsight Systems GUI
3. Navigate to "Stats System View" → "CUDA GPU Kernel Summary"
4. Filter by NVTX ranges
5. Export data or take screenshots

**What you can see**:
- ✅ All kernel names
- ✅ Kernel execution times
- ✅ Call counts
- ✅ Memory operations
- ✅ Filter by NVTX ranges
- ✅ Timeline view

---

## Recommended Workflow

### Environment-Specific Workflow

**On H100 (Production)**:
```bash
# 1. Run profiling
cd cs336_systems/nsight_systems_profiler
./profile_part_a.sh
./profile_part_b_c.sh
./profile_part_d.sh
./profile_part_e.sh

# 2. Extract text summaries
./extract_all_analyses.sh          # NVTX timing
./export_stats_reports.sh          # Kernel statistics

# 3. Download only text summaries to local
# From local machine:
scp h100:~/assignment2-systems/results/nsight_profiles/ANALYSIS_SUMMARY.txt ./results/nsight_profiles/
scp -r h100:~/assignment2-systems/results/nsight_profiles/stats_reports/ ./results/nsight_profiles/
```

**On Local WSL2 (Testing)**:
```bash
# 1. Test profiling scripts (small model only)
cd cs336_systems/nsight_systems_profiler
./profile_part_a.sh  # May have limited kernel data

# 2. Extract what's available
./extract_all_analyses.sh  # NVTX timing works

# 3. Verify scripts run without errors before deploying to H100
```

### For Part (a) - Timing Comparison
```bash
# Check NVTX timing summary:
cat results/nsight_profiles/ANALYSIS_SUMMARY.txt

# Compare with Python benchmarking:
cat results/part_b_results.csv

# Both files work on local and H100
```

### For Parts (b)-(e) - Kernel Analysis
**Option 1: Use exported text reports** (Recommended for automation)
```bash
# On H100:
./export_stats_reports.sh

# Review generated reports:
cat results/nsight_profiles/stats_reports/part_b_c/*_cuda_gpu_kern_sum.txt
```

**Option 2: Use Nsight GUI** (Recommended for visual analysis)
1. Download `.nsys-rep` files from H100 to local Windows machine
2. Open in Nsight Systems GUI
3. Manually extract metrics
4. Document findings in text file

---

## What We Need to Add

### Script to Extract Kernel Data (if available)
```python
# analyze_kernel_stats.py - To be created if H100 sqlite has kernel data
```

This would query:
- `CUPTI_ACTIVITY_KIND_KERNEL` table for kernel execution data
- Group by kernel name
- Calculate totals, averages, call counts
- Filter by NVTX ranges for forward vs backward

### Enhanced Analysis Summary
Should include:
- Top 10 kernels by time (for part b)
- Non-matmul kernels breakdown (for part c)
- Kernel distribution comparison (for part d)
- Attention component breakdown (for part e)

---

## Current Status Summary

| Question | Metrics Needed | Currently Available from H100 | Status |
|----------|---------------|-------------------------------|--------|
| Part (a) | Forward pass timing | ✅ NVTX ranges in ANALYSIS_SUMMARY.txt | ✅ Complete |
| Part (b) | Top kernel, call count | ✅ H100 profiles have kernel data | ⚠️ Need to run export_stats_reports.sh or use GUI |
| Part (c) | Non-matmul kernels | ✅ H100 profiles have kernel data | ⚠️ Need to run export_stats_reports.sh or use GUI |
| Part (d) | Kernel distribution | ✅ H100 profiles have kernel data | ⚠️ Need to run export_stats_reports.sh or use GUI |
| Part (e) | Attention breakdown | ✅ NVTX ranges + kernel data in H100 profiles | ⚠️ Need to run export_stats_reports.sh or use GUI |

**Bottom Line**:
- ✅ Part (a) can be answered NOW with `ANALYSIS_SUMMARY.txt`
- ⚠️ Parts (b)-(e) require running `export_stats_reports.sh` on H100 OR using Nsight GUI
- The H100 profiles ARE complete and contain all needed data

---

## Next Steps

### Option 1: Automated Text Extraction (Recommended)
```bash
# On your H100 instance:
cd cs336_systems/nsight_systems_profiler
./export_stats_reports.sh

# This creates text reports in: results/nsight_profiles/stats_reports/
# Download these text files to local:
scp -r h100:~/assignment2-systems/results/nsight_profiles/stats_reports/ ./results/nsight_profiles/

# Commit only text reports to git (NOT binary .nsys-rep files)
```

### Option 2: Visual Analysis with GUI
1. Download `.nsys-rep` files from H100 to local Windows
2. Install Nsight Systems GUI: https://developer.nvidia.com/nsight-systems
3. Open profiles and extract metrics manually
4. Document findings in text file

### All Scripts Are Portable

All scripts in `cs336_systems/nsight_systems_profiler/` work on **both** environments:
- ✅ Local WSL2 (testing)
- ✅ H100 Lightning AI (production)

The only difference is data completeness:
- Local WSL2 profiles may lack kernel data
- H100 profiles contain complete data
