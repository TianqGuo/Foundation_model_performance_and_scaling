#!/bin/bash
# ==============================================================================
# Export nsys stats reports to text files for analysis
# ==============================================================================
#
# This script runs nsys stats commands to extract metrics from .nsys-rep files
# and saves them as text files that can be committed to git.
#
# IMPORTANT: This must run on NATIVE LINUX (H100/A100), NOT WSL2
#            WSL2 profiles don't contain GPU kernel data
#
# USAGE:
#   # On H100/A100 instance:
#   cd cs336_systems/nsight_systems_profiler
#   ./export_stats_reports.sh
#
# OUTPUT:
#   - Text reports in ../../results/nsight_profiles/stats_reports/
#   - Each profile gets multiple report types
#
# FIX NOTES (Oct 31, 2025):
#   - Added --force-export=true flag to handle stale .sqlite files
#   - Added cleanup step to delete old .sqlite files before processing
#   - This fixes the issue where ~350-byte placeholder files were created
#     when nsys stats refused to regenerate stale exports
#
# ==============================================================================

set -e

echo "=========================================="
echo "Exporting Nsys Stats Reports"
echo "=========================================="
echo ""

# Check if running in WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "WARNING: You appear to be running in WSL2"
    echo "WSL2 profiles don't contain GPU kernel data."
    echo "This script should be run on native Linux (H100/A100)."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

PROFILES_DIR="../../results/nsight_profiles"
STATS_DIR="${PROFILES_DIR}/stats_reports"
mkdir -p "$STATS_DIR"

# Clean up any stale .sqlite export files to avoid conflicts
echo "Cleaning up stale .sqlite export files..."
find "$PROFILES_DIR" -name "*.sqlite" -delete 2>/dev/null || true
echo "  ✓ Cleanup complete"
echo ""

# Report types to generate
REPORTS=(
    "nvtx_sum"                    # NVTX range summary
    "cuda_api_sum"                # CUDA API call summary
    "cuda_gpu_kern_sum"           # GPU kernel summary (MAIN ONE for parts b,c,d,e)
    "cuda_gpu_mem_time_sum"       # GPU memory operations time
    "cuda_gpu_mem_size_sum"       # GPU memory operations size
)

echo "Finding all .nsys-rep files..."
nsys_files=$(find "$PROFILES_DIR" -name "*.nsys-rep" | sort)

if [ -z "$nsys_files" ]; then
    echo "No .nsys-rep files found in $PROFILES_DIR"
    exit 1
fi

echo "Found $(echo "$nsys_files" | wc -l) profile(s)"
echo ""

# Process each profile
for nsys_file in $nsys_files; do
    # Get profile name
    profile_name=$(basename "$nsys_file" .nsys-rep)
    profile_dir=$(dirname "$nsys_file")
    rel_path=$(realpath --relative-to="$PROFILES_DIR" "$profile_dir")

    echo "Processing: $rel_path/$profile_name"

    # Create subdirectory for this profile's reports
    output_dir="${STATS_DIR}/${rel_path}"
    mkdir -p "$output_dir"

    # Generate each report type
    for report_type in "${REPORTS[@]}"; do
        output_file="${output_dir}/${profile_name}_${report_type}.txt"

        echo "  - Generating $report_type report..."

        nsys stats \
            --report "$report_type" \
            --force-export=true \
            "$nsys_file" \
            > "$output_file" 2>&1 || {
            echo "    WARNING: $report_type report failed (may be skipped if no data)"
        }
    done

    echo "  ✓ Reports saved to: $output_dir/"
    echo ""
done

echo "=========================================="
echo "Stats Export Complete!"
echo "=========================================="
echo ""
echo "Reports saved to: $STATS_DIR"
echo ""
echo "Key reports for each question:"
echo "  Part (a): *_nvtx_sum.txt - Forward pass timing"
echo "  Part (b): *_cuda_gpu_kern_sum.txt - Top kernels, call counts"
echo "  Part (c): *_cuda_gpu_kern_sum.txt - Non-matmul kernels"
echo "  Part (d): Compare forward vs training *_cuda_gpu_kern_sum.txt"
echo "  Part (e): attention_analysis_cuda_gpu_kern_sum.txt - Attention kernels"
echo ""
echo "These text files can be committed to git (unlike binary .nsys-rep files)."
echo ""

# Create a summary document
SUMMARY_FILE="${STATS_DIR}/README.md"
cat > "$SUMMARY_FILE" << 'EOF'
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
EOF

echo "Created README: $SUMMARY_FILE"
echo ""
