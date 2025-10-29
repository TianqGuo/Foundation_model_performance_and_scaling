#!/bin/bash
# ==============================================================================
# Parts (b) & (c): Analyze CUDA kernels in forward and forward+backward passes
# ==============================================================================
#
# Part (b): What CUDA kernel takes the most cumulative GPU time during forward pass?
#           How many times is it invoked? Same kernel for forward+backward?
#
# Part (c): What other kernels besides matrix multiply account for non-trivial runtime?
#
# This script profiles both forward-only and forward+backward to compare kernels.
#
# USAGE:
#   cd cs336_systems/nsight_systems_profiler
#   ./profile_part_b_c.sh
#
# OUTPUT:
#   - Profiles with annotated attention saved to: ../../results/nsight_profiles/part_b_c/
#
# ANALYSIS:
#   In Nsight Systems GUI:
#   1. Open profile and go to "Stats System View"
#   2. Look at "CUDA GPU Kernel Summary"
#   3. Filter by NVTX ranges (forward vs backward)
#   4. Sort by "Total Time" to find most expensive kernels
#   5. Check "Instances" column for invocation count
#
# ==============================================================================

set -e

echo "=========================================="
echo "Parts (b) & (c): Kernel Analysis"
echo "=========================================="
echo ""

OUTPUT_DIR="../../results/nsight_profiles/part_b_c"
mkdir -p "$OUTPUT_DIR"

MODEL="small"  # Use small for quick analysis

echo "Profiling $MODEL model with annotated attention..."
echo ""

# Profile forward only
echo "1. Forward pass only..."
uv run nsys profile \
    -o "${OUTPUT_DIR}/${MODEL}_forward_annotated.nsys-rep" \
    --force-overwrite true \
     \
    --trace=cuda,nvtx --stats=true \
    --python-backtrace=cuda \
    python -m cs336_systems.nsight_systems_profiler.profile_model \
    --model-size "$MODEL" \
    --context-length 512 \
    --batch-size 4 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --profile-type forward \
    --use-annotated-attention

echo ""
echo "2. Forward + Backward..."
uv run nsys profile \
    -o "${OUTPUT_DIR}/${MODEL}_forward_backward_annotated.nsys-rep" \
    --force-overwrite true \
     \
    --trace=cuda,nvtx --stats=true \
    --python-backtrace=cuda \
    python -m cs336_systems.nsight_systems_profiler.profile_model \
    --model-size "$MODEL" \
    --context-length 512 \
    --batch-size 4 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --profile-type forward_backward \
    --use-annotated-attention

echo ""
echo "=========================================="
echo "Parts (b) & (c) Complete!"
echo "=========================================="
echo ""
echo "Analysis instructions:"
echo ""
echo "Part (b) - Most expensive kernel:"
echo "  1. Open profiles in Nsight Systems"
echo "  2. Stats System View → CUDA GPU Kernel Summary"
echo "  3. Sort by 'Total Time' (descending)"
echo "  4. Look for kernels like:"
echo "     - ampere_sgemm_* (matrix multiply)"
echo "     - volta_sgemm_* (older GPUs)"
echo "     - Check 'Instances' column for invocation count"
echo "  5. Filter by NVTX 'forward' range to exclude backward"
echo ""
echo "Part (c) - Non-matmul kernels:"
echo "  1. In kernel summary, look for kernels besides *gemm*"
echo "  2. Common kernels to look for:"
echo "     - softmax kernels"
echo "     - layernorm kernels"
echo "     - elementwise operations (add, mul, etc.)"
echo "     - dropout kernels"
echo "  3. Note their total time and percentage"
echo ""
