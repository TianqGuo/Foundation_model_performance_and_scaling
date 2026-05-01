#!/bin/bash
# ==============================================================================
# Profile Naive vs Overlap DDP with Nsight Systems
# ==============================================================================
#
# This script profiles both naive and overlap DDP implementations to visualize
# whether communication overlaps with backward computation.
#
# REQUIREMENTS:
#   - NVIDIA Nsight Systems (nsys) must be installed
#   - 2+ GPUs
#
# USAGE:
#   ./run_profiling.sh
#
# OUTPUT:
#   - naive_ddp.nsys-rep: Profiling report for naive DDP
#   - overlap_ddp.nsys-rep: Profiling report for overlap DDP
#
# ==============================================================================

set -e

echo "================================================================================"
echo "DDP Profiling: Naive vs Overlap"
echo "================================================================================"
echo ""

# Check for nsys
if ! command -v nsys &> /dev/null; then
    echo "✗ Error: nsys (Nsight Systems) not found"
    echo "  Please install NVIDIA Nsight Systems:"
    echo "  https://developer.nvidia.com/nsight-systems"
    exit 1
fi

echo "✓ Found nsys: $(nsys --version | head -n 1)"
echo ""

# Check GPU count
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

if [ $NUM_GPUS -lt 2 ]; then
    echo "✗ Error: Need at least 2 GPUs for profiling"
    echo "  Found: $NUM_GPUS GPU(s)"
    exit 1
fi

echo "✓ Found $NUM_GPUS GPU(s)"
echo ""

# Profile naive DDP
echo "================================================================================"
echo "Profiling Naive DDP (no overlap)"
echo "================================================================================"
echo ""

nsys profile \
    -o naive_ddp \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    uv run python profile_overlap.py --implementation naive --num-steps 3

echo ""
echo "✓ Naive DDP profiling complete: naive_ddp.nsys-rep"
echo ""

# Profile overlap DDP
echo "================================================================================"
echo "Profiling Overlap DDP (with overlap)"
echo "================================================================================"
echo ""

nsys profile \
    -o overlap_ddp \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    uv run python profile_overlap.py --implementation overlap --num-steps 3

echo ""
echo "✓ Overlap DDP profiling complete: overlap_ddp.nsys-rep"
echo ""

# Summary
echo "================================================================================"
echo "Profiling Complete!"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  - naive_ddp.nsys-rep (no overlap)"
echo "  - overlap_ddp.nsys-rep (with overlap)"
echo ""
echo "Next steps:"
echo "  1. Open files in Nsight Systems GUI:"
echo "     nsys-ui naive_ddp.nsys-rep"
echo "     nsys-ui overlap_ddp.nsys-rep"
echo ""
echo "  2. What to look for in the timeline:"
echo ""
echo "     Naive DDP:"
echo "       - Backward pass completes first"
echo "       - All-reduce starts AFTER backward (sequential)"
echo "       - GPU is idle during communication"
echo ""
echo "     Overlap DDP:"
echo "       - NCCL operations start DURING backward pass"
echo "       - Communication and computation run in parallel"
echo "       - Look for overlapping CUDA kernels and NCCL calls"
echo ""
echo "  3. Take screenshots showing the difference for your report"
echo ""