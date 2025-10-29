#!/bin/bash
# ==============================================================================
# Part (d): Profile complete training step with AdamW optimizer
# ==============================================================================
#
# Question: How does the fraction of time spent on matrix multiplication change
# compared to inference (forward only)? How about other kernels?
#
# This script profiles a complete training step including optimizer.
#
# USAGE:
#   cd cs336_systems/nsight_systems_profiler
#   ./profile_part_d.sh
#
# OUTPUT:
#   - Training profile saved to: ../../results/nsight_profiles/part_d/
#
# ANALYSIS:
#   Compare kernel distributions:
#   - Forward only (inference)
#   - Forward + Backward + Optimizer (training)
#
# ==============================================================================

set -e

echo "=========================================="
echo "Part (d): Training Step Profiling"
echo "=========================================="
echo ""

OUTPUT_DIR="../../results/nsight_profiles/part_d"
mkdir -p "$OUTPUT_DIR"

MODEL="small"

echo "Profiling complete training step (forward + backward + optimizer)..."
echo ""

uv run nsys profile \
    -o "${OUTPUT_DIR}/${MODEL}_training_step.nsys-rep" \
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
    --profile-type training \
    --use-annotated-attention

echo ""
echo "=========================================="
echo "Part (d) Complete!"
echo "=========================================="
echo ""
echo "Analysis instructions:"
echo ""
echo "Compare with Part (b) forward-only profile:"
echo ""
echo "1. Open both profiles in Nsight Systems:"
echo "   - part_b_c/${MODEL}_forward_annotated.nsys-rep (inference)"
echo "   - part_d/${MODEL}_training_step.nsys-rep (training)"
echo ""
echo "2. In CUDA GPU Kernel Summary, calculate percentages:"
echo "   - Matrix multiply kernels (*gemm*)"
echo "   - Optimizer kernels (AdamW: *adam*, *mul*, *add*)"
echo "   - Other kernels (softmax, layernorm, etc.)"
echo ""
echo "3. Expected findings:"
echo "   - Training: Lower % for matmul (optimizer adds overhead)"
echo "   - Training: New kernels for optimizer operations"
echo "   - Training: More memory operations (loading/storing gradients)"
echo ""
echo "4. Filter by NVTX ranges to see:"
echo "   - 'forward' range"
echo "   - 'backward' range"
echo "   - 'optimizer_step' range"
echo ""
