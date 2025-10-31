#!/bin/bash
# ==============================================================================
# Part (a): Profile forward pass and compare with Python timing
# ==============================================================================
#
# Question: What is the total time spent on your forward pass? Does it match
# what we measured before with the Python standard library?
#
# This script profiles forward-only passes for comparison with benchmarking results.
#
# USAGE:
#   cd cs336_systems/nsight_systems_profiler
#   ./profile_part_a.sh
#
# OUTPUT:
#   - Profiles saved to: ../../results/nsight_profiles/part_a/
#   - Compare nsys timings with results from part_b.sh
#
# ==============================================================================

set -e

echo "=========================================="
echo "Part (a): Forward Pass Profiling"
echo "=========================================="
echo ""

OUTPUT_DIR="../../results/nsight_profiles/part_a"
mkdir -p "$OUTPUT_DIR"

# Profile forward-only for all model sizes with ctx=512
# Assignment requires: Table 1 model sizes (small, medium, large, xl, 2.7B)
MODELS=("small" "medium" "large" "xl" "2.7B")

for model in "${MODELS[@]}"; do
    output="${OUTPUT_DIR}/${model}_forward_ctx512.nsys-rep"

    echo "Profiling $model model (forward only, ctx=512)..."

    uv run nsys profile \
        -o "$output" \
        --force-overwrite true \
        --trace=cuda,nvtx \
        --stats=true \
        python -m cs336_systems.nsight_systems_profiler.profile_model \
        --model-size "$model" \
        --context-length 512 \
        --batch-size 4 \
        --warmup-steps 5 \
        --measure-steps 10 \
        --profile-type forward

    echo "  ✓ Saved to: $output"
    echo ""
done

echo "=========================================="
echo "Part (a) Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Open .nsys-rep files in Nsight Systems GUI"
echo "  2. Look at total GPU time in the summary"
echo "  3. Compare with Python timings from: ../../results/part_b_results.csv"
echo ""
echo "Expected: nsys timing should match Python timing (within ~5%)"
echo ""
