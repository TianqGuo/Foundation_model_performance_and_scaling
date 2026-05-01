#!/bin/bash
# ==============================================================================
# Assignment Part (c): Warmup Step Analysis
# ==============================================================================
#
# This script tests the effect of different warmup settings as required by
# the assignment.
#
# USAGE:
#   cd cs336_systems/profiling_benchmarking
#   ./part_c.sh
#
# WHAT IT DOES:
#   Tests warmup steps: 0, 1, 2, 5, 10
#   - 0 warmup: No warmup (includes CUDA initialization overhead)
#   - 1-2 warmup: Partial warmup (may still have variance)
#   - 5 warmup: Baseline (assignment requirement)
#   - 10 warmup: Extra warmup for comparison
#
# CONFIGURATION:
#   - Model: small (for quick testing)
#   - Measurement steps: 10
#   - Context length: 512
#   - Batch size: 4
#
# OUTPUT:
#   - Results saved to: ../../results/part_c_results.csv
#   - Console output shows comparison and analysis
#
# EXPECTED FINDINGS (for writeup):
#   - 0 warmup: Slower/higher variance (CUDA kernel compilation)
#   - 1-2 warmup: Still some variance
#   - 5+ warmup: Stable timings (sufficient warmup)
#
# EXPLANATION FOR WRITEUP:
#   Without warmup, the first runs include:
#   1. CUDA kernel JIT compilation overhead
#   2. GPU memory allocation and caching
#   3. GPU frequency ramping up
#   With 5+ warmup steps, these initialization costs are excluded,
#   giving stable and representative timings.
#
# ==============================================================================

set -e

echo "=========================================="
echo "Assignment Part (c): Warmup Effect Analysis"
echo "=========================================="
echo ""
echo "Testing warmup steps: 0, 1, 2, 5, 10"
echo "Model: small | Measurement: 10 steps"
echo ""

# Create results directory
mkdir -p ../../results

# Run warmup comparison
uv run python -m cs336_systems.profiling_benchmarking.warmup_comparison \
    --model-size small \
    --measure-steps 10 \
    --output ../../results/part_c_results.csv

echo ""
echo "=========================================="
echo "Part (c) Complete!"
echo "=========================================="
echo "Results: ../../results/part_c_results.csv"
echo ""
echo "For your writeup, explain:"
echo "  1. Without warmup, results include GPU initialization overhead"
echo "  2. CUDA kernels need JIT compilation on first run"
echo "  3. 5 warmup steps appear sufficient (check CV% in results)"
echo ""
echo "To test with other models (optional):"
echo "uv run python -m cs336_systems.profiling_benchmarking.warmup_comparison \\"
echo "  --model-size medium --output ../../results/part_c_medium.csv"
echo ""
