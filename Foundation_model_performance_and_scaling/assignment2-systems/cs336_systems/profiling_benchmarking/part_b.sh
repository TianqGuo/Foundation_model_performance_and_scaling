#!/bin/bash
# ==============================================================================
# Assignment Part (b): Forward and Backward Pass Timing
# ==============================================================================
#
# This script benchmarks forward and backward passes separately for all model
# sizes as required by the assignment.
#
# USAGE:
#   cd cs336_systems/profiling_benchmarking
#   ./part_b.sh
#
# CONFIGURATION:
#   - Warmup steps: 5 (as required)
#   - Measurement steps: 10 (as required)
#   - Models: small, medium, large, xl, 2.7B
#   - Context length: 512 (reduced for XL/2.7B if OOM)
#   - Batch size: 4 (reduced for XL/2.7B if OOM)
#
# OUTPUT:
#   - Results saved to: ../../results/part_b_results.csv
#   - Console output includes formatted table for writeup
#
# NOTES:
#   - XL and 2.7B models may fail with OOM on GPUs with <24GB VRAM
#   - If OOM occurs, manually run with reduced settings (see bottom of script)
#
# ==============================================================================

set -e

echo "=========================================="
echo "Assignment Part (b): Forward/Backward Timing"
echo "=========================================="
echo ""
echo "Models: small, medium, large, xl, 2.7B"
echo "Warmup: 5 steps | Measurement: 10 steps"
echo ""

# Create results directory
mkdir -p ../../results

# Run benchmark
uv run python -m cs336_systems.profiling_benchmarking.benchmark_separate \
    --model-sizes small medium large xl 2.7B \
    --context-length 512 \
    --batch-size 4 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --output ../../results/part_b_results.csv

echo ""
echo "=========================================="
echo "Part (b) Complete!"
echo "=========================================="
echo "Results: ../../results/part_b_results.csv"
echo ""
echo "If XL or 2.7B failed with OOM, run them separately:"
echo ""
echo "# XL model with reduced settings:"
echo "uv run python -m cs336_systems.profiling_benchmarking.benchmark \\"
echo "  --model-size xl --context-length 256 --batch-size 2 \\"
echo "  --pass-type separate --warmup-steps 5 --measure-steps 10"
echo ""
echo "# 2.7B model with reduced settings:"
echo "uv run python -m cs336_systems.profiling_benchmarking.benchmark \\"
echo "  --model-size 2.7B --context-length 128 --batch-size 1 \\"
echo "  --pass-type separate --warmup-steps 5 --measure-steps 10"
echo ""
