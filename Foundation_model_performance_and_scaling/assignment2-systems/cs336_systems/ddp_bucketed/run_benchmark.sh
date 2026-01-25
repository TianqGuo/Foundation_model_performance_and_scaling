#!/bin/bash
# ==============================================================================
# Run Bucketed DDP Benchmark
# ==============================================================================
#
# Benchmarks bucketed DDP with various bucket sizes on XL model with 2 GPUs.
#
# USAGE:
#   ./run_benchmark.sh
#
# ==============================================================================

set -e

echo "================================================================================"
echo "Bucketed DDP Benchmark"
echo "================================================================================"
echo ""
echo "This benchmark tests DDP with different bucket sizes to find the optimal"
echo "trade-off between communication overhead and bucketing efficiency."
echo ""
echo "Configuration: XL model, 2 GPUs, bucket sizes: 1, 10, 100, 1000 MB"
echo ""

# Check GPU count
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

if [ $NUM_GPUS -lt 2 ]; then
    echo "✗ Error: Need at least 2 GPUs for benchmark"
    echo "  Found: $NUM_GPUS GPU(s)"
    exit 1
fi

echo "✓ Found $NUM_GPUS GPU(s)"
echo ""

# Run benchmark
echo "Running benchmark..."
echo "================================================================================"
echo ""

uv run python benchmark_bucketed.py \
    --model-size xl \
    --bucket-sizes 1 10 100 1000 \
    --num-steps 10 \
    --batch-size 2 \
    --warmup-steps 5 \
    --world-size 2

echo ""
echo "================================================================================"
echo "Benchmark complete!"
echo "================================================================================"