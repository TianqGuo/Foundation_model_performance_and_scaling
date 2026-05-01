#!/bin/bash
# ==============================================================================
# Benchmark Bucketed DDP - Section 2.3.3 Part (a)
# ==============================================================================
#
# DESCRIPTION:
#   Benchmarks DDP with bucketed gradients using different bucket sizes
#   (1, 10, 100, 1000 MB) on XL model with 2 GPUs.
#
# REQUIREMENTS:
#   - 2+ GPUs (tested on H100s)
#   - XL model requires ~40GB GPU memory
#   - Run on remote multi-GPU instance (NOT local laptop)
#
# USAGE:
#   cd cs336_systems/ddp_bucketed
#   bash benchmark_part_a.sh
#
# OUTPUT:
#   results/ddp_bucketed/bucket_size_comparison.csv
#
# ==============================================================================

set -e

echo "================================================================================"
echo "Bucketed DDP Benchmark - Section 2.3.3 Part (a)"
echo "================================================================================"
echo ""
echo "This benchmark tests DDP with different bucket sizes to find the optimal"
echo "trade-off between communication overhead and bucketing efficiency."
echo ""
echo "Configuration: XL model, 2 GPUs, bucket sizes: 1, 10, 100, 1000 MB"
echo ""

# Check GPU count
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ Error: nvidia-smi not found. CUDA not available."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "✗ Error: Need at least 2 GPUs for multi-GPU benchmark"
    echo "  Found: $NUM_GPUS GPU(s)"
    echo ""
    echo "  This benchmark requires 2+ GPUs and should be run on a remote"
    echo "  multi-GPU instance (e.g., H100s on vast.ai/lambda.ai)."
    echo ""
    echo "  For local testing (single GPU), run unit tests instead:"
    echo "    uv run pytest tests/test_ddp.py -v"
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