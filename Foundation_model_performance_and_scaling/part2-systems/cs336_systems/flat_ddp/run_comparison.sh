#!/bin/bash
# ==============================================================================
# Run DDP Implementation Comparison Benchmark
# ==============================================================================
#
# Compares naive DDP (individual parameter all-reduces) vs. flat DDP
# (single batched all-reduce) on the XL model with 2 GPUs.
#
# Usage:
#   ./run_comparison.sh
#
# ==============================================================================

set -e

echo "================================================================================"
echo "DDP Implementation Comparison"
echo "================================================================================"
echo ""
echo "This benchmark compares two DDP implementations:"
echo "  1. Naive DDP: All-reduces each parameter gradient individually"
echo "  2. Flat DDP: Flattens all gradients, single batched all-reduce"
echo ""
echo "Configuration: XL model, 2 GPUs, batch size 2 per GPU"
echo ""

# Check GPU count
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

if [ $NUM_GPUS -lt 2 ]; then
    echo "✗ Error: Need at least 2 GPUs for comparison"
    echo "  Found: $NUM_GPUS GPU(s)"
    exit 1
fi

echo "✓ Found $NUM_GPUS GPU(s)"
echo ""

# Run comparison benchmark
echo "Running comparison benchmark..."
echo "================================================================================"
echo ""

uv run python benchmark_comparison.py \
    --model-size xl \
    --num-steps 10 \
    --batch-size 2 \
    --warmup-steps 5 \
    --world-size 2

echo ""
echo "================================================================================"
echo "Benchmark complete!"
echo "================================================================================"