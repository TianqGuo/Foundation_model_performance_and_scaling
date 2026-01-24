#!/bin/bash
# ==============================================================================
# Run DDP Overlap Benchmark
# ==============================================================================
#
# Benchmarks DDP with overlapping computation and communication on XL model
# with 2 GPUs.
#
# Usage:
#   ./run_benchmark.sh
#
# ==============================================================================

set -e

echo "================================================================================"
echo "DDP with Overlapping Computation and Communication Benchmark"
echo "================================================================================"
echo ""
echo "This benchmark measures DDP performance when overlapping backward"
echo "computation with gradient communication using async all-reduce."
echo ""
echo "Configuration: XL model, 2 GPUs, batch size 2 per GPU"
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

uv run python benchmark_overlap.py \
    --model-size xl \
    --num-steps 10 \
    --batch-size 2 \
    --warmup-steps 5 \
    --world-size 2

echo ""
echo "================================================================================"
echo "Benchmark complete!"
echo "================================================================================"