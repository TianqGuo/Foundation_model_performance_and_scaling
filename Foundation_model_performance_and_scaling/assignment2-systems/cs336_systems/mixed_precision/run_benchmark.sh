#!/bin/bash
# ==============================================================================
# CS336 Assignment 2 - Section 1.1.5 Part (b) Question (c): Mixed Precision Benchmark
# ==============================================================================
#
# This script benchmarks all model sizes with FP32 vs BF16 mixed precision.
# It compares forward/backward pass timings and calculates speedups.
#
# USAGE:
#   cd cs336_systems/mixed_precision
#   ./run_benchmark.sh
#
# OUTPUT:
#   - CSV file: ../../results/mixed_precision/mixed_precision_benchmark.csv
#   - Terminal output with timings and speedups
#
# ASSIGNMENT DELIVERABLE:
#   Part (b) Question (c): 2-3 sentences with timings and commentary on trends
#
# REQUIREMENTS:
#   - CUDA-capable GPU (20+ GB VRAM for all models)
#   - For smaller GPUs, script automatically skips xl/2.7B models
#
# NOTES:
#   - Each model takes 1-3 minutes to benchmark
#   - Total runtime: 10-20 minutes for all models
#   - Results automatically saved to CSV
#
# ==============================================================================

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "CS336 Assignment 2 Section 1.1.5 Part (b) Question (c)"
echo "Mixed Precision Benchmarking"
echo "=========================================="
echo ""

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Create output directory
OUTPUT_DIR="../../results/mixed_precision"
mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE="$OUTPUT_DIR/mixed_precision_benchmark.csv"

echo "Output file: $OUTPUT_FILE"
echo ""

# Check BF16 support
if python -c "import torch; assert torch.cuda.is_bf16_supported()" 2>/dev/null; then
    echo "✓ BF16 supported on this GPU"
    DTYPE="bf16"
else
    echo "⚠ BF16 not supported, using FP16 instead"
    DTYPE="fp16"
fi

echo ""
echo "Starting benchmark..."
echo "This will take 10-20 minutes depending on GPU memory"
echo ""

# Run the benchmark for all models
uv run python -m cs336_systems.mixed_precision.benchmark_mixed_precision \
    --all-models \
    --context-length 512 \
    --batch-size 4 \
    --warmup-steps 5 \
    --measure-steps 10 \
    --dtype "$DTYPE" \
    --output "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "For the writeup:"
echo "  1. Report timings for each model size (FP32 vs $DTYPE)"
echo "  2. Discuss speedup trends as model size increases"
echo "  3. Comment on whether mixed precision is worth the complexity"
echo ""
