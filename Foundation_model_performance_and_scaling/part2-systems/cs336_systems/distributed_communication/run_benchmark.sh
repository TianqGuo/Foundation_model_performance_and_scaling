#!/bin/bash
# ==============================================================================
# Run All-Reduce Benchmarking - Part 2.1.3
# ==============================================================================
#
# DESCRIPTION:
#   Runs the all-reduce benchmarking script with default settings.
#   Tests various backends (Gloo+CPU, NCCL+GPU), data sizes (1MB to 1GB),
#   and number of processes (2, 4, 6).
#
# USAGE:
#   cd cs336_systems/distributed_communication
#   ./run_benchmark.sh
#
# OUTPUT:
#   - CSV results: ../../results/distributed_communication/allreduce_benchmark.csv
#
# REQUIREMENTS:
#   - For full benchmark: up to 6 GPUs
#   - Runtime: ~5 minutes for all configurations
#
# NOTES:
#   - On WSL2 or systems with limited GPUs, the script will skip configurations
#     that require more GPUs than available
#   - Modify the script to test specific configurations if needed
#
# ==============================================================================

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "All-Reduce Benchmarking - Part 2.1.3"
echo "================================================================================"
echo ""

# Check if running in WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "⚠ Running in WSL2 - some features may be limited"
    echo ""
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ Found $GPU_COUNT GPU(s)"
    echo ""
else
    echo "⚠ No GPUs detected - will only run CPU benchmarks"
    echo ""
    GPU_COUNT=0
fi

# Determine which configurations to run based on GPU availability
if [ "$GPU_COUNT" -ge 6 ]; then
    echo "Running full benchmark suite (CPU + GPU, 2/4/6 processes)..."
    NUM_PROCESSES="2 4 6"
elif [ "$GPU_COUNT" -ge 4 ]; then
    echo "Running limited GPU benchmark (up to 4 processes)..."
    NUM_PROCESSES="2 4"
elif [ "$GPU_COUNT" -ge 2 ]; then
    echo "Running minimal GPU benchmark (up to 2 processes)..."
    NUM_PROCESSES="2"
else
    echo "Running CPU-only benchmark..."
    NUM_PROCESSES="2 4 6"
fi

# Run the benchmark
if [ "$GPU_COUNT" -ge 2 ]; then
    # Run both CPU and GPU benchmarks
    uv run python benchmark_allreduce.py \
        --backend gloo nccl \
        --device cpu cuda \
        --data-sizes 1 10 100 1000 \
        --num-processes $NUM_PROCESSES \
        --num-warmup 5 \
        --num-iters 100 \
        --output "../../results/distributed_communication/allreduce_benchmark.csv"
else
    # CPU only
    uv run python benchmark_allreduce.py \
        --backend gloo \
        --device cpu \
        --data-sizes 1 10 100 1000 \
        --num-processes $NUM_PROCESSES \
        --num-warmup 5 \
        --num-iters 100 \
        --output "../../results/distributed_communication/allreduce_benchmark.csv"
fi

echo ""
echo "================================================================================"
echo "✓ Benchmark complete!"
echo "================================================================================"
echo ""
echo "Results saved to: ../../results/distributed_communication/allreduce_benchmark.csv"
echo ""
echo "Next steps:"
echo "  1. Analyze results with visualization script"
echo "  2. Generate plots for writeup"
echo ""