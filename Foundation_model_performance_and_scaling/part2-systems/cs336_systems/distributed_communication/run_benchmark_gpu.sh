#!/bin/bash
# ==============================================================================
# Run All-Reduce Benchmarking - NCCL + GPU Only
# ==============================================================================
#
# DESCRIPTION:
#   Runs all-reduce benchmarking with NCCL backend on GPU.
#   Tests 2, 4, 6 processes with data sizes 1MB, 10MB, 100MB, 1GB.
#   Requires 6 GPUs for full benchmark.
#
# USAGE:
#   cd cs336_systems/distributed_communication
#   ./run_benchmark_gpu.sh
#
# OUTPUT:
#   - CSV results: ../../results/distributed_communication/nccl_gpu_benchmark.csv
#
# REQUIREMENTS:
#   - 6 GPUs (will auto-adjust if fewer available)
#   - CUDA and NCCL properly configured
#   - Runtime: ~3-5 minutes
#
# NOTES:
#   - Automatically detects available GPU count
#   - Adjusts process count based on available GPUs
#   - Run on H100 instance for full benchmark
#
# ==============================================================================

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "All-Reduce Benchmarking - NCCL + GPU"
echo "================================================================================"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ Error: nvidia-smi not found"
    echo "  This script requires CUDA-capable GPUs"
    echo "  Please run on a system with NVIDIA GPUs"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Determine process counts based on GPU availability
if [ "$GPU_COUNT" -ge 6 ]; then
    echo "✓ Running full benchmark suite (2, 4, 6 processes)"
    NUM_PROCESSES="2 4 6"
elif [ "$GPU_COUNT" -ge 4 ]; then
    echo "⚠ Limited GPUs: running with 2, 4 processes only"
    NUM_PROCESSES="2 4"
elif [ "$GPU_COUNT" -ge 2 ]; then
    echo "⚠ Limited GPUs: running with 2 processes only"
    NUM_PROCESSES="2"
else
    echo "✗ Error: Need at least 2 GPUs for benchmarking"
    echo "  Found only $GPU_COUNT GPU(s)"
    exit 1
fi

# Configuration
BACKEND="nccl"
DEVICE="cuda"
DATA_SIZES="1 10 100 1000"
NUM_WARMUP=5
NUM_ITERS=100
OUTPUT_FILE="../../results/distributed_communication/nccl_gpu_benchmark.csv"

echo ""
echo "Configuration:"
echo "  Backend:        $BACKEND"
echo "  Device:         $DEVICE"
echo "  Processes:      $NUM_PROCESSES"
echo "  Data sizes:     $DATA_SIZES MB"
echo "  Warm-up iters:  $NUM_WARMUP"
echo "  Test iters:     $NUM_ITERS"
echo "  Output:         $OUTPUT_FILE"
echo ""
echo "Starting benchmark..."
echo "================================================================================"
echo ""

# Run the benchmark
uv run python benchmark_allreduce.py \
    --backend $BACKEND \
    --device $DEVICE \
    --num-processes $NUM_PROCESSES \
    --data-sizes $DATA_SIZES \
    --num-warmup $NUM_WARMUP \
    --num-iters $NUM_ITERS \
    --output "$OUTPUT_FILE"

echo ""
echo "================================================================================"
echo "✓ GPU benchmark complete!"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  1. Review results: cat $OUTPUT_FILE"
echo "  2. Download to local machine: scp h100:path/to/$OUTPUT_FILE ./results/"
echo "  3. Combine with CPU results and visualize: uv run python visualize_results.py"
echo ""