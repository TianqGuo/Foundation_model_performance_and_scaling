#!/bin/bash
# ==============================================================================
# Run All-Reduce Benchmarking - Gloo + CPU Only
# ==============================================================================
#
# DESCRIPTION:
#   Runs all-reduce benchmarking with Gloo backend on CPU.
#   Tests 2, 4, 6 processes with data sizes 1MB, 10MB, 100MB, 1GB.
#   Suitable for local testing on any machine (no GPU required).
#
# USAGE:
#   cd cs336_systems/distributed_communication
#   ./run_benchmark_cpu.sh
#
# OUTPUT:
#   - CSV results: ../../results/distributed_communication/gloo_cpu_benchmark.csv
#
# REQUIREMENTS:
#   - No GPU required
#   - Runtime: ~3-5 minutes
#
# NOTES:
#   - Tests same configurations as GPU version for easy comparison
#   - Safe to run on WSL2, Linux, or macOS
#   - Use this to verify implementation before running on H100
#
# ==============================================================================

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "All-Reduce Benchmarking - Gloo + CPU"
echo "================================================================================"
echo ""

# Check if running in WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "✓ Running in WSL2"
    echo ""
fi

# Configuration
BACKEND="gloo"
DEVICE="cpu"
NUM_PROCESSES="2 4 6"
DATA_SIZES="1 10 100 1000"
NUM_WARMUP=5
NUM_ITERS=100
OUTPUT_FILE="../../results/distributed_communication/gloo_cpu_benchmark.csv"

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
echo "✓ CPU benchmark complete!"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  1. Review results: cat $OUTPUT_FILE"
echo "  2. Run GPU benchmark on H100: ./run_benchmark_gpu.sh"
echo "  3. Combine and visualize: uv run python visualize_results.py"
echo ""