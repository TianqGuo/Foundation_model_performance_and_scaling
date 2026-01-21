#!/bin/bash
# ==============================================================================
# Run Naïve DDP Benchmark
# ==============================================================================
#
# DESCRIPTION:
#   Benchmarks naïve DDP training to measure communication overhead.
#   Tests with XL model (d_model=1600, 48 layers) on 2 GPUs.
#
# USAGE:
#   cd cs336_systems/naive_ddp
#   ./run_benchmark.sh
#
# OUTPUT:
#   - CSV results: ../../results/naive_ddp/benchmark_results.csv
#   - Console output with timing breakdown
#
# REQUIREMENTS:
#   - At least 2 GPUs (ideally with significant memory for XL model)
#   - Runtime: ~5-10 minutes
#
# NOTES:
#   - Adjust model size if GPU memory is limited
#   - Use --model-size large or medium for GPUs with <16GB memory
#
# ==============================================================================

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "Naïve DDP Training Benchmark"
echo "================================================================================"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ Error: nvidia-smi not found"
    echo "  This benchmark requires GPUs"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "✗ Error: Need at least 2 GPUs for DDP benchmark"
    echo "  Found only $GPU_COUNT GPU(s)"
    exit 1
fi

# Determine model size based on GPU memory
GPU_MEMORY_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
echo "GPU Memory: ${GPU_MEMORY_GB}GB per GPU"
echo ""

if [ "$GPU_MEMORY_GB" -ge 40 ]; then
    MODEL_SIZE="xl"
    echo "✓ Using XL model (sufficient GPU memory)"
elif [ "$GPU_MEMORY_GB" -ge 24 ]; then
    MODEL_SIZE="large"
    echo "⚠ Using Large model (limited GPU memory, XL may OOM)"
else
    MODEL_SIZE="medium"
    echo "⚠ Using Medium model (limited GPU memory, XL would OOM)"
fi

echo ""
echo "Configuration:"
echo "  Model size: $MODEL_SIZE"
echo "  GPUs: 2"
echo "  Batch size: 4"
echo "  Context length: 512"
echo ""
echo "Starting benchmark..."
echo "================================================================================"
echo ""

# Run benchmark
uv run python benchmark_naive_ddp.py \
    --model-size "$MODEL_SIZE" \
    --batch-size 4 \
    --context-length 512 \
    --num-warmup 3 \
    --num-steps 10 \
    --world-size 2 \
    --output "../../results/naive_ddp/benchmark_results.csv"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================================"
    echo "✓ Benchmark complete!"
    echo "================================================================================"
    echo ""
    echo "Results saved to: ../../results/naive_ddp/benchmark_results.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Review results: cat ../../results/naive_ddp/benchmark_results.csv"
    echo "  2. Analyze communication overhead in the output above"
    echo "  3. Compare with optimized DDP implementations (future sections)"
    echo ""
else
    echo "================================================================================"
    echo "✗ Benchmark failed"
    echo "================================================================================"
fi
echo ""

exit $EXIT_CODE