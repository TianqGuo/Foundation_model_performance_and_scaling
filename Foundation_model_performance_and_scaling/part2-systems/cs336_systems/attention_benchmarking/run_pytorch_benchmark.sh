#!/bin/bash
# ==============================================================================
# PyTorch Attention Benchmarking - Section 1.2.1
# ==============================================================================
#
# This script runs the PyTorch attention benchmarking for all configurations
# specified in the requirements:
#   - Batch size: 8 (fixed)
#   - d_model: [16, 32, 64, 128]
#   - seq_len: [256, 1024, 4096, 8192, 16384]
#
# USAGE:
#   cd cs336_systems/attention_benchmarking
#   ./run_pytorch_benchmark.sh
#
# OUTPUT:
#   results/attention_benchmarking/pytorch_attention_benchmark.csv
#   results/attention_benchmarking/memory_analysis_*.txt
#
# NOTES:
#   - This will take 10-20 minutes depending on when OOM occurs
#   - The script will automatically identify OOM configurations
#   - Memory accounting analysis will be run for the smallest OOM case
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

# Configuration
BATCH_SIZE=8
D_MODEL_VALUES=(16 32 64 128)
SEQ_LEN_VALUES=(256 1024 4096 8192 16384)
NUM_WARMUP=10
NUM_ITERATIONS=100

OUTPUT_DIR="../../results/attention_benchmarking"
BENCHMARK_CSV="${OUTPUT_DIR}/pytorch_attention_benchmark.csv"

echo "=============================================================================="
echo "PyTorch Attention Benchmarking - Section 1.2.1"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  d_model values: ${D_MODEL_VALUES[@]}"
echo "  seq_len values: ${SEQ_LEN_VALUES[@]}"
echo "  Warmup iterations: ${NUM_WARMUP}"
echo "  Measurement iterations: ${NUM_ITERATIONS}"
echo ""
echo "This will benchmark ${#D_MODEL_VALUES[@]} × ${#SEQ_LEN_VALUES[@]} = $((${#D_MODEL_VALUES[@]} * ${#SEQ_LEN_VALUES[@]})) configurations"
echo ""

# Check if GPU is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Running benchmarks..."
echo ""

# Run the benchmark script
python -m cs336_systems.attention_benchmarking.benchmark_pytorch_attention \
    --batch-size ${BATCH_SIZE} \
    --d-model ${D_MODEL_VALUES[@]} \
    --seq-len ${SEQ_LEN_VALUES[@]} \
    --num-warmup ${NUM_WARMUP} \
    --num-iterations ${NUM_ITERATIONS} \
    --output "${BENCHMARK_CSV}" | tee "${OUTPUT_DIR}/benchmark_output.txt"

echo ""
echo "=============================================================================="
echo "BENCHMARK COMPLETE"
echo "=============================================================================="
echo ""
echo "Results saved to: ${BENCHMARK_CSV}"
echo ""

# Find smallest OOM configuration from the CSV
if [ -f "${BENCHMARK_CSV}" ]; then
    echo "Analyzing OOM configurations..."
    echo ""

    # Extract OOM configurations and find the smallest one
    OOM_LINE=$(grep "OOM" "${BENCHMARK_CSV}" | sort -t',' -k2,2n -k3,3n | head -1)

    if [ -n "$OOM_LINE" ]; then
        # Parse the CSV line to extract d_model and seq_len
        D_MODEL_OOM=$(echo "$OOM_LINE" | cut -d',' -f3)
        SEQ_LEN_OOM=$(echo "$OOM_LINE" | cut -d',' -f2)

        echo "Smallest OOM configuration found:"
        echo "  d_model = ${D_MODEL_OOM}"
        echo "  seq_len = ${SEQ_LEN_OOM}"
        echo ""
        echo "Running memory accounting analysis..."
        echo ""

        # Run memory accounting for this configuration
        python -m cs336_systems.attention_benchmarking.memory_accounting \
            --batch-size ${BATCH_SIZE} \
            --seq-len ${SEQ_LEN_OOM} \
            --d-model ${D_MODEL_OOM} | tee "${OUTPUT_DIR}/memory_analysis_d${D_MODEL_OOM}_s${SEQ_LEN_OOM}.txt"

        echo ""
        echo "Memory analysis saved to:"
        echo "  ${OUTPUT_DIR}/memory_analysis_d${D_MODEL_OOM}_s${SEQ_LEN_OOM}.txt"
    else
        echo "No OOM configurations found. All tests passed!"
        echo "Consider testing with larger configurations."
    fi
fi

echo ""
echo "=============================================================================="
echo "NEXT STEPS FOR WRITEUP"
echo "=============================================================================="
echo ""
echo "1. Review the benchmark results in:"
echo "   ${BENCHMARK_CSV}"
echo ""
echo "2. Check the memory analysis for OOM configuration:"
echo "   ${OUTPUT_DIR}/memory_analysis_*.txt"
echo ""
echo "3. For the writeup, include:"
echo "   - Timing table for all configurations (or note OOM)"
echo "   - Memory accounting for smallest OOM case"
echo "   - Discussion of how backward memory scales with seq_len"
echo "   - Mitigation strategies (FlashAttention-2)"
echo ""
