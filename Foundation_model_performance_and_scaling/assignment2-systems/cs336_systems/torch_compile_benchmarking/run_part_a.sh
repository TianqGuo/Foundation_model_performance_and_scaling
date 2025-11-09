#!/bin/bash
# ==============================================================================
# Torch.compile Attention Benchmarking - Section 1.3.1(a)
# ==============================================================================
#
# This script benchmarks compiled vs vanilla attention for all configurations.
#
# USAGE:
#   cd cs336_systems/torch_compile_benchmarking
#   ./run_part_a.sh
#
# OUTPUT:
#   results/torch_compile_benchmarking/compiled_attention_benchmark.csv
#
# NOTES:
#   - This will take 15-30 minutes depending on GPU and when OOM occurs
#   - Tests both vanilla and torch.compile versions of attention
#   - Compares forward and backward pass timings
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

OUTPUT_DIR="../../results/torch_compile_benchmarking"
BENCHMARK_CSV="${OUTPUT_DIR}/compiled_attention_benchmark.csv"

echo "=============================================================================="
echo "Torch.compile Attention Benchmarking - Section 1.3.1(a)"
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
echo "For BOTH vanilla and compiled versions (total: $(( ${#D_MODEL_VALUES[@]} * ${#SEQ_LEN_VALUES[@]} * 2 )) tests)"
echo ""

# Check if GPU is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Check if torch.compile is available
if ! python -c "import torch; assert hasattr(torch, 'compile')" 2>/dev/null; then
    echo "ERROR: torch.compile not available. Requires PyTorch 2.0+"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Running benchmarks..."
echo ""

# Run the benchmark script
python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_attention \
    --batch-size ${BATCH_SIZE} \
    --d-model ${D_MODEL_VALUES[@]} \
    --seq-len ${SEQ_LEN_VALUES[@]} \
    --num-warmup ${NUM_WARMUP} \
    --num-iterations ${NUM_ITERATIONS} \
    --output "${BENCHMARK_CSV}"

echo ""
echo "=============================================================================="
echo "BENCHMARK COMPLETE"
echo "=============================================================================="
echo ""
echo "Results saved to: ${BENCHMARK_CSV}"
echo ""
echo "=============================================================================="
echo "NEXT STEPS FOR WRITEUP"
echo "=============================================================================="
echo ""
echo "1. Review the comparison tables in the output above"
echo ""
echo "2. For the writeup (Section 1.3.1(a)), include:"
echo "   - Table comparing forward pass timings (vanilla vs compiled)"
echo "   - Table comparing backward pass timings (vanilla vs compiled)"
echo "   - Speedup ratios for each configuration"
echo "   - Brief discussion of performance changes observed"
echo ""
echo "3. Key questions to answer:"
echo "   - Where does torch.compile provide the most benefit?"
echo "   - Does speedup vary with sequence length or model dimension?"
echo "   - Are there cases where compiled is slower (compilation overhead)?"
echo ""
echo "=============================================================================="
