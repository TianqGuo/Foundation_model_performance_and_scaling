#!/bin/bash
# ==============================================================================
# Part (e): Compare softmax vs matmul in self-attention layer
# ==============================================================================
#
# Question: Compare the runtime of softmax vs matrix multiplication operations
# within the self-attention layer during forward pass. How does the difference
# in runtimes compare to the difference in FLOPs?
#
# This script profiles with detailed attention annotations to isolate operations.
#
# USAGE:
#   cd cs336_systems/nsight_systems_profiler
#   ./profile_part_e.sh
#
# OUTPUT:
#   - Profile with detailed attention breakdown: ../../results/nsight_profiles/part_e/
#
# ANALYSIS:
#   Compare runtimes and FLOPs:
#   - Attention score matmul (Q @ K^T)
#   - Softmax operation
#   - Output matmul (attention @ V)
#
# ==============================================================================

set -e

echo "=========================================="
echo "Part (e): Attention Component Analysis"
echo "=========================================="
echo ""

OUTPUT_DIR="../../results/nsight_profiles/part_e"
mkdir -p "$OUTPUT_DIR"

# Profile all model sizes for attention analysis
MODELS=("small" "medium" "large" "xl" "2.7B")
CTX=512
BATCH=4

echo "Profiling self-attention components..."
echo "Context: $CTX"
echo "Batch: $BATCH"
echo ""

# Function to get model dimensions
get_model_dims() {
    case $1 in
        "small")   echo "768 12" ;;   # d_model, num_heads
        "medium")  echo "1024 16" ;;
        "large")   echo "1280 20" ;;
        "xl")      echo "1600 25" ;;
        "2.7B")    echo "2560 32" ;;
    esac
}

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Profiling model: $model (attention analysis)"
    echo "=========================================="
    echo ""

    # Get model-specific dimensions
    read D_MODEL NUM_HEADS <<< $(get_model_dims "$model")
    D_HEAD=$((D_MODEL / NUM_HEADS))

    echo "FLOPs calculation (per attention layer):"
    echo "  d_model = $D_MODEL"
    echo "  num_heads = $NUM_HEADS"
    echo "  d_head = $D_HEAD"
    echo "  seq_len = $CTX"
    echo "  batch_size = $BATCH"
    echo ""
    echo "  QK^T matmul: 2 * batch * num_heads * seq_len^2 * d_head"
    echo "            = 2 * $BATCH * $NUM_HEADS * $CTX^2 * $D_HEAD"
    echo "            = $((2 * BATCH * NUM_HEADS * CTX * CTX * D_HEAD)) FLOPs"
    echo ""
    echo "  Softmax: ~5 * batch * num_heads * seq_len^2 (approx)"
    echo "         = 5 * $BATCH * $NUM_HEADS * $CTX^2"
    echo "         = $((5 * BATCH * NUM_HEADS * CTX * CTX)) FLOPs"
    echo ""
    echo "  Attention @ V: 2 * batch * num_heads * seq_len^2 * d_head"
    echo "               = $((2 * BATCH * NUM_HEADS * CTX * CTX * D_HEAD)) FLOPs"
    echo ""

    uv run nsys profile \
        -o "${OUTPUT_DIR}/${model}_attention_analysis.nsys-rep" \
        --force-overwrite true \
        --trace=cuda,nvtx --stats=true \
        --python-backtrace=cuda \
        python -m cs336_systems.nsight_systems_profiler.profile_model \
        --model-size "$model" \
        --context-length "$CTX" \
        --batch-size "$BATCH" \
        --warmup-steps 5 \
        --measure-steps 10 \
        --profile-type forward \
        --use-annotated-attention

    echo "  ✓ Attention profile saved for $model"
    echo ""
done

echo ""
echo "=========================================="
echo "Part (e) Complete!"
echo "=========================================="
echo ""
echo "Analysis instructions:"
echo ""
echo "1. Open profile in Nsight Systems"
echo ""
echo "2. In NVTX row, find 'scaled_dot_product_attention' ranges"
echo "   Inside each, you'll see:"
echo "   - computing_attention_scores (Q @ K^T)"
echo "   - computing_softmax"
echo "   - final_matmul (attention @ V)"
echo ""
echo "3. For each sub-range, check CUDA kernels below it:"
echo "   - Note kernel names and total time"
echo "   - Matmuls: look for *gemm* kernels"
echo "   - Softmax: look for *softmax* or *reduce* kernels"
echo ""
echo "4. Calculate time ratios:"
echo "   - softmax_time / matmul_time"
echo "   - Compare with FLOP ratio from above"
echo ""
echo "5. Expected: Softmax takes proportionally MORE time than its FLOP share"
echo "   (memory-bound vs compute-bound operations)"
echo ""
