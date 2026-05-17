#!/bin/bash
# ==============================================================================
# Section 4: Supervised Fine-Tuning (SFT) for MATH
# ==============================================================================
#
# USAGE:
#   cd cs336_alignment/section4_sft
#   ./part_5_4.sh                        # run helper tests + SFT training
#   ./part_5_4.sh --tests-only           # helper tests only (no training)
#   ./part_5_4.sh --train-only           # SFT training only
#
#   Override paths via environment variables:
#   MODEL=/path/to/model DATA=/path/to/sft.jsonl ./part_5_4.sh
#
# WHAT IT DOES:
#   1. Resolves the Qwen 2.5 Math 1.5B model path (cluster → local → HuggingFace)
#   2. Downloads the model/tokenizer locally if not found on the cluster
#   3. Runs pytest for Section 4 helper method tests
#   4. Runs the full SFT training experiment on the MATH reasoning dataset,
#      varying dataset sizes {128, 256, 512, 1024, full} and logging to wandb
#
# OUTPUT:
#   ${ROOT}/results/section4/           — evaluation logs and accuracy curves
#   /data/${USER}/sft_model/            — trained model checkpoint (cluster)
#   ${ROOT}/assets/sft_model/           — trained model checkpoint (local)
#
# NOTES:
#   Helper tests (step 3) need only the tokenizer; full training (step 4)
#   requires 2 GPUs with ~80 GB VRAM each (H100 recommended).
#   On the cluster, the model and SFT data are pre-downloaded at /data/a5-alignment/.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

TESTS_ONLY=false
TRAIN_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --tests-only) TESTS_ONLY=true ;;
        --train-only) TRAIN_ONLY=true ;;
    esac
done

# --- Model path ---
CLUSTER_MODEL="/data/a5-alignment/models/Qwen2.5-Math-1.5B"
LOCAL_MODEL="${ROOT}/assets/Qwen2.5-Math-1.5B"
if [ -z "${MODEL}" ]; then
    if [ -d "${CLUSTER_MODEL}" ]; then
        MODEL="${CLUSTER_MODEL}"
    elif [ -d "${LOCAL_MODEL}" ]; then
        MODEL="${LOCAL_MODEL}"
    else
        echo "INFO: Model not found locally. Downloading from HuggingFace..."
        mkdir -p "${ROOT}/assets"
        uv run huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
            --local-dir "${LOCAL_MODEL}"
        MODEL="${LOCAL_MODEL}"
    fi
fi
echo "==> Model: ${MODEL}"

# --- SFT data path ---
CLUSTER_SFT="/data/a5-alignment/MATH/sft.jsonl"
LOCAL_SFT="${ROOT}/data/math/sft.jsonl"
if [ -z "${DATA}" ]; then
    if [ -f "${CLUSTER_SFT}" ]; then
        DATA="${CLUSTER_SFT}"
    elif [ -f "${LOCAL_SFT}" ]; then
        DATA="${LOCAL_SFT}"
    else
        echo "WARNING: SFT data not found. Training will fail if --train-only is set."
        DATA="${CLUSTER_SFT}"
    fi
fi
echo "==> SFT data: ${DATA}"
echo ""

# --- Step 1: Helper method tests ---
if [ "${TRAIN_ONLY}" = false ]; then
    echo "==> Running Section 4 helper tests..."
    cd "${ROOT}"
    uv run pytest tests/test_sft.py -v
    cd "${ROOT}/cs336_alignment/section4_sft"
    echo "==> Helper tests passed."
    echo ""
fi

# --- Step 2: SFT training experiment ---
if [ "${TESTS_ONLY}" = false ]; then
    echo "==> Running SFT training experiment..."
    OUTPUT_DIR="${ROOT}/results/section4"
    mkdir -p "${OUTPUT_DIR}"

    uv run python "${ROOT}/cs336_alignment/section4_sft/train_sft.py" \
        --model   "${MODEL}" \
        --data    "${DATA}" \
        --output  "${OUTPUT_DIR}" \
        "$@"

    echo ""
    echo "==> Done. Results at ${OUTPUT_DIR}"
fi