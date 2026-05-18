#!/bin/bash
# ==============================================================================
# Section 4: Supervised Fine-Tuning (SFT) for MATH
# ==============================================================================
#
# USAGE:
#   cd cs336_alignment/section4_sft
#   ./part_5_4.sh                  # tests + all training experiments (2 GPUs)
#   ./part_5_4.sh --tests-only     # helper tests only (no training, CPU)
#   ./part_5_4.sh --train-only     # all training experiments, skip tests
#   ./part_5_4.sh --smoke-test     # single-GPU local smoke test (no eval, no wandb)
#
# WHAT IT DOES:
#   1. Resolves model and data paths (cluster → local assets → HuggingFace)
#   2. Runs pytest for Section 4 helper method tests
#   3. Runs SFT training for dataset size ablation: 128, 256, 512, 1024, full
#   4. Runs SFT training on the filtered (correct-answers-only) dataset
#   Each run logs to wandb and saves a model checkpoint.
#
# OUTPUT:
#   ${ROOT}/results/section4/           — eval logs, accuracy curves, dataset info
#   /data/${USER}/sft_n{size}/          — model checkpoints (cluster)
#   ${ROOT}/assets/sft_n{size}/         — model checkpoints (local fallback)
#
# NOTES:
#   Full training requires 2 GPUs (~80 GB VRAM each, H100 recommended).
#   --smoke-test runs on a single GPU with 32 examples, skipping vLLM eval.
#   Helper tests (--tests-only) only require the tokenizer and run on CPU.
#   On the cluster the model and data are pre-downloaded at /data/a5-alignment/.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

TESTS_ONLY=false
TRAIN_ONLY=false
SMOKE_TEST=false
for arg in "$@"; do
    case "$arg" in
        --tests-only)  TESTS_ONLY=true ;;
        --train-only)  TRAIN_ONLY=true ;;
        --smoke-test)  SMOKE_TEST=true ;;
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

# --- Data paths ---
CLUSTER_SFT="/data/a5-alignment/MATH/sft.jsonl"
CLUSTER_VAL="/data/a5-alignment/MATH/validation.jsonl"
LOCAL_SFT="${ROOT}/data/math/sft.jsonl"
LOCAL_VAL="${ROOT}/data/math/validation.jsonl"

if [ -z "${DATA}" ]; then
    if   [ -f "${CLUSTER_SFT}" ]; then DATA="${CLUSTER_SFT}"
    elif [ -f "${LOCAL_SFT}" ];   then DATA="${LOCAL_SFT}"
    else echo "ERROR: SFT data not found at ${CLUSTER_SFT} or ${LOCAL_SFT}"; exit 1
    fi
fi

if [ -z "${VAL_DATA}" ]; then
    if   [ -f "${CLUSTER_VAL}" ]; then VAL_DATA="${CLUSTER_VAL}"
    elif [ -f "${LOCAL_VAL}" ];   then VAL_DATA="${LOCAL_VAL}"
    else VAL_DATA="${CLUSTER_VAL}"  # will be skipped gracefully if missing
    fi
fi

OUTPUT="${ROOT}/results/section4"

echo "==> Section 4: SFT for MATH"
echo "    Model:    ${MODEL}"
echo "    SFT data: ${DATA}"
echo "    Val data: ${VAL_DATA}"
echo "    Output:   ${OUTPUT}"
echo ""

# ---------------------------------------------------------------------------
# Smoke test: single-GPU local run (no vLLM eval, no wandb)
# ---------------------------------------------------------------------------
if [ "${SMOKE_TEST}" = true ]; then
    echo "==> [smoke-test] SFT smoke test (32 examples, 1 epoch, no eval)"
    uv run python "${ROOT}/cs336_alignment/section4_sft/train_sft.py" \
        --model    "${MODEL}" \
        --data     "${DATA}" \
        --val_data "${VAL_DATA}" \
        --output   "${OUTPUT}" \
        --max_train_examples 32 \
        --n_epochs 1 \
        --gradient_accumulation_steps 4 \
        --skip_eval \
        --no_wandb \
        --run_name "sft_smoke"
    echo "==> Smoke test done. Results at ${OUTPUT}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: Helper method tests
# ---------------------------------------------------------------------------
if [ "${TRAIN_ONLY}" = false ]; then
    echo "==> [1/2] Running Section 4 helper tests..."
    cd "${ROOT}"
    uv run pytest tests/test_sft.py -v
    cd "${ROOT}/cs336_alignment/section4_sft"
    echo "==> Helper tests passed."
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: SFT training experiments
# ---------------------------------------------------------------------------
if [ "${TESTS_ONLY}" = false ]; then
    echo "==> [2/2] Running SFT training experiments..."
    echo ""

    TRAIN_CMD="uv run python ${ROOT}/cs336_alignment/section4_sft/train_sft.py
        --model    ${MODEL}
        --data     ${DATA}
        --val_data ${VAL_DATA}
        --output   ${OUTPUT}"

    # --- Dataset size ablation ---
    for N in 128 256 512 1024; do
        echo "--- SFT with ${N} training examples ---"
        ${TRAIN_CMD} \
            --max_train_examples ${N} \
            --run_name "sft_n${N}"
        echo ""
    done

    echo "--- SFT with full dataset ---"
    ${TRAIN_CMD} \
        --run_name "sft_full"
    echo ""

    # --- Filtered dataset experiment ---
    echo "--- SFT with correct-answer-filtered dataset ---"
    ${TRAIN_CMD} \
        --filter_correct \
        --run_name "sft_filtered"
    echo ""

    echo "==> All training experiments done. Results at ${OUTPUT}"
fi