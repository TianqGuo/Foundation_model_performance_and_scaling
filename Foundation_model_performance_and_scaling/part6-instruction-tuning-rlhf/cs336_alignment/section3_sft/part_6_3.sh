#!/usr/bin/env bash
# =============================================================================
# USAGE:   bash cs336_alignment/section3_sft/part_6_3.sh [--smoke-test]
#
# WHAT IT DOES:
#   Instruction fine-tunes Llama 3.1 8B Base on safety-augmented UltraChat-200K
#   (single-turn) using packed-sequence SFT (Alpaca prompt format).
#
#   Training setup:
#     - 1 epoch, seq_length=512, effective batch size=32
#       (micro_batch_size=2 × gradient_accumulation_steps=16)
#     - LR 2e-5 with cosine decay + 3% linear warmup
#     - AdamW, bfloat16 + FlashAttention-2
#     - Logs train loss and periodic validation loss (console + W&B)
#     - Checkpoint saved to assets/sft_ultrachat/
#
# OUTPUT:
#   results/section3/train_metrics_sft_ultrachat.jsonl  — per-step metrics
#   results/section3/final_val_sft_ultrachat.json       — final validation loss
#   assets/sft_ultrachat/                               — model checkpoint for §4
#
# NOTES:
#   --smoke-test runs 50 optimizer steps (for quick local validation).
#   Requires 1× H100 (or comparable) GPU and ~40 GB VRAM.
#   W&B logging is enabled by default; pass --no-wandb to disable.
# =============================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SMOKE_TEST=0
NO_WANDB=""
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST=1 ;;
        --no-wandb)   NO_WANDB="--no_wandb" ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR="${ROOT}/data/ultrachat"
CLUSTER_DATA="/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn"
TRAIN_URL="https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
TEST_URL="https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz"

CLUSTER_MODEL="/data/a5-alignment/models/Llama-3.1-8B"
LOCAL_MODEL="${ROOT}/assets/Llama-3.1-8B"

RESULTS_DIR="${ROOT}/results/section3"
CHECKPOINT_DIR="${ROOT}/assets/sft_ultrachat"

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
echo "==> Checking UltraChat data ..."
if [ -d "${CLUSTER_DATA}" ]; then
    echo "    Using cluster data at ${CLUSTER_DATA}"
    TRAIN_DATA="${CLUSTER_DATA}/train.jsonl.gz"
    VAL_DATA="${CLUSTER_DATA}/test.jsonl.gz"
else
    echo "    Cluster data not found — downloading ..."
    for FILENAME in train.jsonl.gz test.jsonl.gz; do
        if [ ! -f "${DATA_DIR}/${FILENAME}" ]; then
            URL=$([ "$FILENAME" = "train.jsonl.gz" ] && echo "$TRAIN_URL" || echo "$TEST_URL")
            wget -q --show-progress "${URL}" -O "${DATA_DIR}/${FILENAME}"
        fi
    done
    TRAIN_DATA="${DATA_DIR}/train.jsonl.gz"
    VAL_DATA="${DATA_DIR}/test.jsonl.gz"
fi
echo "    Train: ${TRAIN_DATA}"
echo "    Val:   ${VAL_DATA}"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
echo "==> Checking Llama 3.1 8B ..."
if [ -f "${CLUSTER_MODEL}/config.json" ] && [ -f "${CLUSTER_MODEL}/tokenizer_config.json" ]; then
    echo "    Using cluster model at ${CLUSTER_MODEL}"
    MODEL_PATH="${CLUSTER_MODEL}"
elif [ -f "${LOCAL_MODEL}/config.json" ] && [ -f "${LOCAL_MODEL}/tokenizer_config.json" ]; then
    echo "    Using local model at ${LOCAL_MODEL}"
    MODEL_PATH="${LOCAL_MODEL}"
else
    echo "    Not found locally — running get_assets.sh to download ..."
    bash "${ROOT}/get_assets.sh"
    if [ -f "${LOCAL_MODEL}/config.json" ] && [ -f "${LOCAL_MODEL}/tokenizer_config.json" ]; then
        MODEL_PATH="${LOCAL_MODEL}"
    else
        echo "    ERROR: Download failed. Check your HuggingFace login:" >&2
        echo "      uv run huggingface-cli login" >&2
        echo "    and accept the Llama 3.1 licence at https://huggingface.co/meta-llama/Llama-3.1-8B" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Extra flags for smoke test
# ---------------------------------------------------------------------------
EXTRA_FLAGS=""
if [ "${SMOKE_TEST}" -eq 1 ]; then
    echo "==> Smoke-test mode: 50 optimizer steps"
    # Use a tiny subset by limiting the dataset to a tiny file won't work cleanly
    # with gzip; we rely on --max_steps instead
    EXTRA_FLAGS="--max_steps 50 --val_interval 10"
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
echo "==> Starting SFT training ..."
uv run python "${ROOT}/cs336_alignment/section3_sft/train_sft.py" \
    --model         "${MODEL_PATH}" \
    --train_data    "${TRAIN_DATA}" \
    --val_data      "${VAL_DATA}" \
    --output        "${RESULTS_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --seq_length    512 \
    --micro_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --n_epochs      1 \
    --lr            2e-5 \
    --val_interval  100 \
    --run_name      sft_ultrachat \
    --wandb_project cs336-part6-sft \
    ${NO_WANDB} \
    ${EXTRA_FLAGS}

echo "==> Done. Checkpoint at ${CHECKPOINT_DIR}"
