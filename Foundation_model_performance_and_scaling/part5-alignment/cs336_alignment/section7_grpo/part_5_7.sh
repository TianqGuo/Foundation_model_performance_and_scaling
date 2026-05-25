#!/usr/bin/env bash
# USAGE:   bash part_5_7.sh [OPTIONS]
#
# WHAT IT DOES:
#   Runs GRPO training on MATH using Qwen 2.5 Math 1.5B (base model).
#   Generates G rollouts per question, computes group-normalized advantages,
#   and updates the policy with the specified policy-gradient loss type.
#
# OUTPUT:
#   results/section7/eval_metrics_<run_name>.jsonl  — per-step val metrics
#   results/section7/final_eval.json                — final evaluation
#   assets/grpo_<run_name>/                         — saved model checkpoint
#   results/section7/grpo_accuracy.png              — accuracy curves
#
# OPTIONS:
#   --smoke-test              3 GRPO steps on 64 examples (single GPU OK)
#   --loss-type=TYPE          no_baseline | reinforce_with_baseline (default) |
#                             grpo_clip | grpo_no_clip
#   --off-policy              epochs_per_rollout_batch=4 (required for grpo_clip/no_clip)
#   --no-std                  disable group std normalization (Dr. GRPO variant)
#   --length-norm=TYPE        masked_mean (default) | masked_normalize
#   --prompt-type=TYPE        r1_zero (default) | question_only
#   --lr=VALUE                learning rate (default: 1e-5)
#   --epochs=N                epochs_per_rollout_batch (overrides --off-policy)
#   --train-batch-size=N      train_batch_size (default: 256)
#   --grad-accum=N            gradient_accumulation_steps (default: 128)
#
# NOTES:
#   Full runs require 2× H100s.
#   For off-policy sweeps, adjust --epochs, --train-batch-size, and --grad-accum together.

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SMOKE_TEST=0
DRY_RUN=0
LOSS_TYPE="reinforce_with_baseline"
OFF_POLICY=0
NO_STD=0
LENGTH_NORM="masked_mean"
PROMPT_TYPE="r1_zero"
LR="1e-5"
EPOCHS=""           # empty = use default (1 on-policy, 4 off-policy)
TRAIN_BATCH_SIZE=""
GRAD_ACCUM=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --smoke-test)          SMOKE_TEST=1 ;;
        --dry-run)             DRY_RUN=1 ;;
        --off-policy)          OFF_POLICY=1 ;;
        --no-std)              NO_STD=1 ;;
        --loss-type=*)         LOSS_TYPE="${arg#*=}" ;;
        --length-norm=*)       LENGTH_NORM="${arg#*=}" ;;
        --prompt-type=*)       PROMPT_TYPE="${arg#*=}" ;;
        --lr=*)                LR="${arg#*=}" ;;
        --epochs=*)            EPOCHS="${arg#*=}" ;;
        --train-batch-size=*)  TRAIN_BATCH_SIZE="${arg#*=}" ;;
        --grad-accum=*)        GRAD_ACCUM="${arg#*=}" ;;
        *)                     EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
MODEL_NAME="Qwen2.5-Math-1.5B"
CLUSTER_MODEL="/data/a5-alignment/models/${MODEL_NAME}"
LOCAL_MODEL="${ROOT}/assets/${MODEL_NAME}"

if [ -d "${CLUSTER_MODEL}" ]; then
    MODEL_PATH="${CLUSTER_MODEL}"
    echo "Using cluster model: ${MODEL_PATH}"
elif [ -d "${LOCAL_MODEL}" ]; then
    MODEL_PATH="${LOCAL_MODEL}"
    echo "Using local model: ${MODEL_PATH}"
else
    echo "INFO: Model not found locally. Downloading from HuggingFace..."
    mkdir -p "${ROOT}/assets"
    uv run huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
        --local-dir "${LOCAL_MODEL}"
    MODEL_PATH="${LOCAL_MODEL}"
fi

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
CLUSTER_DATA="/data/a5-alignment/MATH"
LOCAL_DATA="${ROOT}/data/math"

if [ -d "${CLUSTER_DATA}" ]; then
    DATA_DIR="${CLUSTER_DATA}"
    echo "Using cluster data: ${DATA_DIR}"
elif [ -d "${LOCAL_DATA}" ]; then
    DATA_DIR="${LOCAL_DATA}"
    echo "Using local data: ${DATA_DIR}"
else
    echo "ERROR: MATH data not found. Expected at ${CLUSTER_DATA} or ${LOCAL_DATA}"
    exit 1
fi

TRAIN_DATA="${DATA_DIR}/train.jsonl"
VAL_DATA="${DATA_DIR}/validation.jsonl"
# Smoke tests → results/section7/smoke (quick local checks)
# Full runs   → results/section8 (all Section 8 experiments)
if [ "${SMOKE_TEST}" -eq 1 ]; then
    OUTPUT_DIR="${ROOT}/results/section7/smoke"
else
    OUTPUT_DIR="${ROOT}/results/section8"
fi
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Build run arguments
# ---------------------------------------------------------------------------

# Resolve derived hyperparameters first
if [ -n "${EPOCHS}" ]; then
    EPOCHS_PER_ROLLOUT="${EPOCHS}"
elif [ "${OFF_POLICY}" -eq 1 ]; then
    EPOCHS_PER_ROLLOUT=4
else
    EPOCHS_PER_ROLLOUT=1
fi

RESOLVED_TRAIN_BS="${TRAIN_BATCH_SIZE:-256}"
# Keep micro_bs=2: grad_accum = train_bs / 2
DEFAULT_GRAD_ACCUM=$(( RESOLVED_TRAIN_BS / 2 ))
RESOLVED_GRAD_ACCUM="${GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}"

# Build run name — LR always included so runs from different sections never collide
LR_TAG=$(echo "${LR}" | sed 's/[^0-9e.-]//g')
RUN_NAME="grpo_${LOSS_TYPE}_lr${LR_TAG}"
[ "${EPOCHS_PER_ROLLOUT}" -gt 1 ]       && RUN_NAME="${RUN_NAME}_e${EPOCHS_PER_ROLLOUT}"
[ "${RESOLVED_TRAIN_BS}" != "256" ]     && RUN_NAME="${RUN_NAME}_bs${RESOLVED_TRAIN_BS}"
[ "${NO_STD}" -eq 1 ]                   && RUN_NAME="${RUN_NAME}_nostd"
[ "${LENGTH_NORM}" != "masked_mean" ]   && RUN_NAME="${RUN_NAME}_${LENGTH_NORM}"
[ "${PROMPT_TYPE}" != "r1_zero" ]       && RUN_NAME="${RUN_NAME}_${PROMPT_TYPE}"

# Optional flags
STD_ARG=""
[ "${NO_STD}" -eq 1 ] && STD_ARG="--no_std_normalization"

if [ "${SMOKE_TEST}" -eq 1 ]; then
    echo "=== SMOKE TEST MODE ==="
    COMMON_ARGS=(
        --model "${MODEL_PATH}"
        --data "${TRAIN_DATA}"
        --val_data "${VAL_DATA}"
        --output "${OUTPUT_DIR}"
        --n_grpo_steps 3
        --max_train_examples 64
        --group_size 4
        --rollout_batch_size 16
        --train_batch_size 16
        --gradient_accumulation_steps 8
        --max_response_tokens 256
        --n_eval_examples 32
        --eval_interval 1
        --epochs_per_rollout_batch "${EPOCHS_PER_ROLLOUT}"
        --loss_type "${LOSS_TYPE}"
        --length_norm "${LENGTH_NORM}"
        --prompt_type "${PROMPT_TYPE}"
        --run_name "${RUN_NAME}_smoke"
        --no_wandb
        --skip_eval
        --gradient_checkpointing
        ${STD_ARG}
        ${EXTRA_ARGS}
    )
else
    COMMON_ARGS=(
        --model "${MODEL_PATH}"
        --data "${TRAIN_DATA}"
        --val_data "${VAL_DATA}"
        --output "${OUTPUT_DIR}"
        --n_grpo_steps 200
        --group_size 8
        --rollout_batch_size 256
        --train_batch_size "${RESOLVED_TRAIN_BS}"
        --gradient_accumulation_steps "${RESOLVED_GRAD_ACCUM}"
        --lr "${LR}"
        --advantage_eps 1e-6
        --temperature 1.0
        --max_response_tokens 1024
        --gpu_memory_utilization 0.85
        --n_eval_examples 1024
        --eval_interval 5
        --epochs_per_rollout_batch "${EPOCHS_PER_ROLLOUT}"
        --loss_type "${LOSS_TYPE}"
        --length_norm "${LENGTH_NORM}"
        --prompt_type "${PROMPT_TYPE}"
        --run_name "${RUN_NAME}"
        ${STD_ARG}
        ${EXTRA_ARGS}
    )
fi

# ---------------------------------------------------------------------------
# Run GRPO training
# ---------------------------------------------------------------------------
echo ""
echo "=== Starting GRPO training ==="
echo "  run_name    : ${RUN_NAME}"
echo "  loss_type   : ${LOSS_TYPE}"
echo "  lr          : ${LR}"
echo "  length_norm : ${LENGTH_NORM}"
echo "  prompt_type : ${PROMPT_TYPE}"
echo "  epochs/batch: ${EPOCHS_PER_ROLLOUT}"
echo "  output      : ${OUTPUT_DIR}"
echo ""

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] uv run python train_grpo.py ${COMMON_ARGS[*]}"
    echo ""
    echo "Done (dry run). No training was performed."
    exit 0
fi

uv run python "${ROOT}/cs336_alignment/section7_grpo/train_grpo.py" "${COMMON_ARGS[@]}"

echo ""
echo "Done. Results in ${OUTPUT_DIR}"
