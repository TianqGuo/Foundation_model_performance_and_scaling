#!/usr/bin/env bash
# USAGE:   bash part_5_7.sh [--smoke-test] [--loss-type TYPE] [--off-policy]
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
# NOTES:
#   Default run requires 2× H100s (~6-8 hrs for 200 steps).
#   --smoke-test runs 3 GRPO steps on 64 train examples (single GPU OK).
#   --loss-type  choices: reinforce_with_baseline (default), no_baseline, grpo_clip
#   --off-policy sets epochs_per_rollout_batch=4 (required for grpo_clip).

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SMOKE_TEST=0
LOSS_TYPE="reinforce_with_baseline"
OFF_POLICY=0
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --smoke-test)   SMOKE_TEST=1 ;;
        --off-policy)   OFF_POLICY=1 ;;
        --loss-type=*)  LOSS_TYPE="${arg#*=}" ;;
        --loss-type)    shift; LOSS_TYPE="$1" ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
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
    echo "ERROR: Model not found at ${CLUSTER_MODEL} or ${LOCAL_MODEL}"
    echo "Run get_assets.sh first to download Qwen2.5-Math-1.5B."
    exit 1
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
OUTPUT_DIR="${ROOT}/results/section7"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Build run arguments
# ---------------------------------------------------------------------------
RUN_NAME="grpo_${LOSS_TYPE}"
EPOCHS_ARG=""
if [ "${OFF_POLICY}" -eq 1 ]; then
    EPOCHS_ARG="--epochs_per_rollout_batch 4"
    RUN_NAME="${RUN_NAME}_offpolicy"
fi

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
        --loss_type "${LOSS_TYPE}"
        --run_name "${RUN_NAME}_smoke"
        --no_wandb
        --skip_eval
        --gradient_checkpointing
        ${EPOCHS_ARG}
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
        --train_batch_size 256
        --gradient_accumulation_steps 128
        --lr 1e-5
        --advantage_eps 1e-6
        --temperature 1.0
        --max_response_tokens 1024
        --gpu_memory_utilization 0.85
        --n_eval_examples 1024
        --eval_interval 5
        --loss_type "${LOSS_TYPE}"
        --run_name "${RUN_NAME}"
        ${EPOCHS_ARG}
        ${EXTRA_ARGS}
    )
fi

# ---------------------------------------------------------------------------
# Run GRPO training
# ---------------------------------------------------------------------------
echo ""
echo "=== Starting GRPO training ==="
echo "  loss_type   : ${LOSS_TYPE}"
echo "  off_policy  : ${OFF_POLICY}"
echo "  output      : ${OUTPUT_DIR}"
echo ""

uv run python "${ROOT}/cs336_alignment/section7_grpo/train_grpo.py" "${COMMON_ARGS[@]}"

echo ""
echo "=== Plotting results ==="
uv run python "${ROOT}/cs336_alignment/section7_grpo/plot_grpo_results.py" \
    --results_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Done. Results in ${OUTPUT_DIR}"
