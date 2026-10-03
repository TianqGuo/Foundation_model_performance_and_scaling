#!/usr/bin/env bash
# CLOUD ONLY: matched GRPO-Clip / GSPO runs in the validated Stage 2 environment.
# USAGE: bash cs336_alignment/section7_grpo/stage3_cloud_compare.sh [--pilot] [--gspo-tight]
# Default: two 3-step smoke runs, 16 rollouts, 2 optimization epochs, eval32.
# --pilot: two 20-step pilot runs, 64 rollouts, 4 epochs, fixed eval128 every5.
# --gspo-tight: exploratory GSPO bounds 3e-4/4e-4; GRPO remains .2.
# Same model/data/prompt/seeds/batches/cliprange=.2; only importance weighting differs.
# A pilot is not evidence of algorithm superiority. W&B disabled; .venv-server.
# Usual paths auto-discovered; MODEL_PATH/TRAIN_DATA/VAL_DATA optional overrides.
# Outputs/checkpoints are isolated; restore cloud MATH data before running.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv-server/bin/python"
test -x "${PYTHON}"

"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("Stage 3 comparison requires two visible CUDA GPUs.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
PY

if [ -z "${TRAIN_DATA:-}" ] || [ -z "${VAL_DATA:-}" ]; then
    if [ -d /data/a5-alignment/MATH ]; then
        DATA_DIR=/data/a5-alignment/MATH
    elif [ -d "${ROOT}/data/math" ]; then
        DATA_DIR="${ROOT}/data/math"
    else
        echo "ERROR: MATH data not found at /data/a5-alignment/MATH or ${ROOT}/data/math."
        echo "Restore the same data as previous runs, or set optional TRAIN_DATA and VAL_DATA overrides."
        exit 1
    fi
    TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.jsonl}"
    VAL_DATA="${VAL_DATA:-${DATA_DIR}/validation.jsonl}"
fi
for DATA_FILE in "${TRAIN_DATA}" "${VAL_DATA}"; do
    if [ ! -f "${DATA_FILE}" ]; then
        echo "ERROR: Required data file missing: ${DATA_FILE}"
        exit 1
    fi
done

if [ -z "${MODEL_PATH:-}" ]; then
    if [ -d /data/a5-alignment/models/Qwen2.5-Math-1.5B ]; then
        MODEL_PATH=/data/a5-alignment/models/Qwen2.5-Math-1.5B
    else
        MODEL_PATH="${ROOT}/assets/Qwen2.5-Math-1.5B"
        if [ ! -d "${MODEL_PATH}" ]; then
            echo "Downloading model on the cloud to ${MODEL_PATH}..."
            "${PYTHON}" - "${MODEL_PATH}" <<'PYMODEL'
from huggingface_hub import snapshot_download
import sys
snapshot_download("Qwen/Qwen2.5-Math-1.5B", local_dir=sys.argv[1])
PYMODEL
        fi
    fi
fi
echo "Model: ${MODEL_PATH}"
echo "Training data: ${TRAIN_DATA}"
echo "Validation data: ${VAL_DATA}"


MODE=smoke
STEPS=3
ROLLOUT=16
TRAIN_BATCH=16
EPOCHS=2
EVAL=32
INTERVAL=1
TIME_LIMIT=20m
TIGHT=0
for OPTION in "$@"; do
if [ "${OPTION}" = --pilot ]; then
    MODE=pilot; STEPS=20; ROLLOUT=64; TRAIN_BATCH=32; EPOCHS=4
    EVAL=128; INTERVAL=5; TIME_LIMIT=90m
elif [ "${OPTION}" = --gspo-tight ]; then
    TIGHT=1
else
    echo 'Usage: stage3_cloud_compare.sh [--pilot] [--gspo-tight]'; exit 2
fi
done
COMPARE_ARGS=()
if [ "${TIGHT}" = 1 ]; then
    MODE="tight_${MODE}"
    COMPARE_ARGS=(--allow-clipping-difference)
fi
mkdir -p "${ROOT}/results/section7/stage3_cloud"
OUTPUT_DIR="$(mktemp -d "${ROOT}/results/section7/stage3_cloud/${MODE}_XXXXXXXX")"
COMMON_ARGS=(
    --model "${MODEL_PATH}" --data "${TRAIN_DATA}" --val_data "${VAL_DATA}"
    --output "${OUTPUT_DIR}" --n_grpo_steps "${STEPS}" --max_train_examples 1024
    --group_size 4 --rollout_batch_size "${ROLLOUT}" --train_batch_size "${TRAIN_BATCH}"
    --gradient_accumulation_steps "$((TRAIN_BATCH / 2))" --epochs_per_rollout_batch "${EPOCHS}"
    --max_response_tokens 256 --n_eval_examples "${EVAL}" --eval_interval "${INTERVAL}"
    --seed 42 --eval_seed 12345 --train_device cuda:0 --vllm_device cuda:1
    --rollout_backend server --server_enforce_eager --no_wandb
    --loss_type grpo_clip --baseline mean --advantage_normalizer std
    --loss_normalization sequence --cliprange 0.2
)
for METHOD in grpo gspo; do
    METHOD_ARGS=()
    if [ "${TIGHT}" = 1 ] && [ "${METHOD}" = gspo ]; then
        METHOD_ARGS=(--cliprange_low 0.0003 --cliprange_high 0.0004)
    fi
    echo "Running ${MODE}: ${METHOD}"
    timeout --signal=TERM --kill-after=30s "${TIME_LIMIT}" "${PYTHON}" \
      -m cs336_alignment.section7_grpo.train_grpo "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}" \
      --importance_reweighting "${METHOD}" --run_name "stage3_${METHOD}_${MODE}" \
      2>&1 | tee "${OUTPUT_DIR}/${METHOD}_console.log"
done
"${PYTHON}" -m cs336_alignment.section7_grpo.compare_runs "${OUTPUT_DIR}" "${COMPARE_ARGS[@]}"
echo "Keep this folder before destroying the instance: ${OUTPUT_DIR}"
