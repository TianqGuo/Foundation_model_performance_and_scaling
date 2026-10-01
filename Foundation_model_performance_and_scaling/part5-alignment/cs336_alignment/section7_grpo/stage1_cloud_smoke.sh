#!/usr/bin/env bash
# Run both Stage 1 smoke checks on a cloud machine, never on the local laptop.
# USAGE (from any directory, with the cloud environment already installed):
#   bash cs336_alignment/section7_grpo/stage1_cloud_smoke.sh
# Finds model/data in the same locations as part_5_7.sh; downloads a missing
# model on the cloud. MODEL_PATH/TRAIN_DATA/VAL_DATA are optional overrides.
# OUTPUT: unique validation folder under results/section7/stage1_cloud/,
# console logs, two isolated run folders, configs, eval records and checkpoints.
# Requires two visible cloud GPUs and the existing compatible legacy stack.
# W&B is disabled. Uses .venv directly to preserve manually installed wheels.

set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
test -x "${PYTHON}"

"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("Stage 1 cloud validation requires two visible CUDA GPUs.")
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
            "${ROOT}/.venv/bin/huggingface-cli" download Qwen/Qwen2.5-Math-1.5B \
                --local-dir "${MODEL_PATH}"
        fi
    fi
fi
echo "Model: ${MODEL_PATH}"
echo "Training data: ${TRAIN_DATA}"
echo "Validation data: ${VAL_DATA}"

mkdir -p "${ROOT}/results/section7/stage1_cloud"
OUTPUT_DIR="$(mktemp -d "${ROOT}/results/section7/stage1_cloud/validation_XXXXXXXX")"
echo "Validation outputs: ${OUTPUT_DIR}"

COMMON_ARGS=(
    --model "${MODEL_PATH}" --data "${TRAIN_DATA}" --val_data "${VAL_DATA}"
    --output "${OUTPUT_DIR}" --n_grpo_steps 3 --max_train_examples 64
    --group_size 4 --rollout_batch_size 16 --train_batch_size 16
    --gradient_accumulation_steps 8 --max_response_tokens 256
    --n_eval_examples 32 --eval_interval 1 --eval_seed 12345
    --train_device cuda:0 --vllm_device cuda:1 --no_wandb
)

echo "Running fixed-subset evaluation smoke check..."
"${PYTHON}" -m cs336_alignment.section7_grpo.train_grpo \
    "${COMMON_ARGS[@]}" --run_name stage1_fixed_eval \
    2>&1 | tee "${OUTPUT_DIR}/fixed_eval_console.log"

echo "Running skip-evaluation smoke check..."
"${PYTHON}" -m cs336_alignment.section7_grpo.train_grpo \
    "${COMMON_ARGS[@]}" --run_name stage1_skip_eval --skip_eval \
    2>&1 | tee "${OUTPUT_DIR}/skip_eval_console.log"

"${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json
from pathlib import Path
import sys

base = Path(sys.argv[1])
for name, skip in (("stage1_fixed_eval", False), ("stage1_skip_eval", True)):
    matches = list(base.glob(f"{name}_*"))
    assert len(matches) == 1, (name, matches)
    run = matches[0]
    config = json.loads((run / "run_config.json").read_text())
    subset = json.loads((run / "evaluation_subset.json").read_text())
    records = [json.loads(line) for line in (run / f"eval_metrics_{name}.jsonl").read_text().splitlines()]
    assert config["args"]["skip_eval"] == skip
    assert config["resolved"]["train_device"] == "cuda:0"
    assert config["resolved"]["vllm_device"] == "cuda:1"
    assert config["resolved"]["micro_batch_size"] == 2
    checkpoint = Path(config["checkpoint"])
    assert (checkpoint / "config.json").is_file(), checkpoint
    assert list(checkpoint.glob("*.safetensors")) or list(checkpoint.glob("*.bin")), checkpoint
    if skip:
        assert subset["protocol"] == "disabled" and not subset["indices"]
        assert not records and not (run / "final_eval.json").exists()
    else:
        assert subset["protocol"] == "fixed_subset_v1"
        assert 0 < len(subset["indices"]) <= 32
        assert len(set(subset["indices"])) == len(subset["indices"])
        assert [r["grpo_step"] for r in records] == [1, 2, 3]
        assert (run / "final_eval.json").is_file()
    print(f"PASS: {name}: {run}")
print("Both cloud smoke runs completed. These are execution checks, not accuracy benchmarks.")
PY

echo "Keep this validation folder and both checkpoint paths before deleting the instance."
echo "Results saved to ${OUTPUT_DIR}"
