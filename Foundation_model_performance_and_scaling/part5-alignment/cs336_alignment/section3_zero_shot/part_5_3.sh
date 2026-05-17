#!/bin/bash
# ==============================================================================
# Section 3: Zero-shot MATH baseline evaluation
# ==============================================================================
#
# USAGE:
#   cd cs336_alignment/section3_zero_shot
#   ./part_5_3.sh
#
#   Override model or data paths via environment variables:
#   MODEL=/path/to/model DATA=/path/to/validation.jsonl ./part_5_3.sh
#
#   Limit examples for a quick local smoke test:
#   ./part_5_3.sh --max_examples 20
#
# WHAT IT DOES:
#   Evaluates Qwen 2.5 Math 1.5B zero-shot on the MATH validation set
#   using the r1_zero prompt and the DrGRPO reward function.
#   Prints per-category counts and example outputs, then serializes results.
#
# OUTPUT:
#   ${ROOT}/results/section3/zero_shot_eval.jsonl  — one JSON object per example
#   Fields: prompt, response, ground_truth, format_reward, answer_reward, reward
#
# NOTES:
#   Requires a GPU with enough VRAM for Qwen 2.5 Math 1.5B (>=8 GB).
#   On the Together cluster the model and data are pre-downloaded at
#   /data/a5-alignment/; no download needed.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

# --- Model path ---
# Prefer the pre-downloaded cluster copy; fall back to HuggingFace auto-download.
if [ -z "${MODEL}" ]; then
    if [ -d "/data/a5-alignment/models/Qwen2.5-Math-1.5B" ]; then
        MODEL="/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    else
        MODEL="Qwen/Qwen2.5-Math-1.5B"
        echo "INFO: Cluster model not found, will download from HuggingFace: ${MODEL}"
    fi
fi

# --- Data path ---
# Priority: cluster MATH > local MATH > local GSM8K (smoke test)
CLUSTER_DATA="/data/a5-alignment/MATH/validation.jsonl"
LOCAL_MATH="${ROOT}/data/math/validation.jsonl"
LOCAL_GSM8K="${ROOT}/data/gsm8k/test.jsonl"
if [ -z "${DATA}" ]; then
    if [ -f "${CLUSTER_DATA}" ]; then
        DATA="${CLUSTER_DATA}"
    elif [ -f "${LOCAL_MATH}" ]; then
        DATA="${LOCAL_MATH}"
    elif [ -f "${LOCAL_GSM8K}" ]; then
        DATA="${LOCAL_GSM8K}"
        echo "INFO: MATH data not found, falling back to GSM8K for local smoke test: ${DATA}"
    else
        echo "ERROR: No evaluation data found."
        echo "  Cluster MATH: ${CLUSTER_DATA}"
        echo "  Local MATH:   ${LOCAL_MATH}"
        echo "  Local GSM8K:  ${LOCAL_GSM8K}"
        exit 1
    fi
fi

OUTPUT="${ROOT}/results/section3/zero_shot_eval.jsonl"

echo "==> Section 3: Zero-shot MATH baseline"
echo "    Model:  ${MODEL}"
echo "    Data:   ${DATA}"
echo "    Output: ${OUTPUT}"
echo ""

uv run python "${ROOT}/cs336_alignment/section3_zero_shot/evaluate_zero_shot.py" \
    --model  "${MODEL}" \
    --data   "${DATA}" \
    --output "${OUTPUT}" \
    "$@"

echo ""
echo "==> Done. Results at ${OUTPUT}"