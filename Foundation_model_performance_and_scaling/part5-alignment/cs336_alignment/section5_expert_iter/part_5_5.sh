#!/bin/bash
# ==============================================================================
# Section 5: Expert Iteration (EI) for MATH
# ==============================================================================
#
# USAGE:
#   cd cs336_alignment/section5_expert_iter
#   ./part_5_5.sh                     # all 3 EI runs (G=1,4,16) on 2 GPUs
#   ./part_5_5.sh --train-only        # skip tests, run all training
#   ./part_5_5.sh --smoke-test        # single-GPU smoke test (32 examples, 1 step)
#   ./part_5_5.sh --G 4               # single run with specified G value
#
# WHAT IT DOES:
#   Runs Expert Iteration on Qwen 2.5 Math 1.5B starting from the base model
#   (not the SFT checkpoint). Each EI step: rollout G responses per training
#   question with vLLM, keep those with reward > 0, fine-tune with SFT.
#   Repeats for n_ei_steps=5. Experiments vary G ∈ {1, 4, 16}.
#
# OUTPUT:
#   ${ROOT}/results/section5/eval_metrics_{run_name}.jsonl  — per EI step metrics
#   ${ROOT}/results/section5/ei_accuracy.png                — accuracy curves
#   ${ROOT}/results/section5/ei_entropy.png                 — entropy curves
#   ${ROOT}/results/section5/ei_rollout_size.png            — rollout size per step
#
# NOTES:
#   Requires 2 GPUs: policy on cuda:0, vLLM on cuda:1.
#   G=16 generates ~120k responses per EI step (7499 questions × 16).
#   Estimated runtime: ~4 H100 hours for all 3 runs.
#   --smoke-test runs on a single GPU with 32 questions, 1 EI step, no eval.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

TRAIN_ONLY=false
SMOKE_TEST=false
SINGLE_G=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-only)  TRAIN_ONLY=true ;;
        --smoke-test)  SMOKE_TEST=true ;;
        --G)           SINGLE_G="$2"; shift ;;
        --G=*)         SINGLE_G="${1#--G=}" ;;
    esac
    shift
done

# --- Model path ---
CLUSTER_MODEL="/data/a5-alignment/models/Qwen2.5-Math-1.5B"
LOCAL_MODEL="${ROOT}/assets/Qwen2.5-Math-1.5B"
if [ -z "${MODEL}" ]; then
    if   [ -d "${CLUSTER_MODEL}" ]; then MODEL="${CLUSTER_MODEL}"
    elif [ -d "${LOCAL_MODEL}" ];   then MODEL="${LOCAL_MODEL}"
    else
        echo "INFO: Model not found locally. Downloading from HuggingFace..."
        mkdir -p "${ROOT}/assets"
        uv run huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
            --local-dir "${LOCAL_MODEL}"
        MODEL="${LOCAL_MODEL}"
    fi
fi

# --- Data paths ---
CLUSTER_TRAIN="/data/a5-alignment/MATH/train.jsonl"
CLUSTER_VAL="/data/a5-alignment/MATH/validation.jsonl"
LOCAL_TRAIN="${ROOT}/data/math/train.jsonl"
LOCAL_VAL="${ROOT}/data/math/validation.jsonl"

if [ -z "${DATA}" ]; then
    if   [ -f "${CLUSTER_TRAIN}" ]; then DATA="${CLUSTER_TRAIN}"
    elif [ -f "${LOCAL_TRAIN}" ];   then DATA="${LOCAL_TRAIN}"
    else echo "ERROR: train.jsonl not found at ${CLUSTER_TRAIN} or ${LOCAL_TRAIN}"; exit 1
    fi
fi

if [ -z "${VAL_DATA}" ]; then
    if   [ -f "${CLUSTER_VAL}" ]; then VAL_DATA="${CLUSTER_VAL}"
    elif [ -f "${LOCAL_VAL}" ];   then VAL_DATA="${LOCAL_VAL}"
    else VAL_DATA="${CLUSTER_VAL}"
    fi
fi

OUTPUT="${ROOT}/results/section5"

echo "==> Section 5: Expert Iteration for MATH"
echo "    Model:    ${MODEL}"
echo "    Data:     ${DATA}"
echo "    Val data: ${VAL_DATA}"
echo "    Output:   ${OUTPUT}"
echo ""

# ---------------------------------------------------------------------------
# Smoke test: single GPU, 32 questions, 1 EI step, no eval
# ---------------------------------------------------------------------------
if [ "${SMOKE_TEST}" = true ]; then
    echo "==> [smoke-test] EI smoke test (32 questions, G=1, 1 step)"
    echo "    Single-GPU mode: vLLM and policy share cuda:0 (auto-detected)"
    uv run python "${ROOT}/cs336_alignment/section5_expert_iter/train_expert_iter.py" \
        --model    "${MODEL}" \
        --data     "${DATA}" \
        --val_data "${VAL_DATA}" \
        --output   "${OUTPUT}" \
        --max_train_examples 32 \
        --n_ei_steps 1 \
        --G 1 \
        --n_eval_examples 50 \
        --no_wandb \
        --run_name "ei_smoke"
    echo "==> Smoke test done. Results at ${OUTPUT}"
    exit 0
fi

TRAIN_CMD="uv run python ${ROOT}/cs336_alignment/section5_expert_iter/train_expert_iter.py
    --model    ${MODEL}
    --data     ${DATA}
    --val_data ${VAL_DATA}
    --output   ${OUTPUT}
    --n_ei_steps 5"

# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------
if [ -n "${SINGLE_G}" ]; then
    echo "==> Running single EI experiment: G=${SINGLE_G}"
    ${TRAIN_CMD} --G "${SINGLE_G}" --run_name "ei_g${SINGLE_G}"
    echo ""
else
    for G in 1 4 16; do
        echo "--- EI with G=${G} rollouts per question ---"
        ${TRAIN_CMD} --G ${G} --run_name "ei_g${G}"
        echo ""
    done
fi

echo "==> All EI experiments done. Generating plots..."
uv run python "${ROOT}/cs336_alignment/section5_expert_iter/plot_ei_results.py" \
    --results "${OUTPUT}"
echo "==> Results at ${OUTPUT}"