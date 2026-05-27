#!/usr/bin/env bash
# USAGE:   bash part_6_2.sh [--smoke-test] [--dry-run] [--skip-alpaca] [--skip-safety]
#
# WHAT IT DOES:
#   Section 2 — Zero-Shot Baselines.
#   Evaluates Llama 3.1 8B zero-shot performance on four benchmarks:
#     - MMLU          (factual knowledge, multiple-choice)
#     - GSM8K         (math reasoning, numeric answer)
#     - AlpacaEval    (chatbot quality, collect predictions for offline eval)
#     - SimpleSafetyTests (safety, collect predictions for offline eval)
#
#   AlpacaEval and SimpleSafetyTests require a separate annotator step using
#   Llama 3.3 70B Instruct (2 GPUs, >80 GB each). Use --skip-alpaca and
#   --skip-safety to skip those collection steps if not yet ready.
#
# OUTPUT:
#   results/section2/eval_mmlu_baseline.jsonl
#   results/section2/eval_mmlu_baseline.summary.json
#   results/section2/eval_gsm8k_baseline.jsonl
#   results/section2/eval_gsm8k_baseline.summary.json
#   results/section2/alpaca_eval_baseline.json
#   results/section2/sst_baseline.jsonl
#
# NOTES:
#   Requires 1 A100/H100 for MMLU, GSM8K, AlpacaEval, SST collection.
#   Annotator steps (AlpacaEval winrate, safety scoring) require 2 GPUs.
#   Model is loaded fresh per benchmark — each run takes ~5–15 mins.

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

SMOKE_FLAG=""
DRY_RUN=0
SKIP_ALPACA=0
SKIP_SAFETY=0

for arg in "$@"; do
    case $arg in
        --smoke-test)   SMOKE_FLAG="--smoke-test" ;;
        --dry-run)      DRY_RUN=1 ;;
        --skip-alpaca)  SKIP_ALPACA=1 ;;
        --skip-safety)  SKIP_SAFETY=1 ;;
    esac
done

MODEL_PATH="/data/a5-alignment/models/Llama-3.1-8B"
RESULTS_DIR="${ROOT}/results/section2"

echo "========================================"
echo "  Section 2 — Zero-Shot Baselines"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${RESULTS_DIR}"
[ -n "${SMOKE_FLAG}" ] && echo "  Mode: SMOKE TEST"
echo "========================================"
echo ""

# ── MMLU ──────────────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  §2.1 — MMLU zero-shot baseline"
echo "------------------------------------------------------------"
CMD="uv run python ${ROOT}/cs336_alignment/section2_zero_shot/evaluate_mmlu.py \
    --model-path ${MODEL_PATH} \
    --data-dir ${ROOT}/data/mmlu/test \
    --output-path ${RESULTS_DIR}/eval_mmlu_baseline.jsonl \
    ${SMOKE_FLAG}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
echo ""

# ── GSM8K ─────────────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  §2.2 — GSM8K zero-shot baseline"
echo "------------------------------------------------------------"
CMD="uv run python ${ROOT}/cs336_alignment/section2_zero_shot/evaluate_gsm8k.py \
    --model-path ${MODEL_PATH} \
    --data-path ${ROOT}/data/gsm8k/test.jsonl \
    --output-path ${RESULTS_DIR}/eval_gsm8k_baseline.jsonl \
    ${SMOKE_FLAG}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
echo ""

# ── AlpacaEval ────────────────────────────────────────────────────────────────
if [ "${SKIP_ALPACA}" -eq 0 ]; then
    echo "------------------------------------------------------------"
    echo "  §2.3 — AlpacaEval prediction collection"
    echo "------------------------------------------------------------"
    CMD="uv run python ${ROOT}/cs336_alignment/section2_zero_shot/evaluate_alpaca_eval.py \
    --model-path ${MODEL_PATH} \
    --data-path ${ROOT}/data/alpaca_eval/alpaca_eval.jsonl \
    --output-path ${RESULTS_DIR}/alpaca_eval_baseline.json \
    --generator llama-3.1-8b-base \
    ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi

    echo ""
    echo "  To compute AlpacaEval winrate (requires 2 GPUs, >80 GB each):"
    echo "  uv run alpaca_eval \\"
    echo "      --model_outputs ${RESULTS_DIR}/alpaca_eval_baseline.json \\"
    echo "      --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \\"
    echo "      --base-dir '.'"
    echo ""
fi

# ── SimpleSafetyTests ─────────────────────────────────────────────────────────
if [ "${SKIP_SAFETY}" -eq 0 ]; then
    echo "------------------------------------------------------------"
    echo "  §2.4 — SimpleSafetyTests prediction collection"
    echo "------------------------------------------------------------"
    CMD="uv run python ${ROOT}/cs336_alignment/section2_zero_shot/evaluate_sst.py \
    --model-path ${MODEL_PATH} \
    --data-path ${ROOT}/data/simple_safety_tests/simple_safety_tests.csv \
    --output-path ${RESULTS_DIR}/sst_baseline.jsonl \
    ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi

    echo ""
    echo "  To score safety (requires 2 GPUs, >80 GB each):"
    echo "  uv run python scripts/evaluate_safety.py \\"
    echo "      --input-path ${RESULTS_DIR}/sst_baseline.jsonl \\"
    echo "      --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \\"
    echo "      --num-gpus 2 \\"
    echo "      --output-path ${RESULTS_DIR}/sst_baseline_annotated.jsonl"
    echo ""
fi

echo "========================================"
echo "  Done. Results in ${RESULTS_DIR}"
echo "========================================"
