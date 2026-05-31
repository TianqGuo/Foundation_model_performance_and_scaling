#!/usr/bin/env bash
# =============================================================================
# USAGE:   bash cs336_alignment/section4_eval/part_6_4.sh [--smoke-test] [--skip-judges]
#
# WHAT IT DOES:
#   Evaluates the SFT-tuned Llama 3.1 8B model on all four benchmarks using
#   the Alpaca prompt format (same format used during SFT training), enabling
#   direct comparison against the zero-shot baseline from §2.
#
#   §4.1 MMLU          — rule-based scoring, single GPU, ~10 min
#   §4.2 GSM8K         — rule-based scoring, single GPU, ~2 min
#   §4.3 AlpacaEval    — generation on single GPU, then judge requires 2× 80 GB
#   §4.4 SST           — generation on single GPU, then judge requires 2× 80 GB
#
# OUTPUT:
#   results/section4/eval_mmlu_sft.jsonl / .summary.json
#   results/section4/eval_gsm8k_sft.jsonl / .summary.json
#   results/section4/alpaca_eval_sft.json
#   results/section4/sst_sft.jsonl
#   results/section4/sst_sft_annotated.jsonl   (if judge runs)
#   results/section4/leaderboard.csv            (after alpaca_eval)
#
# NOTES:
#   --smoke-test   runs each eval on a tiny subset for quick validation.
#   --skip-judges  skips AlpacaEval and SST annotation (which need 2× 80 GB GPUs).
#                  Use this if you only have 1 GPU available.
#   The SFT checkpoint must exist at assets/sft_ultrachat/ before running.
# =============================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SMOKE_TEST=0
SKIP_JUDGES=0
for arg in "$@"; do
    case "$arg" in
        --smoke-test)   SMOKE_TEST=1 ;;
        --skip-judges)  SKIP_JUDGES=1 ;;
    esac
done

SMOKE_FLAG=""
[ "${SMOKE_TEST}" -eq 1 ] && SMOKE_FLAG="--smoke-test"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLUSTER_MODEL="/data/a5-alignment/models/Llama-3.1-8B"
CLUSTER_SFT="/data/a5-alignment/sft_ultrachat"
LOCAL_SFT="${ROOT}/assets/sft_ultrachat"

RESULTS_DIR="${ROOT}/results/section4"
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# SFT checkpoint
# ---------------------------------------------------------------------------
echo "==> Checking SFT checkpoint ..."
if [ -f "${CLUSTER_SFT}/config.json" ] && [ -f "${CLUSTER_SFT}/tokenizer_config.json" ]; then
    SFT_MODEL="${CLUSTER_SFT}"
elif [ -f "${LOCAL_SFT}/config.json" ] && [ -f "${LOCAL_SFT}/tokenizer_config.json" ]; then
    SFT_MODEL="${LOCAL_SFT}"
else
    echo "  ERROR: SFT checkpoint not found at ${LOCAL_SFT}" >&2
    echo "  Run bash cs336_alignment/section3_sft/part_6_3.sh first." >&2
    exit 1
fi
echo "    Using checkpoint: ${SFT_MODEL}"

# ---------------------------------------------------------------------------
# §4.1 — MMLU
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.1 MMLU ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_mmlu_sft.py" \
    --model-path    "${SFT_MODEL}" \
    --data-dir      "${ROOT}/data/mmlu/test" \
    --output-path   "${RESULTS_DIR}/eval_mmlu_sft.jsonl" \
    ${SMOKE_FLAG}

# ---------------------------------------------------------------------------
# §4.2 — GSM8K
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.2 GSM8K ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_gsm8k_sft.py" \
    --model-path    "${SFT_MODEL}" \
    --data-path     "${ROOT}/data/gsm8k/test.jsonl" \
    --output-path   "${RESULTS_DIR}/eval_gsm8k_sft.jsonl" \
    ${SMOKE_FLAG}

if [ "${SKIP_JUDGES}" -eq 1 ]; then
    echo ""
    echo "==> Skipping AlpacaEval and SST annotation (--skip-judges set)."
    echo "    Run without --skip-judges on a 2× 80 GB instance to complete §4.3 and §4.4."
    exit 0
fi

# ---------------------------------------------------------------------------
# §4.3 — AlpacaEval generation
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.3 AlpacaEval — generating outputs ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_alpaca_sft.py" \
    --model-path    "${SFT_MODEL}" \
    --data-path     "${ROOT}/data/alpaca_eval/alpaca_eval.jsonl" \
    --output-path   "${RESULTS_DIR}/alpaca_eval_sft.json" \
    --generator     "llama-3.1-8b-sft" \
    ${SMOKE_FLAG}

# AlpacaEval evaluator — patch annotator model path and run from project root
echo ""
echo "==> §4.3 AlpacaEval — running judge (Llama 3.3 70B) ..."
CLUSTER_ANN="/data/a5-alignment/models/Llama-3.3-70B-Instruct"
LOCAL_ANN="${ROOT}/assets/Llama-3.3-70B-Instruct"
if [ -f "${CLUSTER_ANN}/config.json" ]; then
    ANN_PATH="${CLUSTER_ANN}"
elif [ -f "${LOCAL_ANN}/config.json" ]; then
    ANN_PATH="${LOCAL_ANN}"
else
    echo "  ERROR: Llama 3.3 70B Instruct not found. Run bash get_assets.sh --annotator" >&2
    exit 1
fi

CONFIGS_YAML="${ROOT}/scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml"
sed -i "s|ANNOTATOR_MODEL_PATH|${ANN_PATH}|g" "${CONFIGS_YAML}"

# Convert JSONL reference to JSON array (alpaca_eval requires .json)
ALPACA_REF_JSONL="${ROOT}/data/alpaca_eval/alpaca_eval.jsonl"
ALPACA_REF_JSON="${ROOT}/data/alpaca_eval/alpaca_eval_ref.json"
python3 -c "
import json, sys
lines = [json.loads(l) for l in open('${ALPACA_REF_JSONL}') if l.strip()]
json.dump(lines, open('${ALPACA_REF_JSON}', 'w'))
print(f'Converted {len(lines)} examples to ${ALPACA_REF_JSON}')
"

pushd "${ROOT}" > /dev/null
uv run alpaca_eval \
    --model_outputs "${RESULTS_DIR}/alpaca_eval_sft.json" \
    --annotators_config "scripts/alpaca_eval_vllm_llama3_3_70b_fn" \
    --base-dir "." \
    --reference_outputs "${ALPACA_REF_JSON}" \
    --output_path "${RESULTS_DIR}"
popd > /dev/null

# Restore placeholder in configs.yaml
sed -i "s|${ANN_PATH}|ANNOTATOR_MODEL_PATH|g" "${CONFIGS_YAML}"

# ---------------------------------------------------------------------------
# §4.4 — SimpleSafetyTests generation + annotation
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.4 SimpleSafetyTests — generating outputs ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_sst_sft.py" \
    --model-path    "${SFT_MODEL}" \
    --data-path     "${ROOT}/data/simple_safety_tests/simple_safety_tests.csv" \
    --output-path   "${RESULTS_DIR}/sst_sft.jsonl" \
    ${SMOKE_FLAG}

echo ""
echo "==> §4.4 SimpleSafetyTests — running safety annotator ..."
uv run python "${ROOT}/scripts/evaluate_safety.py" \
    --input-path    "${RESULTS_DIR}/sst_sft.jsonl" \
    --model-name-or-path "${ANN_PATH}" \
    --num-gpus      2 \
    --output-path   "${RESULTS_DIR}/sst_sft_annotated.jsonl"

echo ""
echo "==> §4 evaluation complete. Results in ${RESULTS_DIR}"
echo "    Run the comparison plots:"
echo "    uv run python cs336_alignment/section2_zero_shot/plot_zero_shot_results.py"
