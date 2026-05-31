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
#   §4.3 AlpacaEval    — generation on single GPU + judge requires 2× 80 GB
#   §4.4 SST           — generation on single GPU + judge requires 2× 80 GB
#
# MODEL RESOLUTION (in priority order):
#   SFT checkpoint:  cluster path → assets/sft_ultrachat → HuggingFace Hub (sclion/llama-3.1-8b-sft-ultrachat)
#   70B judge:       cluster path → assets/Llama-3.3-70B-Instruct → HuggingFace Hub (meta-llama/Llama-3.3-70B-Instruct)
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
#   --skip-judges  skips AlpacaEval and SST annotation (1 GPU only).
#   Data files are reused from §2 — no re-download needed.
# =============================================================================
set -e
ulimit -n 65536 2>/dev/null || true
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
# resolve_model: cluster → local → HuggingFace Hub download
# A valid model dir must have both config.json AND tokenizer_config.json.
# ---------------------------------------------------------------------------
resolve_model() {
    local cluster="$1" local_path="$2" hf_repo="$3"
    _model_complete() { [ -f "${1}/config.json" ] && [ -f "${1}/tokenizer_config.json" ]; }
    if _model_complete "${cluster}"; then
        echo "${cluster}"
    elif _model_complete "${local_path}"; then
        echo "${local_path}"
    else
        echo "INFO: Downloading ${hf_repo} -> ${local_path}" >&2
        mkdir -p "$(dirname "${local_path}")"
        local attempt=0
        while true; do
            attempt=$(( attempt + 1 ))
            if uv run huggingface-cli download "${hf_repo}" \
                    --local-dir "${local_path}" --max-workers 2 >&2; then
                break
            fi
            if [ "${attempt}" -ge 3 ]; then
                echo "ERROR: Failed to download ${hf_repo} after 3 attempts." >&2
                exit 1
            fi
            echo "INFO: Attempt ${attempt} failed, retrying..." >&2
            sleep 5
        done
        echo "${local_path}"
    fi
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR="${ROOT}/results/section4"
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# SFT checkpoint — falls back to HuggingFace Hub (private repo)
# ---------------------------------------------------------------------------
echo "==> Resolving SFT checkpoint ..."
SFT_MODEL=$(resolve_model \
    "/data/a5-alignment/sft_ultrachat" \
    "${ROOT}/assets/sft_ultrachat" \
    "sclion/llama-3.1-8b-sft-ultrachat")
echo "    ${SFT_MODEL}"

# ---------------------------------------------------------------------------
# §4.1 — MMLU
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.1 MMLU ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_mmlu_sft.py" \
    --model-path  "${SFT_MODEL}" \
    --data-dir    "${ROOT}/data/mmlu/test" \
    --output-path "${RESULTS_DIR}/eval_mmlu_sft.jsonl" \
    ${SMOKE_FLAG}

# ---------------------------------------------------------------------------
# §4.2 — GSM8K
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.2 GSM8K ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_gsm8k_sft.py" \
    --model-path  "${SFT_MODEL}" \
    --data-path   "${ROOT}/data/gsm8k/test.jsonl" \
    --output-path "${RESULTS_DIR}/eval_gsm8k_sft.jsonl" \
    ${SMOKE_FLAG}

if [ "${SKIP_JUDGES}" -eq 1 ]; then
    echo ""
    echo "==> Skipping AlpacaEval and SST annotation (--skip-judges)."
    echo "    Re-run without --skip-judges on a 2× 80 GB instance for §4.3 and §4.4."
    exit 0
fi

# ---------------------------------------------------------------------------
# 70B judge — resolved lazily (only when judges actually run)
# ---------------------------------------------------------------------------
echo ""
echo "==> Resolving Llama 3.3 70B Instruct judge ..."
MODEL_ANNOTATOR=$(resolve_model \
    "/data/a5-alignment/models/Llama-3.3-70B-Instruct" \
    "${ROOT}/assets/Llama-3.3-70B-Instruct" \
    "meta-llama/Llama-3.3-70B-Instruct")
echo "    ${MODEL_ANNOTATOR}"

# ---------------------------------------------------------------------------
# §4.3 — AlpacaEval generation + annotation
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.3 AlpacaEval — generating outputs ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_alpaca_sft.py" \
    --model-path  "${SFT_MODEL}" \
    --data-path   "${ROOT}/data/alpaca_eval/alpaca_eval.jsonl" \
    --output-path "${RESULTS_DIR}/alpaca_eval_sft.json" \
    --generator   "llama-3.1-8b-sft" \
    ${SMOKE_FLAG}

echo ""
echo "==> §4.3 AlpacaEval — running judge ..."
CONFIGS_YAML="${ROOT}/scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml"
sed -i "s|model_name: \"ANNOTATOR_MODEL_PATH\".*|model_name: \"${MODEL_ANNOTATOR}\"|" "${CONFIGS_YAML}"

# alpaca_eval requires a .json array (not .jsonl) as reference
ALPACA_REF_JSON="${ROOT}/data/alpaca_eval/alpaca_eval_ref.json"
uv run python -c "
import json, pathlib
data = [json.loads(l) for l in pathlib.Path('${ROOT}/data/alpaca_eval/alpaca_eval.jsonl').read_text().splitlines() if l.strip()]
pathlib.Path('${ALPACA_REF_JSON}').write_text(json.dumps(data, indent=2))
print(f'Wrote {len(data)} reference records to ${ALPACA_REF_JSON}')
"

pushd "${ROOT}" > /dev/null
uv run alpaca_eval \
    --model_outputs        "${RESULTS_DIR}/alpaca_eval_sft.json" \
    --reference_outputs    "${ALPACA_REF_JSON}" \
    --annotators_config    "scripts/alpaca_eval_vllm_llama3_3_70b_fn" \
    --base-dir             "." \
    --output_path          "${RESULTS_DIR}"
popd > /dev/null

# Restore placeholder
sed -i "s|model_name: \"${MODEL_ANNOTATOR}\"|model_name: \"ANNOTATOR_MODEL_PATH\"|" "${CONFIGS_YAML}"

# ---------------------------------------------------------------------------
# §4.4 — SimpleSafetyTests generation + annotation
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.4 SimpleSafetyTests — generating outputs ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_sst_sft.py" \
    --model-path  "${SFT_MODEL}" \
    --data-path   "${ROOT}/data/simple_safety_tests/simple_safety_tests.csv" \
    --output-path "${RESULTS_DIR}/sst_sft.jsonl" \
    ${SMOKE_FLAG}

echo ""
echo "==> §4.4 SimpleSafetyTests — running safety annotator ..."
uv run python "${ROOT}/scripts/evaluate_safety.py" \
    --input-path         "${RESULTS_DIR}/sst_sft.jsonl" \
    --model-name-or-path "${MODEL_ANNOTATOR}" \
    --num-gpus           2 \
    --output-path        "${RESULTS_DIR}/sst_sft_annotated.jsonl"

echo ""
echo "==> §4 evaluation complete. Results in ${RESULTS_DIR}"
echo "    Generate comparison plots:"
echo "    uv run python cs336_alignment/section2_zero_shot/plot_zero_shot_results.py"