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
#   SFT checkpoint:  cluster → assets/sft_ultrachat → HuggingFace (sclion/llama-3.1-8b-sft-ultrachat)
#   70B judge:       cluster → assets/Llama-3.3-70B-Instruct → HuggingFace (meta-llama/Llama-3.3-70B-Instruct)
#
# DATA: reused from §2 — downloaded automatically if not present.
#
# OUTPUT:
#   results/section4/eval_mmlu_sft.jsonl / .summary.json
#   results/section4/eval_gsm8k_sft.jsonl / .summary.json
#   results/section4/alpaca_eval_sft.json + leaderboard.csv
#   results/section4/sst_sft.jsonl + sst_sft_annotated.jsonl
#
# FLAGS:
#   --smoke-test   tiny subset for quick validation
#   --skip-judges  skip AlpacaEval + SST annotation (1 GPU only)
#   --plot         generate plots after evaluation completes
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
PLOT=0
for arg in "$@"; do
    case "$arg" in
        --smoke-test)   SMOKE_TEST=1 ;;
        --skip-judges)  SKIP_JUDGES=1 ;;
        --plot)         PLOT=1 ;;
    esac
done
SMOKE_FLAG=""
[ "${SMOKE_TEST}" -eq 1 ] && SMOKE_FLAG="--smoke-test"

# ---------------------------------------------------------------------------
# resolve_model: cluster → local → HuggingFace download (retries up to 3×)
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
        mkdir -p "${local_path}"
        local attempt=0
        while true; do
            attempt=$(( attempt + 1 ))
            if uv run huggingface-cli download "${hf_repo}" \
                    --local-dir "${local_path}" --max-workers 2 >&2; then
                break
            fi
            [ "${attempt}" -ge 3 ] && { echo "ERROR: Download failed after 3 attempts." >&2; exit 1; }
            echo "INFO: Attempt ${attempt} failed, retrying..." >&2
            sleep 5
        done
        echo "${local_path}"
    fi
}

# ---------------------------------------------------------------------------
# resolve_data: cluster → local → download
# ---------------------------------------------------------------------------
resolve_data() {
    local cluster="$1" local_path="$2" download_cmd="$3" glob="${4:-}"
    _path_ok() {
        [ -e "$1" ] || return 1
        [ -z "${glob}" ] && return 0
        ls $1/${glob} >/dev/null 2>&1
    }
    if _path_ok "${cluster}"; then echo "${cluster}"
    elif _path_ok "${local_path}"; then echo "${local_path}"
    else
        echo "INFO: Downloading data -> ${local_path}" >&2
        eval "${download_cmd}"
        echo "${local_path}"
    fi
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR="${ROOT}/results/section4"
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# SFT checkpoint
# ---------------------------------------------------------------------------
echo "==> Resolving SFT checkpoint ..."
SFT_MODEL=$(resolve_model \
    "/data/a5-alignment/sft_ultrachat" \
    "${ROOT}/assets/sft_ultrachat" \
    "sclion/llama-3.1-8b-sft-ultrachat")
echo "    ${SFT_MODEL}"

# ---------------------------------------------------------------------------
# Data — reuse §2 downloads; fetch if missing
# ---------------------------------------------------------------------------
echo "==> Resolving data ..."
MMLU_DIR=$(resolve_data \
    "/data/a5-alignment/mmlu/test" \
    "${ROOT}/data/mmlu/test" \
    "mkdir -p '${ROOT}/data/mmlu' && \
     wget --show-progress -O /tmp/mmlu_data.tar 'https://people.eecs.berkeley.edu/~hendrycks/data.tar' && \
     tar -xf /tmp/mmlu_data.tar -C '${ROOT}/data/mmlu' --strip-components=1 && \
     rm /tmp/mmlu_data.tar" \
    "*_test.csv")

GSM8K_FILE=$(resolve_data \
    "/data/a5-alignment/gsm8k/test.jsonl" \
    "${ROOT}/data/gsm8k/test.jsonl" \
    "mkdir -p '${ROOT}/data/gsm8k' && \
     wget -O '${ROOT}/data/gsm8k/test.jsonl' \
     'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'")

ALPACA_FILE=$(resolve_data \
    "/data/a5-alignment/alpaca_eval/alpaca_eval.jsonl" \
    "${ROOT}/data/alpaca_eval/alpaca_eval.jsonl" \
    "mkdir -p '${ROOT}/data/alpaca_eval' && \
     uv run python -c \
     \"import json; from datasets import load_dataset; \
     ds = load_dataset('tatsu-lab/alpaca_eval', 'alpaca_eval')['eval']; \
     open('${ROOT}/data/alpaca_eval/alpaca_eval.jsonl','w').writelines(json.dumps(dict(r))+'\\n' for r in ds)\"")

SST_FILE=$(resolve_data \
    "/data/a5-alignment/simple_safety_tests/simple_safety_tests.csv" \
    "${ROOT}/data/simple_safety_tests/simple_safety_tests.csv" \
    "mkdir -p '${ROOT}/data/simple_safety_tests' && \
     wget -O '${ROOT}/data/simple_safety_tests/simple_safety_tests.csv' \
     'https://raw.githubusercontent.com/bertiev/SimpleSafetyTests/main/SimpleSafetyTests.csv'")

echo "    MMLU:  ${MMLU_DIR}"
echo "    GSM8K: ${GSM8K_FILE}"
[ "${SKIP_JUDGES}" -eq 0 ] && echo "    AE:    ${ALPACA_FILE}"
[ "${SKIP_JUDGES}" -eq 0 ] && echo "    SST:   ${SST_FILE}"

# ---------------------------------------------------------------------------
# §4.1 — MMLU
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.1 MMLU ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_mmlu_sft.py" \
    --model-path  "${SFT_MODEL}" \
    --data-dir    "${MMLU_DIR}" \
    --output-path "${RESULTS_DIR}/eval_mmlu_sft.jsonl" \
    ${SMOKE_FLAG}

# ---------------------------------------------------------------------------
# §4.2 — GSM8K
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.2 GSM8K ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_gsm8k_sft.py" \
    --model-path  "${SFT_MODEL}" \
    --data-path   "${GSM8K_FILE}" \
    --output-path "${RESULTS_DIR}/eval_gsm8k_sft.jsonl" \
    ${SMOKE_FLAG}

if [ "${SKIP_JUDGES}" -eq 1 ]; then
    echo ""
    echo "==> Skipping AlpacaEval and SST annotation (--skip-judges)."
    echo "    Re-run without --skip-judges on a 2× 80 GB instance for §4.3 and §4.4."
    exit 0
fi

# ---------------------------------------------------------------------------
# 70B judge — resolved only when judges actually run
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
    --data-path   "${ALPACA_FILE}" \
    --output-path "${RESULTS_DIR}/alpaca_eval_sft.json" \
    --generator   "llama-3.1-8b-sft" \
    ${SMOKE_FLAG}

echo ""
echo "==> §4.3 AlpacaEval — running judge ..."

# Patch configs.yaml with resolved model path (file always contains placeholder)
CONFIGS_YAML="${ROOT}/scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml"
sed -i "s|model_name: \"ANNOTATOR_MODEL_PATH\"|model_name: \"${MODEL_ANNOTATOR}\"|" "${CONFIGS_YAML}"

# alpaca_eval requires .json array (not .jsonl) as reference
ALPACA_REF_JSON="${ROOT}/data/alpaca_eval/alpaca_eval_ref.json"
uv run python -c "
import json, pathlib
data = [json.loads(l) for l in pathlib.Path('${ALPACA_FILE}').read_text().splitlines() if l.strip()]
pathlib.Path('${ALPACA_REF_JSON}').write_text(json.dumps(data, indent=2))
print(f'Reference: {len(data)} examples -> ${ALPACA_REF_JSON}')
"

pushd "${ROOT}" > /dev/null
uv run alpaca_eval \
    --model_outputs        "${RESULTS_DIR}/alpaca_eval_sft.json" \
    --reference_outputs    "${ALPACA_REF_JSON}" \
    --annotators_config    "scripts/alpaca_eval_vllm_llama3_3_70b_fn" \
    --base-dir             "." \
    --output_path          "${RESULTS_DIR}"
popd > /dev/null

# Restore placeholder so configs.yaml stays clean in git
sed -i "s|model_name: \"${MODEL_ANNOTATOR}\"|model_name: \"ANNOTATOR_MODEL_PATH\"|" "${CONFIGS_YAML}"

# ---------------------------------------------------------------------------
# §4.4 — SimpleSafetyTests generation + annotation
# ---------------------------------------------------------------------------
echo ""
echo "==> §4.4 SimpleSafetyTests — generating outputs ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_sst_sft.py" \
    --model-path  "${SFT_MODEL}" \
    --data-path   "${SST_FILE}" \
    --output-path "${RESULTS_DIR}/sst_sft.jsonl" \
    ${SMOKE_FLAG}

echo ""
echo "==> §4.4 SimpleSafetyTests — running safety annotator ..."
uv run python "${ROOT}/scripts/evaluate_safety.py" \
    --input-path         "${RESULTS_DIR}/sst_sft.jsonl" \
    --model-name-or-path "${MODEL_ANNOTATOR}" \
    --num-gpus           2 \
    --output-path        "${RESULTS_DIR}/sst_sft_annotated.jsonl"

if [ "${PLOT}" -eq 1 ]; then
    echo ""
    echo "==> Generating plots ..."
    uv run python "${ROOT}/cs336_alignment/section4_eval/plot_sft_eval.py" \
        --results-section4 "${RESULTS_DIR}" \
        --results-section2 "${ROOT}/results/section2"
fi

echo ""
echo "==> §4 complete. Results in ${RESULTS_DIR}"
echo "    To generate plots separately:"
echo "    uv run python cs336_alignment/section4_eval/plot_sft_eval.py"
