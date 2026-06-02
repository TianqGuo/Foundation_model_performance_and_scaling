#!/usr/bin/env bash
# =============================================================================
# USAGE:   bash cs336_alignment/section5_dpo/part_6_5_eval.sh [--skip-judges] [--smoke-test]
#
# WHAT IT DOES:
#   Evaluates the DPO-trained Llama 3.1 8B model on all four benchmarks.
#   Downloads all required datasets automatically — no dependency on §4 having
#   been run first. Can be run standalone after DPO training completes.
#
#   §5.4.1 MMLU          — rule-based scoring, single GPU, ~10 min
#   §5.4.2 GSM8K         — rule-based scoring, single GPU, ~2 min
#   §5.4.3 AlpacaEval    — generation + 70B judge, requires 2× 80 GB
#   §5.4.4 SST           — generation + 70B judge, requires 2× 80 GB
#
# MODEL RESOLUTION (in priority order):
#   DPO checkpoint:  cluster → assets/dpo_hh/best → HuggingFace (sclion/llama-3.1-8b-dpo-hh)
#   70B judge:       cluster → assets/Llama-3.3-70B-Instruct → HuggingFace
#
# OUTPUT:
#   results/section5/eval_mmlu_dpo.jsonl / .summary.json
#   results/section5/eval_gsm8k_dpo.jsonl / .summary.json
#   results/section5/alpaca_eval_dpo.json + leaderboard.csv
#   results/section5/sst_dpo.jsonl + sst_dpo_annotated.jsonl
#   results/section5/dpo_eval_summary.png (+ comparison plots)
#
# FLAGS:
#   --skip-judges  skip AlpacaEval + SST annotation (1 GPU only)
#   --smoke-test   tiny subset for quick validation
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
# resolve_model: cluster → local → HuggingFace download
# ---------------------------------------------------------------------------
resolve_model() {
    local cluster="$1" local_path="$2" hf_repo="$3"
    _model_complete() { [ -f "${1}/config.json" ] && [ -f "${1}/tokenizer_config.json" ]; }
    if _model_complete "${cluster}"; then echo "${cluster}"
    elif _model_complete "${local_path}"; then echo "${local_path}"
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
            [ "${attempt}" -ge 3 ] && { echo "ERROR: Download failed." >&2; exit 1; }
            sleep 5
        done
        echo "${local_path}"
    fi
}

# ---------------------------------------------------------------------------
# resolve_data: cluster → local → download command
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
# DPO checkpoint
# ---------------------------------------------------------------------------
echo "==> Resolving DPO checkpoint ..."
DPO_MODEL=$(resolve_model \
    "/data/a5-alignment/dpo_hh/best" \
    "${ROOT}/assets/dpo_hh/best" \
    "sclion/llama-3.1-8b-dpo-hh")
echo "    ${DPO_MODEL}"

# ---------------------------------------------------------------------------
# Data — downloaded automatically if not present
# ---------------------------------------------------------------------------
echo "==> Resolving eval data ..."

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
# §5.4.1 — MMLU
# ---------------------------------------------------------------------------
RESULTS_DIR="${ROOT}/results/section5"
mkdir -p "${RESULTS_DIR}"

echo ""
echo "==> §5.4.1 MMLU ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_mmlu_sft.py" \
    --model-path  "${DPO_MODEL}" \
    --data-dir    "${MMLU_DIR}" \
    --output-path "${RESULTS_DIR}/eval_mmlu_dpo.jsonl" \
    ${SMOKE_FLAG}

# ---------------------------------------------------------------------------
# §5.4.2 — GSM8K
# ---------------------------------------------------------------------------
echo ""
echo "==> §5.4.2 GSM8K ..."
uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_gsm8k_sft.py" \
    --model-path  "${DPO_MODEL}" \
    --data-path   "${GSM8K_FILE}" \
    --output-path "${RESULTS_DIR}/eval_gsm8k_dpo.jsonl" \
    ${SMOKE_FLAG}

if [ "${SKIP_JUDGES}" -eq 1 ]; then
    echo ""
    echo "==> Skipping AlpacaEval + SST annotation (--skip-judges)."
    echo "    Re-run without --skip-judges on a 2× 80 GB instance for §5.4.3 and §5.4.4."
else
    # -------------------------------------------------------------------------
    # 70B judge — resolved only when judges actually run
    # -------------------------------------------------------------------------
    echo ""
    echo "==> Resolving Llama 3.3 70B judge ..."
    MODEL_ANNOTATOR=$(resolve_model \
        "/data/a5-alignment/models/Llama-3.3-70B-Instruct" \
        "${ROOT}/assets/Llama-3.3-70B-Instruct" \
        "meta-llama/Llama-3.3-70B-Instruct")
    echo "    ${MODEL_ANNOTATOR}"

    # -------------------------------------------------------------------------
    # §5.4.3 — AlpacaEval generation + annotation
    # -------------------------------------------------------------------------
    echo ""
    echo "==> §5.4.3 AlpacaEval — generating outputs ..."
    uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_alpaca_sft.py" \
        --model-path  "${DPO_MODEL}" \
        --data-path   "${ALPACA_FILE}" \
        --output-path "${RESULTS_DIR}/alpaca_eval_dpo.json" \
        --generator   "llama-3.1-8b-dpo"

    echo ""
    echo "==> §5.4.3 AlpacaEval — running judge ..."
    CONFIGS_YAML="${ROOT}/scripts/alpaca_eval_vllm_llama3_3_70b_fn/configs.yaml"
    sed -i "s|model_name: \"ANNOTATOR_MODEL_PATH\"|model_name: \"${MODEL_ANNOTATOR}\"|" "${CONFIGS_YAML}"

    ALPACA_REF_JSON="${ROOT}/data/alpaca_eval/alpaca_eval_ref.json"
    uv run python -c "
import json, pathlib
data = [json.loads(l) for l in pathlib.Path('${ALPACA_FILE}').read_text().splitlines() if l.strip()]
pathlib.Path('${ALPACA_REF_JSON}').write_text(json.dumps(data, indent=2))
"
    pushd "${ROOT}" > /dev/null
    uv run alpaca_eval \
        --model_outputs        "${RESULTS_DIR}/alpaca_eval_dpo.json" \
        --reference_outputs    "${ALPACA_REF_JSON}" \
        --annotators_config    "scripts/alpaca_eval_vllm_llama3_3_70b_fn" \
        --base-dir             "." \
        --output_path          "${RESULTS_DIR}"
    popd > /dev/null
    sed -i "s|model_name: \"${MODEL_ANNOTATOR}\"|model_name: \"ANNOTATOR_MODEL_PATH\"|" "${CONFIGS_YAML}"

    # -------------------------------------------------------------------------
    # §5.4.4 — SimpleSafetyTests
    # -------------------------------------------------------------------------
    echo ""
    echo "==> §5.4.4 SimpleSafetyTests ..."
    uv run python "${ROOT}/cs336_alignment/section4_eval/evaluate_sst_sft.py" \
        --model-path  "${DPO_MODEL}" \
        --data-path   "${SST_FILE}" \
        --output-path "${RESULTS_DIR}/sst_dpo.jsonl"

    uv run python "${ROOT}/scripts/evaluate_safety.py" \
        --input-path         "${RESULTS_DIR}/sst_dpo.jsonl" \
        --model-name-or-path "${MODEL_ANNOTATOR}" \
        --num-gpus           2 \
        --output-path        "${RESULTS_DIR}/sst_dpo_annotated.jsonl"
fi

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
echo ""
echo "==> Generating plots ..."
uv run python "${ROOT}/cs336_alignment/section5_dpo/plot_dpo_training.py" \
    --results          "${RESULTS_DIR}" \
    --results-section2 "${ROOT}/results/section2" \
    --results-section4 "${ROOT}/results/section4"

echo ""
echo "==> §5.4 eval complete. Results in ${RESULTS_DIR}"
