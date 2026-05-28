#!/usr/bin/env bash
# USAGE:
#   bash part_6_2.sh [--smoke-test] [--dry-run] [--skip-alpaca] [--skip-safety] [--plot]
#
# WHAT IT DOES:
#   Section 2 — Zero-Shot Baselines (full end-to-end).
#   Runs all four benchmarks in sequence:
#
#   §2.1  MMLU evaluation          (1 GPU, ~15 min full / ~2 min smoke)
#   §2.2  GSM8K evaluation         (1 GPU, ~15 min full / ~2 min smoke)
#   §2.3  AlpacaEval collection    (1 GPU, ~20 min full / ~2 min smoke)
#         AlpacaEval annotation    (2 GPUs, ~30 min)
#   §2.4  SimpleSafetyTests        (1 GPU, ~5 min full / ~1 min smoke)
#         Safety annotation        (2 GPUs, ~15 min)
#   Plot  Generate result charts   (no GPU)
#
# MODEL & DATA RESOLUTION (each resolved in priority order):
#   Models: cluster /data/a5-alignment/models/ → assets/ → HuggingFace download
#   Data:   cluster /data/a5-alignment/        → data/   → sibling part5 data/ → download
#
# FLAGS:
#   --smoke-test     Run on tiny subset (3 MMLU subjects, 20 GSM8K, 10 AlpacaEval, 10 SST).
#   --dry-run        Print commands without executing them.
#   --skip-alpaca    Skip AlpacaEval collection + annotation entirely.
#   --skip-safety    Skip SimpleSafetyTests collection + annotation entirely.
#   --plot           Generate result charts at the end (written to results/section2/).
#
# OUTPUTS:
#   results/section2/eval_mmlu_baseline.jsonl / .summary.json
#   results/section2/eval_gsm8k_baseline.jsonl / .summary.json
#   results/section2/alpaca_eval_baseline.json
#   results/section2/sst_baseline.jsonl
#   results/section2/sst_baseline_annotated.jsonl
#   scripts/alpaca_eval_vllm_llama3_3_70b_fn/  (AlpacaEval annotation outputs)
#   results/section2/*.png                     (if --plot)

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
SIBLING_DATA="$(cd "${ROOT}/../part5-alignment/data" 2>/dev/null && pwd)" || SIBLING_DATA=""

SMOKE_FLAG=""
DRY_RUN=0
SKIP_ALPACA=0
SKIP_SAFETY=0
PLOT=0

for arg in "$@"; do
    case $arg in
        --smoke-test)   SMOKE_FLAG="--smoke-test" ;;
        --dry-run)      DRY_RUN=1 ;;
        --skip-alpaca)  SKIP_ALPACA=1 ;;
        --skip-safety)  SKIP_SAFETY=1 ;;
        --plot)         PLOT=1 ;;
    esac
done

RESULTS_DIR="${ROOT}/results/section2"

# ── resolve() helper — returns first existing path, else downloads ────────────
# Usage: resolve_model <cluster_path> <local_path> <hf_repo>
# A valid model directory must contain config.json — a bare directory from a
# failed/partial download is treated as missing and triggers a fresh download.
# Exits with a clear message if the download fails (e.g. gated repo awaiting approval).
resolve_model() {
    local cluster="$1" local_path="$2" hf_repo="$3"
    if [ -f "${cluster}/config.json" ]; then
        echo "${cluster}"
    elif [ -f "${local_path}/config.json" ]; then
        echo "${local_path}"
    else
        echo "INFO: Model not found locally. Downloading ${hf_repo} -> ${local_path}" >&2
        mkdir -p "${ROOT}/assets"
        if ! uv run huggingface-cli download "${hf_repo}" --local-dir "${local_path}" >&2; then
            echo "" >&2
            echo "ERROR: Failed to download ${hf_repo}." >&2
            echo "  If this is a gated model, you must:" >&2
            echo "    1. Accept the licence at https://huggingface.co/${hf_repo}" >&2
            echo "    2. Wait for Meta's approval email (usually a few minutes)" >&2
            echo "    3. Re-run this script" >&2
            exit 1
        fi
        echo "${local_path}"
    fi
}

# Usage: resolve_data <cluster_path> <local_path> <sibling_path> <download_cmd> [<glob_check>]
# <glob_check>: optional shell glob — if given, path must also match at least one file.
# Returns the resolved path (echoes it), or exits with error if all fail.
resolve_data() {
    local cluster="$1" local_path="$2" sibling="$3" download_cmd="$4" glob="${5:-}"

    # Helper: returns 0 if path exists AND (no glob OR glob matches at least one file)
    _path_ok() {
        local p="$1"
        [ -e "${p}" ] || return 1
        if [ -n "${glob}" ]; then
            ls ${p}/${glob} >/dev/null 2>&1 || return 1
        fi
        return 0
    }

    if _path_ok "${cluster}"; then
        echo "${cluster}"
    elif _path_ok "${local_path}"; then
        echo "${local_path}"
    elif [ -n "${sibling}" ] && _path_ok "${sibling}"; then
        echo "${sibling}"
    elif [ -n "${download_cmd}" ]; then
        echo "INFO: Data not found locally. Downloading..." >&2
        eval "${download_cmd}"
        echo "${local_path}"
    else
        echo "ERROR: Data not found at any of:" >&2
        echo "  cluster: ${cluster}" >&2
        echo "  local:   ${local_path}" >&2
        [ -n "${sibling}" ] && echo "  sibling: ${sibling}" >&2
        [ -n "${glob}" ] && echo "  (required glob: ${glob})" >&2
        exit 1
    fi
}

# ── Resolve model paths ───────────────────────────────────────────────────────
# The 8B base model is resolved upfront (needed for all evaluations).
# The 70B annotator is resolved lazily inside each annotation block — only when
# AlpacaEval / SimpleSafetyTests annotation actually runs, so --skip-alpaca
# --skip-safety skips the download entirely.
if [ "${DRY_RUN}" -eq 0 ]; then
    MODEL_PATH=$(resolve_model \
        "/data/a5-alignment/models/Llama-3.1-8B" \
        "${ROOT}/assets/Llama-3.1-8B" \
        "meta-llama/Llama-3.1-8B")
else
    MODEL_PATH="/data/a5-alignment/models/Llama-3.1-8B"
fi

# ── Resolve data paths ────────────────────────────────────────────────────────
# MMLU test CSVs — require at least one *_test.csv file (not just directory existence)
MMLU_DIR=$(resolve_data \
    "/data/a5-alignment/mmlu/test" \
    "${ROOT}/data/mmlu/test" \
    "${SIBLING_DATA}/mmlu/test" \
    "mkdir -p '${ROOT}/data/mmlu' && \
     wget -q -O /tmp/mmlu_data.tar 'https://people.eecs.berkeley.edu/~hendrycks/data.tar' && \
     tar -xf /tmp/mmlu_data.tar -C '${ROOT}/data/mmlu' --strip-components=1 && \
     rm /tmp/mmlu_data.tar" \
    "*_test.csv")

# GSM8K test JSONL
GSM8K_FILE=$(resolve_data \
    "/data/a5-alignment/gsm8k/test.jsonl" \
    "${ROOT}/data/gsm8k/test.jsonl" \
    "${SIBLING_DATA}/gsm8k/test.jsonl" \
    "mkdir -p '${ROOT}/data/gsm8k' && \
     wget -q -O '${ROOT}/data/gsm8k/test.jsonl' \
     'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'")

# AlpacaEval JSONL
ALPACA_FILE=$(resolve_data \
    "/data/a5-alignment/alpaca_eval/alpaca_eval.jsonl" \
    "${ROOT}/data/alpaca_eval/alpaca_eval.jsonl" \
    "${SIBLING_DATA}/alpaca_eval/alpaca_eval.jsonl" \
    "mkdir -p '${ROOT}/data/alpaca_eval' && \
     uv run python -c \
     \"import json; from datasets import load_dataset; \
     ds = load_dataset('tatsu-lab/alpaca_eval', 'alpaca_eval')['eval']; \
     open('${ROOT}/data/alpaca_eval/alpaca_eval.jsonl','w').writelines(json.dumps(dict(r))+'\\n' for r in ds)\"")

# SimpleSafetyTests CSV
SST_FILE=$(resolve_data \
    "/data/a5-alignment/simple_safety_tests/simple_safety_tests.csv" \
    "${ROOT}/data/simple_safety_tests/simple_safety_tests.csv" \
    "${SIBLING_DATA}/simple_safety_tests/simple_safety_tests.csv" \
    "mkdir -p '${ROOT}/data/simple_safety_tests' && \
     wget -q -O '${ROOT}/data/simple_safety_tests/simple_safety_tests.csv' \
     'https://raw.githubusercontent.com/bertiev/SimpleSafetyTests/main/SimpleSafetyTests.csv'")

run() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[DRY RUN] $*"
    else
        eval "$@"
    fi
}

echo "========================================"
echo "  Section 2 — Zero-Shot Baselines"
echo "  Model:      ${MODEL_PATH}"
echo "  Output:     ${RESULTS_DIR}"
echo "  MMLU:       ${MMLU_DIR}"
echo "  GSM8K:      ${GSM8K_FILE}"
[ "${SKIP_ALPACA}" -eq 0 ] && echo "  AlpacaEval: ${ALPACA_FILE}"
[ "${SKIP_SAFETY}" -eq 0 ] && echo "  SST:        ${SST_FILE}"
[ -n "${SMOKE_FLAG}" ] && echo "  Mode:       SMOKE TEST"
[ "${PLOT}" -eq 1 ]    && echo "  Plotting:   ON"
echo "========================================"
echo ""

# ── §2.1  MMLU ────────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  §2.1 — MMLU zero-shot baseline"
echo "------------------------------------------------------------"
run uv run python "${ROOT}/cs336_alignment/section2_zero_shot/evaluate_mmlu.py" \
    --model-path "${MODEL_PATH}" \
    --data-dir "${MMLU_DIR}" \
    --output-path "${RESULTS_DIR}/eval_mmlu_baseline.jsonl" \
    ${SMOKE_FLAG}
echo ""

# ── §2.2  GSM8K ───────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  §2.2 — GSM8K zero-shot baseline"
echo "------------------------------------------------------------"
run uv run python "${ROOT}/cs336_alignment/section2_zero_shot/evaluate_gsm8k.py" \
    --model-path "${MODEL_PATH}" \
    --data-path "${GSM8K_FILE}" \
    --output-path "${RESULTS_DIR}/eval_gsm8k_baseline.jsonl" \
    ${SMOKE_FLAG}
echo ""

# ── §2.3  AlpacaEval ──────────────────────────────────────────────────────────
if [ "${SKIP_ALPACA}" -eq 0 ]; then
    echo "------------------------------------------------------------"
    echo "  §2.3 — AlpacaEval: prediction collection"
    echo "------------------------------------------------------------"
    run uv run python "${ROOT}/cs336_alignment/section2_zero_shot/evaluate_alpaca_eval.py" \
        --model-path "${MODEL_PATH}" \
        --data-path "${ALPACA_FILE}" \
        --output-path "${RESULTS_DIR}/alpaca_eval_baseline.json" \
        --generator llama-3.1-8b-base \
        ${SMOKE_FLAG}
    echo ""

    echo "------------------------------------------------------------"
    echo "  §2.3 — AlpacaEval: winrate annotation (2 GPUs)"
    echo "------------------------------------------------------------"
    if [ "${DRY_RUN}" -eq 0 ]; then
        MODEL_ANNOTATOR=$(resolve_model \
            "/data/a5-alignment/models/Llama-3.3-70B-Instruct" \
            "${ROOT}/assets/Llama-3.3-70B-Instruct" \
            "meta-llama/Llama-3.3-70B-Instruct")
    else
        MODEL_ANNOTATOR="/data/a5-alignment/models/Llama-3.3-70B-Instruct"
    fi
    run cd "${ROOT}" \&\& \
        uv run alpaca_eval \
            --model_outputs "${RESULTS_DIR}/alpaca_eval_baseline.json" \
            --annotators_config "scripts/alpaca_eval_vllm_llama3_3_70b_fn" \
            --base-dir "." \&\& \
        cd - \> /dev/null
    echo ""
fi

# ── §2.4  SimpleSafetyTests ───────────────────────────────────────────────────
if [ "${SKIP_SAFETY}" -eq 0 ]; then
    echo "------------------------------------------------------------"
    echo "  §2.4 — SimpleSafetyTests: prediction collection"
    echo "------------------------------------------------------------"
    run uv run python "${ROOT}/cs336_alignment/section2_zero_shot/evaluate_sst.py" \
        --model-path "${MODEL_PATH}" \
        --data-path "${SST_FILE}" \
        --output-path "${RESULTS_DIR}/sst_baseline.jsonl" \
        ${SMOKE_FLAG}
    echo ""

    echo "------------------------------------------------------------"
    echo "  §2.4 — SimpleSafetyTests: safety annotation (2 GPUs)"
    echo "------------------------------------------------------------"
    if [ "${DRY_RUN}" -eq 0 ]; then
        MODEL_ANNOTATOR=$(resolve_model \
            "/data/a5-alignment/models/Llama-3.3-70B-Instruct" \
            "${ROOT}/assets/Llama-3.3-70B-Instruct" \
            "meta-llama/Llama-3.3-70B-Instruct")
    else
        MODEL_ANNOTATOR="/data/a5-alignment/models/Llama-3.3-70B-Instruct"
    fi
    run uv run python "${ROOT}/scripts/evaluate_safety.py" \
        --input-path "${RESULTS_DIR}/sst_baseline.jsonl" \
        --model-name-or-path "${MODEL_ANNOTATOR}" \
        --num-gpus 2 \
        --output-path "${RESULTS_DIR}/sst_baseline_annotated.jsonl"
    echo ""
fi

# ── Plotting ──────────────────────────────────────────────────────────────────
if [ "${PLOT}" -eq 1 ]; then
    echo "------------------------------------------------------------"
    echo "  Generating result charts"
    echo "------------------------------------------------------------"
    run uv run python "${ROOT}/cs336_alignment/section2_zero_shot/plot_zero_shot_results.py" \
        --results-section2 "${RESULTS_DIR}" \
        --output "${RESULTS_DIR}"
    echo ""
fi

echo "========================================"
echo "  Done. Results in ${RESULTS_DIR}"
echo "========================================"
