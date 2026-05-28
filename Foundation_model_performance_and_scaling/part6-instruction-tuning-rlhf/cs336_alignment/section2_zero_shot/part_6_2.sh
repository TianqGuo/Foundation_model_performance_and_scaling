#!/usr/bin/env bash
# USAGE:
#   bash part_6_2.sh [--smoke-test] [--dry-run] [--skip-alpaca] [--skip-safety] [--plot]
#
# WHAT IT DOES:
#   Section 2 — Zero-Shot Baselines (full end-to-end).
#   Runs all four benchmarks in sequence on a single machine:
#
#   §2.1  MMLU evaluation          (1 GPU, ~15 min full / ~2 min smoke)
#   §2.2  GSM8K evaluation         (1 GPU, ~15 min full / ~2 min smoke)
#   §2.3  AlpacaEval collection    (1 GPU, ~20 min full / ~2 min smoke)
#         AlpacaEval annotation    (2 GPUs, ~30 min)
#   §2.4  SimpleSafetyTests        (1 GPU, ~5 min full / ~1 min smoke)
#         Safety annotation        (2 GPUs, ~15 min)
#   Plot  Generate result charts   (no GPU)
#
# FLAGS:
#   --smoke-test     Run on tiny subset to verify the pipeline (3 MMLU subjects,
#                    20 GSM8K examples, 10 AlpacaEval, 10 SST). Fast, no GPU waste.
#   --dry-run        Print commands without executing them.
#   --skip-alpaca    Skip AlpacaEval collection + annotation entirely.
#   --skip-safety    Skip SimpleSafetyTests collection + annotation entirely.
#   --plot           Generate result charts at the end (written to results/section2/).
#
# OUTPUTS:
#   results/section2/eval_mmlu_baseline.jsonl
#   results/section2/eval_mmlu_baseline.summary.json
#   results/section2/eval_gsm8k_baseline.jsonl
#   results/section2/eval_gsm8k_baseline.summary.json
#   results/section2/alpaca_eval_baseline.json
#   results/section2/sst_baseline.jsonl
#   results/section2/sst_baseline_annotated.jsonl
#   scripts/alpaca_eval_vllm_llama3_3_70b_fn/  (AlpacaEval annotation outputs)
#   results/section2/*.png                     (if --plot)
#
# NOTES:
#   - Steps 1-4 each load the 8B model fresh; vLLM handles GPU memory.
#   - Annotation steps require 2 GPUs with >80 GB VRAM each.
#     If you only have 1 GPU, use --skip-alpaca --skip-safety and run
#     the annotation steps later when 2 GPUs are available.
#   - The script uses set -e: any failure stops execution immediately.

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

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

MODEL_PATH="/data/a5-alignment/models/Llama-3.1-8B"
MODEL_ANNOTATOR="/data/a5-alignment/models/Llama-3.3-70B-Instruct"
RESULTS_DIR="${ROOT}/results/section2"

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
[ -n "${SMOKE_FLAG}" ] && echo "  Mode:       SMOKE TEST"
[ "${SKIP_ALPACA}" -eq 1 ] && echo "  Skipping:   AlpacaEval"
[ "${SKIP_SAFETY}" -eq 1 ] && echo "  Skipping:   SimpleSafetyTests"
[ "${PLOT}" -eq 1 ]        && echo "  Plotting:   ON"
echo "========================================"
echo ""

# ── §2.1  MMLU ────────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  §2.1 — MMLU zero-shot baseline"
echo "------------------------------------------------------------"
run uv run python "${ROOT}/cs336_alignment/section2_zero_shot/evaluate_mmlu.py" \
    --model-path "${MODEL_PATH}" \
    --data-dir "${ROOT}/data/mmlu/test" \
    --output-path "${RESULTS_DIR}/eval_mmlu_baseline.jsonl" \
    ${SMOKE_FLAG}
echo ""

# ── §2.2  GSM8K ───────────────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  §2.2 — GSM8K zero-shot baseline"
echo "------------------------------------------------------------"
run uv run python "${ROOT}/cs336_alignment/section2_zero_shot/evaluate_gsm8k.py" \
    --model-path "${MODEL_PATH}" \
    --data-path "${ROOT}/data/gsm8k/test.jsonl" \
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
        --data-path "${ROOT}/data/alpaca_eval/alpaca_eval.jsonl" \
        --output-path "${RESULTS_DIR}/alpaca_eval_baseline.json" \
        --generator llama-3.1-8b-base \
        ${SMOKE_FLAG}
    echo ""

    echo "------------------------------------------------------------"
    echo "  §2.3 — AlpacaEval: winrate annotation (2 GPUs)"
    echo "------------------------------------------------------------"
    # alpaca_eval writes its annotation outputs relative to --base-dir,
    # so we run it from ROOT where scripts/ lives.
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
        --data-path "${ROOT}/data/simple_safety_tests/simple_safety_tests.csv" \
        --output-path "${RESULTS_DIR}/sst_baseline.jsonl" \
        ${SMOKE_FLAG}
    echo ""

    echo "------------------------------------------------------------"
    echo "  §2.4 — SimpleSafetyTests: safety annotation (2 GPUs)"
    echo "------------------------------------------------------------"
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
