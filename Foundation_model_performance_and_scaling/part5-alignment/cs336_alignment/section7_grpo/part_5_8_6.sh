#!/usr/bin/env bash
# USAGE:   bash part_5_8_6.sh [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.6 — Prompt Ablation.
#   Compares the r1_zero prompt (default) against the question_only prompt using
#   the best hyperparameters from §8.5 (grpo_clip, lr=1e-5, epochs=4, bs=128).
#
#   The r1_zero run is reused from §8.5 if already present.
#   Only the question_only run needs to be run fresh.
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_grpo_clip_lr1e-5_e4_bs128_question_only.jsonl
#   results/section8/prompt_ablation/prompt_grpo_accuracy.png
#   results/section8/prompt_ablation/prompt_grpo_entropy.png
#   results/section8/prompt_ablation/prompt_grpo_response_length.png
#   results/section8/prompt_ablation/prompt_grpo_grad_norm.png
#
# NOTES:
#   Requires 2× A100. ~3 hrs for the one new run.

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

SMOKE_TEST=0
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --smoke-test) SMOKE_TEST=1 ;;
        --dry-run)    DRY_RUN=1 ;;
    esac
done

SCRIPT="${ROOT}/cs336_alignment/section7_grpo/part_5_7.sh"
PLOT="${ROOT}/cs336_alignment/section7_grpo/plot_grpo_results.py"

if [ "${SMOKE_TEST}" -eq 1 ]; then
    RESULTS_DIR="${ROOT}/results/section7/smoke"
else
    RESULTS_DIR="${ROOT}/results/section8"
fi

SMOKE_FLAG=""
[ "${SMOKE_TEST}" -eq 1 ] && SMOKE_FLAG="--smoke-test"

BEST_LR="1e-5"
BEST_EPOCHS=4
BEST_BS=128

# Run names
R1_ZERO_RUN="grpo_grpo_clip_lr${BEST_LR}_e${BEST_EPOCHS}_bs${BEST_BS}"
QUESTION_ONLY_RUN="grpo_grpo_clip_lr${BEST_LR}_e${BEST_EPOCHS}_bs${BEST_BS}_question_only"

echo "========================================"
echo "  Section 8.6 — Prompt Ablation"
echo "  Best config: epochs=${BEST_EPOCHS}, bs=${BEST_BS}, lr=${BEST_LR}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps)"
echo "========================================"
echo ""

# r1_zero run — reused from §8.5 if present
R1_ZERO_FILE="${RESULTS_DIR}/eval_metrics_${R1_ZERO_RUN}.jsonl"
if [ -f "${R1_ZERO_FILE}" ] && [ "${SMOKE_TEST}" -eq 0 ]; then
    echo "  r1_zero baseline (${R1_ZERO_RUN}): already present (from §8.5), skipping."
else
    echo "------------------------------------------------------------"
    echo "  Training: r1_zero prompt  (epochs=${BEST_EPOCHS}, bs=${BEST_BS})"
    echo "------------------------------------------------------------"
    CMD="bash ${SCRIPT} --loss-type=grpo_clip --lr=${BEST_LR} --epochs=${BEST_EPOCHS} --train-batch-size=${BEST_BS} ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
fi
echo ""

# question_only run
echo "------------------------------------------------------------"
echo "  Training: question_only prompt  (epochs=${BEST_EPOCHS}, bs=${BEST_BS})"
echo "------------------------------------------------------------"
CMD="bash ${SCRIPT} --loss-type=grpo_clip --lr=${BEST_LR} --epochs=${BEST_EPOCHS} --train-batch-size=${BEST_BS} --prompt-type=question_only ${SMOKE_FLAG}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
echo ""

PROMPT_RUNS="${R1_ZERO_RUN},${QUESTION_ONLY_RUN}"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/prompt_ablation"

echo "========================================"
echo "  Plotting §8.6 prompt ablation"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --output_prefix prompt_ --runs ${PROMPT_RUNS}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${PLOT_CMD}"; else eval "${PLOT_CMD}"; fi

echo ""
echo "Done. Results in ${PLOT_OUTPUT_DIR}"