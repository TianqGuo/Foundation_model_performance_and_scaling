#!/usr/bin/env bash
# USAGE:   bash part_5_8_5_focused.sh [--best-epochs=N] [--best-bs=N] [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.5.2 — Off-Policy Hyperparameter Sweep (Phase 2: focused 200-step runs).
#   Runs the best off-policy config from the broad sweep to convergence (200 steps),
#   alongside the on-policy grpo_clip baseline (epochs=1) for direct comparison.
#   Generates reward curves vs both grpo_step and wall-clock time.
#
# OPTIONS:
#   --best-epochs=N    epochs_per_rollout_batch for best config (default: 4)
#   --best-bs=N        train_batch_size for best config (default: 256)
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_grpo_clip_lr1e-5.jsonl
#   results/section8/eval_metrics_grpo_grpo_clip_lr1e-5_e{N}[_bs{N}].jsonl
#   results/section8/off_policy/focused_grpo_accuracy.png
#   results/section8/off_policy/focused_grpo_accuracy_wall_clock_hours.png
#
# NOTES:
#   Requires 2× A100. ~1.5 hrs per run, ~3 hrs total.

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

SMOKE_TEST=0
DRY_RUN=0
BEST_EPOCHS=4
BEST_BS=256
for arg in "$@"; do
    case $arg in
        --smoke-test)      SMOKE_TEST=1 ;;
        --dry-run)         DRY_RUN=1 ;;
        --best-epochs=*)   BEST_EPOCHS="${arg#*=}" ;;
        --best-bs=*)       BEST_BS="${arg#*=}" ;;
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

# Derive best config run name
BEST_RUN="grpo_grpo_clip_lr${BEST_LR}"
[ "${BEST_EPOCHS}" -gt 1 ]    && BEST_RUN="${BEST_RUN}_e${BEST_EPOCHS}"
[ "${BEST_BS}" != "256" ]     && BEST_RUN="${BEST_RUN}_bs${BEST_BS}"

echo "========================================"
echo "  Section 8.5.2 — Off-Policy Focused Runs"
echo "  Best config: epochs=${BEST_EPOCHS}, bs=${BEST_BS}"
echo "  Best LR: ${BEST_LR}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps)"
echo "========================================"
echo ""

# On-policy grpo_clip baseline (epochs=1, bs=256)
BASELINE_FILE="${RESULTS_DIR}/eval_metrics_grpo_grpo_clip_lr${BEST_LR}.jsonl"
if [ -f "${BASELINE_FILE}" ] && [ "${SMOKE_TEST}" -eq 0 ]; then
    echo "  on-policy clip baseline: already present, skipping."
else
    echo "------------------------------------------------------------"
    echo "  Training: on-policy grpo_clip baseline (epochs=1, bs=256)"
    echo "------------------------------------------------------------"
    CMD="bash ${SCRIPT} --loss-type=grpo_clip --lr=${BEST_LR} ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
fi
echo ""

# Best off-policy config
BEST_FILE="${RESULTS_DIR}/eval_metrics_${BEST_RUN}.jsonl"
if [ -f "${BEST_FILE}" ] && [ "${SMOKE_TEST}" -eq 0 ]; then
    echo "  best off-policy config (${BEST_RUN}): already present, skipping."
else
    echo "------------------------------------------------------------"
    echo "  Training: best off-policy config (epochs=${BEST_EPOCHS}, bs=${BEST_BS})"
    echo "------------------------------------------------------------"
    EPOCH_ARG=""
    BS_ARG=""
    [ "${BEST_EPOCHS}" -gt 1 ] && EPOCH_ARG="--epochs=${BEST_EPOCHS}"
    [ "${BEST_BS}" != "256" ]  && BS_ARG="--train-batch-size=${BEST_BS}"
    CMD="bash ${SCRIPT} --loss-type=grpo_clip --lr=${BEST_LR} ${EPOCH_ARG} ${BS_ARG} ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
fi
echo ""

FOCUSED_RUNS="grpo_grpo_clip_lr${BEST_LR},${BEST_RUN}"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/off_policy"

for X_AXIS in grpo_step wall_clock_hours; do
    echo "========================================"
    echo "  Plotting §8.5.2 focused runs (x=${X_AXIS})"
    echo "========================================"
    PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --output_prefix focused_ --runs ${FOCUSED_RUNS} --x_axis ${X_AXIS}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${PLOT_CMD}"; else eval "${PLOT_CMD}"; fi
    echo ""
done

echo "Done. Results in ${PLOT_OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  Run: bash part_5_8_5_clip.sh --best-epochs=${BEST_EPOCHS} --best-bs=${BEST_BS}"
