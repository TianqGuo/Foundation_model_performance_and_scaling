#!/usr/bin/env bash
# USAGE:   bash part_5_8_4.sh [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.4 — Effect of Group Standard Deviation Normalization.
#   Compares standard GRPO (use_std_normalization=True, default) against
#   Dr. GRPO (use_std_normalization=False, --no-std) at the best LR (1e-5)
#   with reinforce_with_baseline and masked_mean (on-policy).
#
#   The with-std baseline (grpo_reinforce_with_baseline_lr1e-5) is reused
#   from §8.1 if already present. Only the no-std run needs to be fresh.
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_reinforce_with_baseline_lr1e-5_nostd.jsonl
#   results/section8/std_norm/grpo_accuracy.png   — overlaid comparison
#   results/section8/std_norm/grpo_reward.png
#   results/section8/std_norm/grpo_grad_norm.png
#
# NOTES:
#   Requires 2× A100. ~1.5 H100 hrs for the one new run.
#   --smoke-test runs 3 GRPO steps locally.
#   --dry-run prints commands without executing them.

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

echo "========================================"
echo "  Section 8.4 — Std Normalization"
echo "  Best LR: ${BEST_LR}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps)"
echo "========================================"
echo ""

# with-std baseline is reused from §8.1 — skip if already present
BASELINE_FILE="${RESULTS_DIR}/eval_metrics_grpo_reinforce_with_baseline_lr${BEST_LR}.jsonl"
if [ -f "${BASELINE_FILE}" ] && [ "${SMOKE_TEST}" -eq 0 ]; then
    echo "  with_std (reinforce_with_baseline_lr${BEST_LR}): already present (from §8.1), skipping."
else
    echo "------------------------------------------------------------"
    echo "  Training: with_std (standard GRPO)  lr=${BEST_LR}"
    echo "------------------------------------------------------------"
    CMD="bash ${SCRIPT} --loss-type=reinforce_with_baseline --lr=${BEST_LR} ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
fi
echo ""

echo "------------------------------------------------------------"
echo "  Training: no_std (Dr. GRPO)  lr=${BEST_LR}"
echo "------------------------------------------------------------"
CMD="bash ${SCRIPT} --loss-type=reinforce_with_baseline --lr=${BEST_LR} --no-std ${SMOKE_FLAG}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
echo ""

STD_NORM_RUNS="grpo_reinforce_with_baseline_lr${BEST_LR},grpo_reinforce_with_baseline_lr${BEST_LR}_nostd"
[ "${SMOKE_TEST}" -eq 1 ] && STD_NORM_RUNS="grpo_reinforce_with_baseline_lr${BEST_LR}_smoke,grpo_reinforce_with_baseline_lr${BEST_LR}_nostd_smoke"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/std_norm"

echo "========================================"
echo "  Plotting §8.4 comparison"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --runs ${STD_NORM_RUNS}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] ${PLOT_CMD}"
else
    eval "${PLOT_CMD}"
fi

echo ""
echo "Done. Results in ${PLOT_OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Compare grpo_accuracy.png and grpo_grad_norm.png for with_std vs no_std."
echo "  2. Use the better-performing normalization for §8.5 onwards."