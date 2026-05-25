#!/usr/bin/env bash
# USAGE:   bash part_5_8_2.sh [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.2 — Effect of Baselining.
#   Compares no_baseline vs reinforce_with_baseline at the best LR from §8.1 (1e-5),
#   both on-policy (epochs_per_rollout_batch=1).
#
#   reinforce_with_baseline_lr1e-5 is reused from §8.1 if already present.
#   Only no_baseline_lr1e-5 needs to be run fresh.
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_no_baseline_lr1e-5.jsonl
#   results/section8/grpo_accuracy.png    — overlaid comparison (all runs in dir)
#   results/section8/grpo_reward.png
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
    RESULTS_DIR="${ROOT}/results/section7"
else
    RESULTS_DIR="${ROOT}/results/section8"
fi

SMOKE_FLAG=""
[ "${SMOKE_TEST}" -eq 1 ] && SMOKE_FLAG="--smoke-test"

BEST_LR="1e-5"

echo "========================================"
echo "  Section 8.2 — Effect of Baselining"
echo "  Best LR: ${BEST_LR}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps)"
echo "========================================"
echo ""

# reinforce_with_baseline is already present from §8.1 — skip if file exists
BASELINE_FILE="${RESULTS_DIR}/eval_metrics_grpo_reinforce_with_baseline_lr${BEST_LR}.jsonl"
if [ -f "${BASELINE_FILE}" ] && [ "${SMOKE_TEST}" -eq 0 ]; then
    echo "  reinforce_with_baseline_lr${BEST_LR}: already present (from §8.1), skipping."
else
    echo "------------------------------------------------------------"
    echo "  Training: reinforce_with_baseline  lr=${BEST_LR}"
    echo "------------------------------------------------------------"
    CMD="bash ${SCRIPT} --loss-type=reinforce_with_baseline --lr=${BEST_LR} ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
fi
echo ""

echo "------------------------------------------------------------"
echo "  Training: no_baseline  lr=${BEST_LR}"
echo "------------------------------------------------------------"
CMD="bash ${SCRIPT} --loss-type=no_baseline --lr=${BEST_LR} ${SMOKE_FLAG}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
echo ""

BASELINE_RUNS="grpo_no_baseline_lr${BEST_LR},grpo_reinforce_with_baseline_lr${BEST_LR}"
[ "${SMOKE_TEST}" -eq 1 ] && BASELINE_RUNS="grpo_no_baseline_lr${BEST_LR}_smoke,grpo_reinforce_with_baseline_lr${BEST_LR}_smoke"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/baselines"

echo "========================================"
echo "  Plotting §8.2 comparison"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --runs ${BASELINE_RUNS}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] ${PLOT_CMD}"
else
    eval "${PLOT_CMD}"
fi

echo ""
echo "Done. Results in ${RESULTS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Compare grpo_accuracy.png and grpo_reward.png for no_baseline vs reinforce_with_baseline."
echo "  2. Use the better-performing loss type for §8.3 onwards."