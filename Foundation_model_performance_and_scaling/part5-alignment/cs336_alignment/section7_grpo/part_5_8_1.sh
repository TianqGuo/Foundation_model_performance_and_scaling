#!/usr/bin/env bash
# USAGE:   bash part_5_8_1.sh [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.1 learning rate sweep.
#   Trains GRPO (reinforce_with_baseline, on-policy) with four learning rates
#   and overlays the resulting validation reward curves in one plot.
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_reinforce_with_baseline_lr<N>.jsonl  per run
#   results/section8/grpo_accuracy.png   — all LR curves overlaid
#   results/section8/grpo_reward.png
#   results/section8/grpo_entropy.png
#   results/section8/grpo_response_length.png
#
# NOTES:
#   Requires 2× H100s. Rough estimate: ~6 H100 hrs total (4 runs × ~1.5 hrs each).
#   --smoke-test runs 3 GRPO steps per LR for quick local verification.
#   --dry-run prints commands without executing them.
#   Runs are sequential; run in parallel manually if you have enough GPUs.

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
# Smoke tests write to section7; full runs write to section8 (mirrors part_5_7.sh logic)
if [ "${SMOKE_TEST}" -eq 1 ]; then
    RESULTS_DIR="${ROOT}/results/section7"
else
    RESULTS_DIR="${ROOT}/results/section8"
fi

SMOKE_FLAG=""
[ "${SMOKE_TEST}" -eq 1 ] && SMOKE_FLAG="--smoke-test"

# Learning rates to sweep (log-spaced from conservative to aggressive)
LRS=("3e-6" "1e-5" "3e-5" "1e-4")

echo "========================================"
echo "  Section 8.1 — Learning Rate Sweep"
echo "  LRs: ${LRS[*]}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps each)"
echo "========================================"
echo ""

for LR in "${LRS[@]}"; do
    echo "------------------------------------------------------------"
    echo "  Training with lr=${LR}"
    echo "------------------------------------------------------------"
    CMD="bash ${SCRIPT} --lr=${LR} ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[DRY RUN] ${CMD}"
    else
        eval "${CMD}"
    fi
    echo ""
done

LR_RUNS="grpo_reinforce_with_baseline_lr3e-6,grpo_reinforce_with_baseline_lr1e-5,grpo_reinforce_with_baseline_lr3e-5,grpo_reinforce_with_baseline_lr1e-4"
[ "${SMOKE_TEST}" -eq 1 ] && LR_RUNS="${LR_RUNS}_smoke,grpo_reinforce_with_baseline_smoke"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/lr_sweep"

echo "========================================"
echo "  Plotting §8.1 learning rate curves"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --runs ${LR_RUNS}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] ${PLOT_CMD}"
else
    eval "${PLOT_CMD}"
fi

echo ""
echo "Done. Reward curves saved to ${RESULTS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Check grpo_accuracy.png and grpo_reward.png to pick the best LR."
echo "  2. Look for the run that reached >=25% validation accuracy."
echo "  3. Use that LR for Section 8.2 onwards."
