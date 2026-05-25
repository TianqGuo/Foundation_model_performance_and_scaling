#!/usr/bin/env bash
# USAGE:   bash part_5_8_3.sh [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.3.2 — Empirical length normalization comparison.
#   Compares masked_mean vs masked_normalize at the best LR (1e-5) with
#   reinforce_with_baseline (on-policy).
#
#   The masked_mean baseline (grpo_reinforce_with_baseline_lr1e-5) is reused
#   from §8.1 if already present. Only masked_normalize needs a fresh run.
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_reinforce_with_baseline_lr1e-5_masked_normalize.jsonl
#   results/section8/length_norm/grpo_accuracy.png   — overlaid comparison
#   results/section8/length_norm/grpo_reward.png
#   results/section8/length_norm/grpo_grad_norm.png
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
echo "  Section 8.3.2 — Length Normalization"
echo "  Best LR: ${BEST_LR}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps)"
echo "========================================"
echo ""

# masked_mean baseline is reused from §8.1 — skip if already present
BASELINE_FILE="${RESULTS_DIR}/eval_metrics_grpo_reinforce_with_baseline_lr${BEST_LR}.jsonl"
if [ -f "${BASELINE_FILE}" ] && [ "${SMOKE_TEST}" -eq 0 ]; then
    echo "  masked_mean (reinforce_with_baseline_lr${BEST_LR}): already present (from §8.1), skipping."
else
    echo "------------------------------------------------------------"
    echo "  Training: masked_mean  lr=${BEST_LR}"
    echo "------------------------------------------------------------"
    CMD="bash ${SCRIPT} --loss-type=reinforce_with_baseline --lr=${BEST_LR} --length-norm=masked_mean ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
fi
echo ""

echo "------------------------------------------------------------"
echo "  Training: masked_normalize  lr=${BEST_LR}"
echo "------------------------------------------------------------"
CMD="bash ${SCRIPT} --loss-type=reinforce_with_baseline --lr=${BEST_LR} --length-norm=masked_normalize ${SMOKE_FLAG}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
echo ""

LENGTH_NORM_RUNS="grpo_reinforce_with_baseline_lr${BEST_LR},grpo_reinforce_with_baseline_lr${BEST_LR}_masked_normalize"
[ "${SMOKE_TEST}" -eq 1 ] && LENGTH_NORM_RUNS="grpo_reinforce_with_baseline_lr${BEST_LR}_smoke,grpo_reinforce_with_baseline_lr${BEST_LR}_masked_normalize_smoke"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/length_norm"

echo "========================================"
echo "  Plotting §8.3 comparison"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --runs ${LENGTH_NORM_RUNS}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY RUN] ${PLOT_CMD}"
else
    eval "${PLOT_CMD}"
fi

echo ""
echo "Done. Results in ${PLOT_OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Compare grpo_accuracy.png and grpo_grad_norm.png for masked_mean vs masked_normalize."
echo "  2. Use the better-performing length norm for §8.4 onwards."
