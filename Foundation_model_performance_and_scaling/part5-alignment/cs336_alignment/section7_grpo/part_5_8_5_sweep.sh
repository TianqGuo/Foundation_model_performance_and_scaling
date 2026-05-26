#!/usr/bin/env bash
# USAGE:   bash part_5_8_5_sweep.sh [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.5.2 — Off-Policy Hyperparameter Sweep (Phase 1: broad sweep).
#   Runs 3 off-policy configurations for 50 GRPO steps each to quickly identify
#   which (epochs, train_batch_size) setting performs best before committing to
#   full 200-step runs. All runs use grpo_clip at the best LR (1e-5).
#
#   Configurations swept:
#     1. epochs=1,  bs=256  — on-policy grpo_clip baseline
#     2. epochs=4,  bs=256  — standard off-policy (4 gradient steps per rollout)
#     3. epochs=4,  bs=128  — off-policy with smaller batches (8 steps per rollout)
#
#   Run names include _s50 suffix to avoid overwriting full 200-step runs.
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_grpo_clip_lr1e-5_{config}_s50.jsonl
#   results/section8/off_policy/sweep_grpo_accuracy.png    — overlaid comparison
#   results/section8/off_policy/sweep_grpo_format_rate.png
#   results/section8/off_policy/sweep_grpo_grad_norm.png
#
# NOTES:
#   Requires 2× A100. ~45 min per config, ~2.5 hrs total.
#   After reviewing plots, run part_5_8_5_focused.sh with the best config.
#   --smoke-test runs 3 steps per config locally.
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
SWEEP_STEPS=50

echo "========================================"
echo "  Section 8.5.2 — Off-Policy Sweep (Phase 1)"
echo "  Best LR: ${BEST_LR}  |  Steps per run: ${SWEEP_STEPS}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps)"
echo "========================================"
echo ""

run_config() {
    local label="$1"; shift
    echo "------------------------------------------------------------"
    echo "  Training: ${label}"
    echo "------------------------------------------------------------"
    local CMD="bash ${SCRIPT} --loss-type=grpo_clip --lr=${BEST_LR} --steps=${SWEEP_STEPS} $@ ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
    echo ""
}

run_config "on-policy clip  (epochs=1, bs=256)"
run_config "off-policy e4   (epochs=4, bs=256)" --epochs=4
run_config "off-policy e4b  (epochs=4, bs=128)" --epochs=4 --train-batch-size=128

# --- Plots ---
SWEEP_SUFFIX="_s${SWEEP_STEPS}"
[ "${SMOKE_TEST}" -eq 1 ] && SWEEP_SUFFIX="_s${SWEEP_STEPS}_smoke"

SWEEP_RUNS="grpo_grpo_clip_lr${BEST_LR}${SWEEP_SUFFIX},grpo_grpo_clip_lr${BEST_LR}_e4${SWEEP_SUFFIX},grpo_grpo_clip_lr${BEST_LR}_e4_bs128${SWEEP_SUFFIX}"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/off_policy"

echo "========================================"
echo "  Plotting §8.5 sweep comparison (vs grpo_step)"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --output_prefix sweep_ --runs ${SWEEP_RUNS}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${PLOT_CMD}"; else eval "${PLOT_CMD}"; fi

echo ""
echo "========================================"
echo "  Plotting §8.5 sweep comparison (vs wall-clock time)"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --output_prefix sweep_ --runs ${SWEEP_RUNS} --x_axis wall_clock_hours"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${PLOT_CMD}"; else eval "${PLOT_CMD}"; fi

echo ""
echo "Done. Sweep results in ${PLOT_OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review sweep_grpo_accuracy.png and sweep_grpo_accuracy_wall_clock_hours.png"
echo "  2. Pick the best (epochs, batch_size) config."
echo "  3. Run: bash part_5_8_5_focused.sh --best-epochs=<N> --best-bs=<N>"
