#!/usr/bin/env bash
# USAGE:   bash part_5_8_5_clip.sh [--best-epochs=N] [--best-bs=N] [--smoke-test] [--dry-run]
#
# WHAT IT DOES:
#   Section 8.5.3 — Clip Ablation in Off-Policy Setting.
#   Compares grpo_clip vs grpo_no_clip using the best off-policy hyperparameters
#   from §8.5.2. Tests whether PPO-style clipping is necessary for stability.
#
#   grpo_clip run is reused from part_5_8_5_focused.sh if already present.
#   Only grpo_no_clip needs to be run fresh.
#
# OPTIONS:
#   --best-epochs=N    epochs_per_rollout_batch (default: 4)
#   --best-bs=N        train_batch_size (default: 256)
#
# OUTPUT:
#   results/section8/eval_metrics_grpo_grpo_no_clip_lr1e-5_e{N}[_bs{N}].jsonl
#   results/section8/off_policy/clip_grpo_accuracy.png
#   results/section8/off_policy/clip_grpo_clip_frac.png   — key: shows clipping activity
#   results/section8/off_policy/clip_grpo_grad_norm.png
#
# NOTES:
#   Requires 2× A100. ~1.5 hrs for the one new run.

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

EPOCH_ARG=""
BS_ARG=""
[ "${BEST_EPOCHS}" -gt 1 ] && EPOCH_ARG="--epochs=${BEST_EPOCHS}"
[ "${BEST_BS}" != "256" ]  && BS_ARG="--train-batch-size=${BEST_BS}"

# Derive run names
CLIP_RUN="grpo_grpo_clip_lr${BEST_LR}"
NO_CLIP_RUN="grpo_grpo_no_clip_lr${BEST_LR}"
[ "${BEST_EPOCHS}" -gt 1 ]  && CLIP_RUN="${CLIP_RUN}_e${BEST_EPOCHS}"    && NO_CLIP_RUN="${NO_CLIP_RUN}_e${BEST_EPOCHS}"
[ "${BEST_BS}" != "256" ]   && CLIP_RUN="${CLIP_RUN}_bs${BEST_BS}"        && NO_CLIP_RUN="${NO_CLIP_RUN}_bs${BEST_BS}"

echo "========================================"
echo "  Section 8.5.3 — Clip Ablation"
echo "  Best config: epochs=${BEST_EPOCHS}, bs=${BEST_BS}, lr=${BEST_LR}"
echo "  Output: ${RESULTS_DIR}"
[ "${SMOKE_TEST}" -eq 1 ] && echo "  Mode: SMOKE TEST (3 steps)"
echo "========================================"
echo ""

# grpo_clip run reused from focused phase
CLIP_FILE="${RESULTS_DIR}/eval_metrics_${CLIP_RUN}.jsonl"
if [ -f "${CLIP_FILE}" ] && [ "${SMOKE_TEST}" -eq 0 ]; then
    echo "  grpo_clip (${CLIP_RUN}): already present (from focused phase), skipping."
else
    echo "------------------------------------------------------------"
    echo "  Training: grpo_clip  epochs=${BEST_EPOCHS} bs=${BEST_BS}"
    echo "------------------------------------------------------------"
    CMD="bash ${SCRIPT} --loss-type=grpo_clip --lr=${BEST_LR} ${EPOCH_ARG} ${BS_ARG} ${SMOKE_FLAG}"
    if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
fi
echo ""

echo "------------------------------------------------------------"
echo "  Training: grpo_no_clip  epochs=${BEST_EPOCHS} bs=${BEST_BS}"
echo "------------------------------------------------------------"
CMD="bash ${SCRIPT} --loss-type=grpo_no_clip --lr=${BEST_LR} ${EPOCH_ARG} ${BS_ARG} ${SMOKE_FLAG}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${CMD}"; else eval "${CMD}"; fi
echo ""

CLIP_RUNS="${CLIP_RUN},${NO_CLIP_RUN}"
PLOT_OUTPUT_DIR="${RESULTS_DIR}/off_policy"

echo "========================================"
echo "  Plotting §8.5.3 clip ablation"
echo "========================================"
PLOT_CMD="uv run python ${PLOT} --results_dir ${RESULTS_DIR} --output_dir ${PLOT_OUTPUT_DIR} --output_prefix clip_ --runs ${CLIP_RUNS}"
if [ "${DRY_RUN}" -eq 1 ]; then echo "[DRY RUN] ${PLOT_CMD}"; else eval "${PLOT_CMD}"; fi

echo ""
echo "Done. Results in ${PLOT_OUTPUT_DIR}"
echo ""
echo "Next step: bash part_5_8_6.sh"
