#!/bin/bash
# ==============================================================================
# Memory Profiling - Part (c)
# ==============================================================================
#
# Profile peak memory usage with mixed precision (BF16) for the 2.7B model.
# Compares FP32 vs BF16 memory usage for:
# - Forward pass only
# - Full training step
#
# USAGE:
#   cd cs336_systems/memory_profiling
#   ./profile_part_c.sh
#
# OUTPUT:
#   results/memory_profiling/mixed_precision_memory_summary.txt
#   results/memory_profiling/*_bf16_snapshot.pickle (memory snapshots)
#
# NOTES:
#   - Requires GPU with BF16 support (e.g., A100, H100)
#   - Will use FP16 as fallback if BF16 not supported
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

# Configuration
MODEL_SIZE="2.7B"
CONTEXT_LENGTH=512
BATCH_SIZE=4
WARMUP_STEPS=5
MEASURE_STEPS=10

SUMMARY_FILE="../../results/memory_profiling/mixed_precision_memory_summary.txt"

echo "=============================================================================="
echo "Memory Profiling - Part (c): Mixed Precision Memory Usage"
echo "=============================================================================="
echo ""
echo "Comparing FP32 vs BF16 mixed precision for ${MODEL_SIZE} model"
echo "Context length: ${CONTEXT_LENGTH}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Check if GPU is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Check BF16 support
if python -c "import torch; assert torch.cuda.is_bf16_supported()" 2>/dev/null; then
    echo "✓ BF16 is supported on this GPU"
    DTYPE="bf16"
else
    echo "⚠ BF16 not supported, using FP16 instead"
    DTYPE="fp16"
fi
echo ""

# Create output directory
mkdir -p ../../results/memory_profiling

# Initialize summary file
cat > "${SUMMARY_FILE}" << EOF
================================================================================
Mixed Precision Memory Usage Summary - Part (c)
================================================================================

Model: ${MODEL_SIZE}
Context length: ${CONTEXT_LENGTH}
Batch size: ${BATCH_SIZE}
Mixed precision dtype: ${DTYPE^^}

================================================================================
EOF

echo "" >> "${SUMMARY_FILE}"
echo "Configuration        | Peak Memory (MB) | Memory Savings vs FP32" >> "${SUMMARY_FILE}"
echo "---------------------|------------------|------------------------" >> "${SUMMARY_FILE}"

# Profile FP32 forward pass
echo "=============================================================================="
echo "1. FP32 Forward Pass"
echo "=============================================================================="
echo ""

OUTPUT_FP32_FWD=$(python -m cs336_systems.memory_profiling.profile_memory \
    --model-size "${MODEL_SIZE}" \
    --context-length ${CONTEXT_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --warmup-steps ${WARMUP_STEPS} \
    --measure-steps ${MEASURE_STEPS} \
    --profile-type forward \
    --output "../../results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_fp32_snapshot.pickle" 2>&1)

FP32_FWD_PEAK=$(echo "$OUTPUT_FP32_FWD" | grep "Peak memory:" | awk '{print $3}')
echo "FP32 forward pass peak memory: ${FP32_FWD_PEAK} MB"
printf "%-20s | %-16s | %-22s\n" "FP32 Forward" "${FP32_FWD_PEAK}" "baseline" >> "${SUMMARY_FILE}"
echo ""

# Profile BF16 forward pass
echo "=============================================================================="
echo "2. ${DTYPE^^} Forward Pass (Mixed Precision)"
echo "=============================================================================="
echo ""

OUTPUT_MP_FWD=$(python -m cs336_systems.memory_profiling.profile_memory \
    --model-size "${MODEL_SIZE}" \
    --context-length ${CONTEXT_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --warmup-steps ${WARMUP_STEPS} \
    --measure-steps ${MEASURE_STEPS} \
    --profile-type forward \
    --use-mixed-precision \
    --dtype "${DTYPE}" \
    --output "../../results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_${DTYPE}_snapshot.pickle" 2>&1)

MP_FWD_PEAK=$(echo "$OUTPUT_MP_FWD" | grep "Peak memory:" | awk '{print $3}')
FWD_SAVINGS=$(python -c "print(f'{(1 - ${MP_FWD_PEAK}/${FP32_FWD_PEAK}) * 100:.1f}%')")
echo "${DTYPE^^} forward pass peak memory: ${MP_FWD_PEAK} MB"
echo "Memory savings: ${FWD_SAVINGS}"
printf "%-20s | %-16s | %-22s\n" "${DTYPE^^} Forward" "${MP_FWD_PEAK}" "${FWD_SAVINGS}" >> "${SUMMARY_FILE}"
echo ""

# Profile FP32 training step
echo "=============================================================================="
echo "3. FP32 Full Training Step"
echo "=============================================================================="
echo ""

OUTPUT_FP32_TRAIN=$(python -m cs336_systems.memory_profiling.profile_memory \
    --model-size "${MODEL_SIZE}" \
    --context-length ${CONTEXT_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --warmup-steps ${WARMUP_STEPS} \
    --measure-steps ${MEASURE_STEPS} \
    --profile-type training \
    --output "../../results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_training_fp32_snapshot.pickle" 2>&1)

FP32_TRAIN_PEAK=$(echo "$OUTPUT_FP32_TRAIN" | grep "Peak memory:" | awk '{print $3}')
echo "FP32 training step peak memory: ${FP32_TRAIN_PEAK} MB"
printf "%-20s | %-16s | %-22s\n" "FP32 Training" "${FP32_TRAIN_PEAK}" "baseline" >> "${SUMMARY_FILE}"
echo ""

# Profile BF16 training step
echo "=============================================================================="
echo "4. ${DTYPE^^} Full Training Step (Mixed Precision)"
echo "=============================================================================="
echo ""

OUTPUT_MP_TRAIN=$(python -m cs336_systems.memory_profiling.profile_memory \
    --model-size "${MODEL_SIZE}" \
    --context-length ${CONTEXT_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --warmup-steps ${WARMUP_STEPS} \
    --measure-steps ${MEASURE_STEPS} \
    --profile-type training \
    --use-mixed-precision \
    --dtype "${DTYPE}" \
    --output "../../results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_training_${DTYPE}_snapshot.pickle" 2>&1)

MP_TRAIN_PEAK=$(echo "$OUTPUT_MP_TRAIN" | grep "Peak memory:" | awk '{print $3}')
TRAIN_SAVINGS=$(python -c "print(f'{(1 - ${MP_TRAIN_PEAK}/${FP32_TRAIN_PEAK}) * 100:.1f}%')")
echo "${DTYPE^^} training step peak memory: ${MP_TRAIN_PEAK} MB"
echo "Memory savings: ${TRAIN_SAVINGS}"
printf "%-20s | %-16s | %-22s\n" "${DTYPE^^} Training" "${MP_TRAIN_PEAK}" "${TRAIN_SAVINGS}" >> "${SUMMARY_FILE}"
echo ""

# Add analysis notes
cat >> "${SUMMARY_FILE}" << 'EOF'

================================================================================
Analysis Notes:
================================================================================

Memory usage breakdown:
- Model parameters: Store in FP32 for stability (not affected by mixed precision)
- Activations: Stored in lower precision (BF16/FP16) - reduces memory
- Gradients: Computed in lower precision then cast back - reduces memory
- Optimizer states: Stored in FP32 (not affected by mixed precision)

Key observations:
- Mixed precision reduces activation/gradient memory
- Optimizer state memory (AdamW ~2x params) remains in FP32
- Memory savings are less pronounced in training vs inference
- Larger context lengths see more benefit from mixed precision

================================================================================
EOF

# Display summary
echo "=============================================================================="
echo "PART (c) COMPLETE"
echo "=============================================================================="
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""
echo "Memory snapshots saved to:"
echo "  FP32:"
echo "    - results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_fp32_snapshot.pickle"
echo "    - results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_training_fp32_snapshot.pickle"
echo "  ${DTYPE^^}:"
echo "    - results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_${DTYPE}_snapshot.pickle"
echo "    - results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_training_${DTYPE}_snapshot.pickle"
echo ""
echo "For writeup (2-3 sentences):"
echo "  - Does mixed precision significantly affect memory usage?"
echo "  - Compare forward-only vs full training step memory savings"
echo "  - Explain why optimizer states limit memory savings in training"
echo ""
