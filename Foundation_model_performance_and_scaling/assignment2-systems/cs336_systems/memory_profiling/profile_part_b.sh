#!/bin/bash
# ==============================================================================
# Memory Profiling - Part (b)
# ==============================================================================
#
# Profile peak memory usage for the 2.7B model across different context lengths:
# - Context lengths: 128, 256, 512
# - For both forward pass and full training step
#
# This generates a summary table of peak memory usage for each configuration.
#
# USAGE:
#   cd cs336_systems/memory_profiling
#   ./profile_part_b.sh
#
# OUTPUT:
#   results/memory_profiling/peak_memory_summary.txt
#   results/memory_profiling/*_ctx*_*_snapshot.pickle (memory snapshots)
#
# NOTES:
#   - Requires GPU with sufficient memory (~40GB for 2.7B model)
#   - This will take several minutes to complete all runs
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

# Configuration
MODEL_SIZE="2.7B"
CONTEXT_LENGTHS=(128 256 512)
BATCH_SIZE=4
WARMUP_STEPS=5
MEASURE_STEPS=10

SUMMARY_FILE="results/memory_profiling/peak_memory_summary.txt"

echo "=============================================================================="
echo "Memory Profiling - Part (b): Peak Memory Usage"
echo "=============================================================================="
echo ""
echo "Profiling ${MODEL_SIZE} model across different context lengths"
echo "Context lengths: ${CONTEXT_LENGTHS[@]}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Check if GPU is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Create output directory
mkdir -p results/memory_profiling

# Initialize summary file
cat > "${SUMMARY_FILE}" << 'EOF'
================================================================================
Peak Memory Usage Summary - Part (b)
================================================================================

Model: 2.7B
Batch size: 4

EOF

echo "Context Length | Forward Pass (MB) | Full Training Step (MB)" >> "${SUMMARY_FILE}"
echo "---------------|-------------------|-------------------------" >> "${SUMMARY_FILE}"

# Profile each context length
for CTX_LEN in "${CONTEXT_LENGTHS[@]}"; do
    echo "=============================================================================="
    echo "Context Length: ${CTX_LEN}"
    echo "=============================================================================="
    echo ""

    # Profile forward pass
    echo "------------------------------------------------------------------------------"
    echo "Profiling forward pass (context length: ${CTX_LEN})"
    echo "------------------------------------------------------------------------------"

    OUTPUT=$(python -m cs336_systems.memory_profiling.profile_memory \
        --model-size "${MODEL_SIZE}" \
        --context-length ${CTX_LEN} \
        --batch-size ${BATCH_SIZE} \
        --warmup-steps ${WARMUP_STEPS} \
        --measure-steps ${MEASURE_STEPS} \
        --profile-type forward \
        --output "results/memory_profiling/${MODEL_SIZE}_ctx${CTX_LEN}_forward_snapshot.pickle" 2>&1)

    # Extract peak memory from output
    FWD_PEAK=$(echo "$OUTPUT" | grep "Peak memory:" | awk '{print $3}')

    echo "Forward pass peak memory: ${FWD_PEAK} MB"
    echo ""

    # Profile full training step
    echo "------------------------------------------------------------------------------"
    echo "Profiling full training step (context length: ${CTX_LEN})"
    echo "------------------------------------------------------------------------------"

    OUTPUT=$(python -m cs336_systems.memory_profiling.profile_memory \
        --model-size "${MODEL_SIZE}" \
        --context-length ${CTX_LEN} \
        --batch-size ${BATCH_SIZE} \
        --warmup-steps ${WARMUP_STEPS} \
        --measure-steps ${MEASURE_STEPS} \
        --profile-type training \
        --output "results/memory_profiling/${MODEL_SIZE}_ctx${CTX_LEN}_training_snapshot.pickle" 2>&1)

    # Extract peak memory from output
    TRAIN_PEAK=$(echo "$OUTPUT" | grep "Peak memory:" | awk '{print $3}')

    echo "Training step peak memory: ${TRAIN_PEAK} MB"
    echo ""

    # Add to summary
    printf "%-14s | %-17s | %-23s\n" "${CTX_LEN}" "${FWD_PEAK}" "${TRAIN_PEAK}" >> "${SUMMARY_FILE}"
done

# Finalize summary
cat >> "${SUMMARY_FILE}" << 'EOF'

================================================================================
Notes:
- Peak memory includes model parameters, activations, and gradients
- Training step memory includes optimizer states (AdamW has ~2x parameter memory)
- Memory usage scales with context length due to activation growth
================================================================================
EOF

# Display summary
echo "=============================================================================="
echo "PART (b) COMPLETE"
echo "=============================================================================="
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""
echo "Memory snapshots saved to:"
for CTX_LEN in "${CONTEXT_LENGTHS[@]}"; do
    echo "  - results/memory_profiling/${MODEL_SIZE}_ctx${CTX_LEN}_forward_snapshot.pickle"
    echo "  - results/memory_profiling/${MODEL_SIZE}_ctx${CTX_LEN}_training_snapshot.pickle"
done
echo ""
