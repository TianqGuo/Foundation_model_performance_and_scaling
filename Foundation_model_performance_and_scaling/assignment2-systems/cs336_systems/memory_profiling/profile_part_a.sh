#!/bin/bash
# ==============================================================================
# Memory Profiling - Part (a)
# ==============================================================================
#
# Profile the 2.7B model for:
# 1. Forward pass only
# 2. Full training step (forward + backward + optimizer)
#
# This generates memory snapshots to visualize at https://pytorch.org/memory_viz
#
# USAGE:
#   cd cs336_systems/memory_profiling
#   ./profile_part_a.sh
#
# OUTPUT:
#   results/memory_profiling/2.7B_ctx512_forward_snapshot.pickle
#   results/memory_profiling/2.7B_ctx512_training_snapshot.pickle
#
# NOTES:
#   - Requires GPU with sufficient memory (~40GB for 2.7B model)
#   - Each profiling run takes several minutes
#   - Use context length 512 as default for part (a)
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

echo "=============================================================================="
echo "Memory Profiling - Part (a): 2.7B model"
echo "=============================================================================="
echo ""
echo "This will generate memory snapshots for:"
echo "  1. Forward pass only"
echo "  2. Full training step (forward + backward + optimizer)"
echo ""
echo "Context length: ${CONTEXT_LENGTH}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Check if GPU is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Profile forward pass only
echo "------------------------------------------------------------------------------"
echo "1. Profiling forward pass only"
echo "------------------------------------------------------------------------------"
echo ""

python -m cs336_systems.memory_profiling.profile_memory \
    --model-size "${MODEL_SIZE}" \
    --context-length ${CONTEXT_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --warmup-steps ${WARMUP_STEPS} \
    --measure-steps ${MEASURE_STEPS} \
    --profile-type forward \
    --output "../../results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_snapshot.pickle"

echo ""
echo "✓ Forward pass profiling complete"
echo ""

# Profile full training step
echo "------------------------------------------------------------------------------"
echo "2. Profiling full training step"
echo "------------------------------------------------------------------------------"
echo ""

python -m cs336_systems.memory_profiling.profile_memory \
    --model-size "${MODEL_SIZE}" \
    --context-length ${CONTEXT_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --warmup-steps ${WARMUP_STEPS} \
    --measure-steps ${MEASURE_STEPS} \
    --profile-type training \
    --output "../../results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_training_snapshot.pickle"

echo ""
echo "✓ Full training step profiling complete"
echo ""

# Summary
echo "=============================================================================="
echo "PART (a) COMPLETE"
echo "=============================================================================="
echo ""
echo "Memory snapshots saved to:"
echo "  - results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_snapshot.pickle"
echo "  - results/memory_profiling/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_training_snapshot.pickle"
echo ""
echo "To visualize:"
echo "  1. Open https://pytorch.org/memory_viz in your browser"
echo "  2. Drag and drop the pickle files onto the page"
echo "  3. Examine the 'Active memory timeline' for each snapshot"
echo "  4. Take screenshots of the timelines for your writeup"
echo ""
echo "For the writeup, describe:"
echo "  - How the timeline peaks align with execution stages"
echo "  - Differences between forward-only and full training step"
echo ""
