#!/bin/bash
# ==============================================================================
# CS336 Assignment 2 - Section 1.1.5 Part (b) Question (a): ToyModel DTypes
# ==============================================================================
#
# This script analyzes data types in a toy model under FP16 mixed precision.
# It shows how torch.autocast selectively casts operations and why layer
# normalization is treated differently from linear layers.
#
# USAGE:
#   cd cs336_systems/mixed_precision
#   ./run_toy_model.sh
#
# OUTPUT:
#   - Terminal output showing dtypes for:
#     - Model parameters within autocast
#     - fc1 output
#     - Layer norm output
#     - Predicted logits
#     - Loss
#     - Gradients
#
# ASSIGNMENT DELIVERABLE:
#   Part (b) Question (a): List data types for each component
#   Part (b) Question (b): 2-3 sentences about layer norm sensitivity
#
# REQUIREMENTS:
#   - CUDA-capable GPU
#
# ==============================================================================

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "CS336 Assignment 2 Section 1.1.5 Part (b) Question (a)"
echo "ToyModel Data Type Analysis"
echo "=========================================="
echo ""

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Run the toy model dtype analysis
uv run python -m cs336_systems.mixed_precision.toy_model_dtypes

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "For the writeup:"
echo "  Question (a): Record the data types listed above"
echo "  Question (b): Explain why layer norm is kept in FP32 and whether"
echo "               BF16 requires the same treatment (see output above)"
echo ""
