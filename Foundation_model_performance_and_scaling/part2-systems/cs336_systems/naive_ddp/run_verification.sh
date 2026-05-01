#!/bin/bash
# ==============================================================================
# Run Naïve DDP Correctness Verification
# ==============================================================================
#
# DESCRIPTION:
#   Verifies that naïve DDP training produces identical results to single-process
#   training by comparing final model weights.
#
# USAGE:
#   cd cs336_systems/naive_ddp
#   ./run_verification.sh
#
# OUTPUT:
#   - Prints verification results (pass/fail)
#   - Returns exit code 0 if verification passes, 1 if fails
#
# REQUIREMENTS:
#   - At least 2 GPUs
#   - Runtime: ~1-2 minutes
#
# NOTES:
#   - Uses toy model for fast verification
#   - Safe to run on any multi-GPU system
#
# ==============================================================================

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "Naïve DDP Correctness Verification"
echo "================================================================================"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ Found $GPU_COUNT GPU(s)"
    echo ""
else
    echo "✗ Error: nvidia-smi not found"
    echo "  This verification requires GPUs"
    exit 1
fi

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "✗ Error: Need at least 2 GPUs for DDP verification"
    echo "  Found only $GPU_COUNT GPU(s)"
    exit 1
fi

# Run verification
echo "Running verification..."
echo "================================================================================"
echo ""

uv run python verify_correctness.py \
    --num-steps 20 \
    --batch-size 32 \
    --world-size 2 \
    --num-samples 1000

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================================"
    echo "✓ Verification complete - PASSED"
    echo "================================================================================"
else
    echo "================================================================================"
    echo "✗ Verification complete - FAILED"
    echo "================================================================================"
fi
echo ""

exit $EXIT_CODE