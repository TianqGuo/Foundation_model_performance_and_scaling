#!/bin/bash
# ==============================================================================
# CS336 Assignment 2 - Section 1.1.5 Part (a): Accumulation Comparison
# ==============================================================================
#
# This script runs the mixed precision accumulation comparison experiment.
# It demonstrates why keeping accumulations in FP32 is important even when
# the values being accumulated are in lower precision.
#
# USAGE:
#   cd cs336_systems/mixed_precision
#   ./run_accumulation.sh
#
# OUTPUT:
#   - Terminal output showing results of 4 accumulation variants
#   - Demonstrates numerical precision issues with FP16 accumulation
#
# ASSIGNMENT DELIVERABLE:
#   Part (a): 2-3 sentence response about accuracy differences
#
# ==============================================================================

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "CS336 Assignment 2 Section 1.1.5 Part (a)"
echo "Mixed Precision Accumulation Comparison"
echo "=========================================="
echo ""

# Run the accumulation comparison
uv run python -m cs336_systems.mixed_precision.accumulation_comparison

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "For the writeup, explain:"
echo "  1. Why Variant 2 (FP16 accumulator) has the largest error"
echo "  2. Why Variants 3 & 4 have moderate error despite FP32 accumulator"
echo "  3. Why Variant 1 (full FP32) is most accurate"
echo ""
