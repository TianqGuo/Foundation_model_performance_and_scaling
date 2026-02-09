#!/bin/bash
# ==============================================================================
# Part 1: IsoFLOPs Scaling Law Fitting
# ==============================================================================
#
# This script reproduces the IsoFLOPs method from Hoffmann et al. (2022) for
# fitting scaling laws. It processes the synthetic training data and produces:
#   1. Scaling law for model size vs compute budget
#   2. Scaling law for dataset size vs compute budget
#   3. Predictions for 10^23 and 10^24 FLOPs budgets
#
# USAGE:
#   cd part1_isoflops
#   ./run_part1.sh
#
# OUTPUT:
#   results/part1_isoflops/
#       ├── model_size_scaling_law.png      # Model size scaling plot
#       ├── dataset_size_scaling_law.png    # Dataset size scaling plot
#       └── scaling_law_results.json        # Numerical results
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

echo "========================================="
echo "Running Part 1: IsoFLOPs Scaling Laws"
echo "========================================="
echo ""

# Run the fitting script
python fit_scaling_laws.py \
    --data ../data/isoflops_curves.json \
    --output-dir ../results/part1_isoflops

echo ""
echo "========================================="
echo "Part 1 Complete!"
echo "========================================="
echo ""
echo "Check results in: ../results/part1_isoflops/"
echo ""