#!/bin/bash
# ==============================================================================
# Torch.compile Benchmarking - Run All Parts
# ==============================================================================
#
# Master script to run all torch.compile benchmarking tasks for section 1.3.1
#
# USAGE:
#   cd cs336_systems/torch_compile_benchmarking
#   ./run_all.sh
#
# OUTPUT:
#   All benchmarking results in results/torch_compile_benchmarking/
#
# NOTES:
#   - This will take 30-60 minutes to complete all benchmarks
#   - Requires PyTorch 2.0+ for torch.compile support
#   - Requires GPU with sufficient memory
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

echo "=============================================================================="
echo "Torch.compile Benchmarking - Running All Parts (1.3.1)"
echo "=============================================================================="
echo ""
echo "This script will run all torch.compile benchmarking tasks:"
echo "  Part (a): Compiled vs vanilla attention"
echo "  Part (b): Compiled vs vanilla Transformer models"
echo ""
echo "⚠️  This will take 30-60 minutes to complete."
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

START_TIME=$(date +%s)

# Part (a): Attention benchmarking
echo ""
echo "=============================================================================="
echo "Running Part (a): Compiled vs Vanilla Attention"
echo "=============================================================================="
echo ""
./run_part_a.sh

# Part (b): Transformer benchmarking
echo ""
echo "=============================================================================="
echo "Running Part (b): Compiled vs Vanilla Transformer"
echo "=============================================================================="
echo ""
./run_part_b.sh

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Final summary
echo ""
echo "=============================================================================="
echo "ALL TORCH.COMPILE BENCHMARKING TASKS COMPLETE"
echo "=============================================================================="
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Generated files:"
echo "  - results/torch_compile_benchmarking/compiled_attention_benchmark.csv"
echo "  - results/torch_compile_benchmarking/compiled_transformer_benchmark.csv"
echo ""
echo "Next steps for writeup:"
echo ""
echo "  Part (a): Create tables comparing compiled vs vanilla attention"
echo "    - Forward pass timings"
echo "    - Backward pass timings"
echo "    - Speedup analysis"
echo ""
echo "  Part (b): Create tables comparing compiled vs vanilla Transformer"
echo "    - Forward-only performance"
echo "    - Forward+backward performance"
echo "    - Full training step performance"
echo ""
echo "=============================================================================="
