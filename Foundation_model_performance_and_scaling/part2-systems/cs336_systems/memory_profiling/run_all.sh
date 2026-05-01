#!/bin/bash
# ==============================================================================
# Memory Profiling - Run All Parts
# ==============================================================================
#
# Master script to run all memory profiling tasks for section 1.1.6
#
# USAGE:
#   cd cs336_systems/memory_profiling
#   ./run_all.sh
#
# OUTPUT:
#   All memory snapshots and analysis reports in results/memory_profiling/
#
# NOTES:
#   - This will take 30-60 minutes to complete all profiling runs
#   - Requires GPU with ~40GB memory for 2.7B model
#   - Part (d) is a pure calculation and runs locally without GPU
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

echo "=============================================================================="
echo "Memory Profiling - Running All Parts (1.1.6)"
echo "=============================================================================="
echo ""
echo "This script will run all memory profiling tasks:"
echo "  Part (a): Forward and training step profiling for 2.7B model"
echo "  Part (b): Peak memory usage across context lengths (128, 256, 512)"
echo "  Part (c): Mixed precision memory comparison"
echo "  Part (d): Activation tensor size calculation"
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

# Part (a): Forward and training step profiling
echo ""
echo "=============================================================================="
echo "Running Part (a): Forward and Training Step Profiling"
echo "=============================================================================="
echo ""
./profile_part_a.sh

# Part (b): Context length comparison
echo ""
echo "=============================================================================="
echo "Running Part (b): Context Length Memory Comparison"
echo "=============================================================================="
echo ""
./profile_part_b.sh

# Part (c): Mixed precision comparison
echo ""
echo "=============================================================================="
echo "Running Part (c): Mixed Precision Memory Comparison"
echo "=============================================================================="
echo ""
./profile_part_c.sh

# Part (d): Activation size calculation (doesn't require GPU)
echo ""
echo "=============================================================================="
echo "Running Part (d): Activation Tensor Size Analysis"
echo "=============================================================================="
echo ""
python -m cs336_systems.memory_profiling.analyze_activation_size > \
    ../../results/memory_profiling/activation_size_analysis.txt

cat ../../results/memory_profiling/activation_size_analysis.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Final summary
echo ""
echo "=============================================================================="
echo "ALL MEMORY PROFILING TASKS COMPLETE"
echo "=============================================================================="
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Generated files:"
echo "  Memory snapshots:"
echo "    - results/memory_profiling/*.pickle"
echo "  Summary reports:"
echo "    - results/memory_profiling/peak_memory_summary.txt"
echo "    - results/memory_profiling/mixed_precision_memory_summary.txt"
echo "    - results/memory_profiling/activation_size_analysis.txt"
echo ""
echo "Next steps for writeup:"
echo ""
echo "  Part (a): Visualize memory snapshots at https://pytorch.org/memory_viz"
echo "    - Upload the .pickle files from results/memory_profiling/"
echo "    - Take screenshots of the 'Active memory timeline'"
echo "    - Include 2 screenshots: forward-only and full training step"
echo ""
echo "  Part (b): Use the table in peak_memory_summary.txt"
echo ""
echo "  Part (c): Use the comparison in mixed_precision_memory_summary.txt"
echo ""
echo "  Part (d): Use the calculation in activation_size_analysis.txt"
echo ""
echo "  Part (e): In pytorch.org/memory_viz, reduce 'Detail' slider to 10%"
echo "    - Identify the largest allocations shown"
echo "    - Check the stack trace for where they originate"
echo ""
echo "=============================================================================="
