#!/bin/bash
# ==============================================================================
# FlashAttention-2 Benchmarking Script
# ==============================================================================
#
# This script runs comprehensive benchmarks comparing FlashAttention-2
# (Triton implementation) with standard PyTorch attention.
#
# USAGE:
#   cd cs336_systems/flash_attention
#   ./run_benchmarks.sh
#
# OUTPUT:
#   results/flash_attention/flash_benchmarking.csv
#
# REQUIREMENTS:
#   - CUDA-enabled GPU
#   - Sufficient GPU memory (recommended: 16+ GB)
#   - triton, torch, pandas installed
#
# NOTES:
#   - Benchmarks sweep over seq_len (128-65536), d_model (16-128),
#     and dtypes (bfloat16, float32)
#   - Uses batch_size=1 and causal masking enabled
#   - OOM cases are gracefully handled and marked in output
#   - Runtime: ~30-60 minutes depending on GPU
#
# ==============================================================================

set -e  # Exit on error

echo "========================================"
echo "FlashAttention-2 Benchmarking"
echo "========================================"
echo ""

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This benchmark requires a GPU."
    exit 1
fi

# Run benchmarks
echo "Starting benchmarks..."
echo "This may take 30-60 minutes depending on your GPU."
echo ""

cd "$(dirname "$0")"
uv run python benchmark_flash_attention.py

echo ""
echo "========================================"
echo "Benchmarking complete!"
echo "========================================"