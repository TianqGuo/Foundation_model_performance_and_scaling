#!/bin/bash
# ==============================================================================
# Torch.compile Transformer Benchmarking - Section 1.3.1(b)
# ==============================================================================
#
# This script benchmarks compiled vs vanilla Transformer models.
#
# USAGE:
#   cd cs336_systems/torch_compile_benchmarking
#   ./run_part_b.sh
#
# OUTPUT:
#   results/torch_compile_benchmarking/compiled_transformer_benchmark.csv
#
# NOTES:
#   - This will take 20-40 minutes depending on GPU
#   - Tests small, medium, and large models
#   - Compares forward-only, forward+backward, and full training step
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

OUTPUT_DIR="../../results/torch_compile_benchmarking"
BENCHMARK_CSV="${OUTPUT_DIR}/compiled_transformer_benchmark.csv"

echo "=============================================================================="
echo "Torch.compile Transformer Benchmarking - Section 1.3.1(b)"
echo "=============================================================================="
echo ""
echo "This will benchmark Transformer models comparing vanilla vs torch.compile"
echo ""
echo "Model sizes: small, medium, large"
echo "Context lengths: 512, 1024"
echo "Pass types: forward-only, forward+backward, full training step"
echo ""

# Check if GPU is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. This script requires a GPU."
    exit 1
fi

# Check if torch.compile is available
if ! python -c "import torch; assert hasattr(torch, 'compile')" 2>/dev/null; then
    echo "ERROR: torch.compile not available. Requires PyTorch 2.0+"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Running benchmarks..."
echo ""

# Run the benchmark script
python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_transformer \
    --all-configs \
    --warmup-steps 5 \
    --measure-steps 10 \
    --output "${BENCHMARK_CSV}"

echo ""
echo "=============================================================================="
echo "BENCHMARK COMPLETE"
echo "=============================================================================="
echo ""
echo "Results saved to: ${BENCHMARK_CSV}"
echo ""
echo "=============================================================================="
echo "NEXT STEPS FOR WRITEUP"
echo "=============================================================================="
echo ""
echo "1. Review the comparison tables in the output above"
echo ""
echo "2. For the writeup (Section 1.3.1(b)), include:"
echo "   - Table comparing forward pass performance (vanilla vs compiled)"
echo "   - Table comparing forward+backward performance"
echo "   - Table comparing full training step performance"
echo "   - Speedup ratios for each configuration"
echo ""
echo "3. Key questions to answer:"
echo "   - How does forward pass performance change with torch.compile?"
echo "   - What about forward+backward passes?"
echo "   - How does optimizer step affect overall speedup?"
echo "   - Do speedups scale with model size?"
echo ""
echo "=============================================================================="
