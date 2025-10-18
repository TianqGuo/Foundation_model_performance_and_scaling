#!/bin/bash
# Run OpenWebText experiment with same compute budget as TinyStories

set -e

echo "=================================="
echo "OpenWebText Training Experiment"
echo "=================================="
echo ""
echo "Configuration:"
echo "  - Vocab size: 32,000 (vs 10,000 for TinyStories)"
echo "  - Model: 22.7M parameters (same as TinyStories)"
echo "  - Training: 40,000 iterations"
echo "  - Compute: 327.68M tokens (same as TinyStories)"
echo ""
echo "Starting training..."
echo ""

uv run python cs336_basics/basics/run_experiment.py \
    --name openwebtext_main \
    --config cs336_basics/assignment_experiments/openwebtext_experiment/config_owt.json \
    --output-dir cs336_basics/basics/runs/openwebtext \
    --description "Main experiment on OpenWebText with same architecture and compute as TinyStories"

echo ""
echo "=================================="
echo "Training complete!"
echo "=================================="
echo ""
echo "Results saved to: cs336_basics/basics/runs/openwebtext/"
echo ""
echo "Next steps:"
echo "  1. Generate text samples"
echo "  2. Compare with TinyStories results"
echo "  3. Analyze learning curves"
