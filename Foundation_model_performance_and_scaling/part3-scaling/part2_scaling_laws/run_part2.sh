#!/bin/bash
# ==============================================================================
# Part 2: Scaling Laws with Training API
# ==============================================================================
#
# This script runs the full scaling laws pipeline:
#   1. Designs experiment strategy within 2e18 FLOPs budget
#   2. Queries training API for multiple configurations
#   3. Fits power laws using IsoFLOPs method
#   4. Predicts optimal model for 1e19 FLOPs target budget
#   5. Selects hyperparameters for the predicted model
#
# REQUIREMENTS:
#   - API key file: ../api_key.txt (SSH public key, no newlines)
#   - Stanford VPN connection
#
# USAGE:
#   cd part2_scaling_laws
#
#   # Preview strategy without querying API
#   ./run_part2.sh --dry-run
#
#   # Run full pipeline
#   ./run_part2.sh
#
#   # Use only cached runs (no API queries)
#   ./run_part2.sh --use-cached
#
# OUTPUT:
#   results/part2_scaling_laws/
#       ├── experimental_runs.json       # All queried runs
#       ├── scaling_law_results.json     # Fitted power laws
#       ├── scaling_laws.png             # Scaling law plots
#       ├── final_config.json            # Selected hyperparameters
#       └── run_cache.json               # API query cache
#
# ==============================================================================

set -e  # Exit on error

# Navigate to script directory
cd "$(dirname "$0")"

# Parse arguments
DRY_RUN=""
USE_CACHED=""
API_KEY_FILE="../api_key.txt"
BUDGET="2e18"
TARGET_BUDGET="1e19"
OUTPUT_DIR="../results/part2_scaling_laws"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --use-cached)
            USE_CACHED="--use-cached-only"
            shift
            ;;
        --api-key-file)
            API_KEY_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Part 2: Scaling Laws with Training API"
echo "========================================="
echo ""

# Check API key exists
if [[ ! -f "$API_KEY_FILE" && -z "$DRY_RUN" && -z "$USE_CACHED" ]]; then
    echo "ERROR: API key file not found: $API_KEY_FILE"
    echo ""
    echo "Please create the file with your SSH public key (no newlines):"
    echo "  echo 'your-ssh-public-key' > $API_KEY_FILE"
    echo ""
    exit 1
fi

# Run experiments and fit scaling laws
echo "Running experiments and fitting scaling laws..."
echo ""

python run_experiments.py \
    --api-key-file "$API_KEY_FILE" \
    --budget "$BUDGET" \
    --target-budget "$TARGET_BUDGET" \
    --output-dir "$OUTPUT_DIR" \
    $DRY_RUN \
    $USE_CACHED

# Exit if dry run
if [[ -n "$DRY_RUN" ]]; then
    echo ""
    echo "Dry run complete. To run actual experiments, use:"
    echo "  ./run_part2.sh"
    exit 0
fi

# Check if we have results
if [[ ! -f "$OUTPUT_DIR/scaling_law_results.json" ]]; then
    echo ""
    echo "ERROR: Scaling law results not found."
    echo "Cannot proceed to hyperparameter selection."
    exit 1
fi

echo ""
echo "========================================="
echo "Selecting Hyperparameters"
echo "========================================="
echo ""

# Run hyperparameter selection
python -c "
import json
import sys
from pathlib import Path

# Import our modules
from hyperparameter_selector import select_hyperparameters_from_prediction

# Load scaling law results
results_file = Path('$OUTPUT_DIR') / 'scaling_law_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Load experimental runs
runs_file = Path('$OUTPUT_DIR') / 'experimental_runs.json'
if runs_file.exists():
    with open(runs_file, 'r') as f:
        runs = json.load(f)
else:
    runs = None

# Select hyperparameters
prediction = results['prediction']
config = select_hyperparameters_from_prediction(prediction, runs)

# Save final configuration
output_file = Path('$OUTPUT_DIR') / 'final_config.json'
with open(output_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f'\nFinal configuration saved to {output_file}')
"

echo ""
echo "========================================="
echo "Part 2 Complete!"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - scaling_law_results.json  (fitted power laws)"
echo "  - scaling_laws.png          (visualizations)"
echo "  - final_config.json         (selected hyperparameters)"
echo ""
echo "Next steps:"
echo "  1. Review the scaling law plots"
echo "  2. Check final_config.json for your predictions"
echo "  3. Submit predictions to Google form"
echo "  4. Write your methodology in writeup.pdf"
echo ""