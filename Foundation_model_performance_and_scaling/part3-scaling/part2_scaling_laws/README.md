# Part 2: Scaling Laws with Training API

This module implements the full scaling laws workflow: querying the training API, fitting power laws, and predicting optimal model configurations for the target compute budget.

## Overview

Given a FLOPs budget of 2e18 for experiments, we:
1. Design experiment strategy (IsoFLOPs approach)
2. Query training API for multiple model sizes at different compute budgets
3. Fit power laws: N_opt ~ C^a, D_opt ~ C^b, L ~ C^c
4. Extrapolate to predict optimal configuration for 1e19 FLOPs
5. Select hyperparameters for the predicted model size

## Files

- `api_client.py` - Client for querying the training API
- `experiment_design.py` - Experiment strategy design
- `scaling_law_fitter.py` - Power law fitting using IsoFLOPs method
- `hyperparameter_selector.py` - Hyperparameter selection for target model size
- `run_experiments.py` - Main orchestration script
- `run_part2.sh` - Shell script to run the full pipeline
- `README.md` - This file

## Setup

### 1. Create API Key File

Create a file `../api_key.txt` containing your SSH public key (no newlines):

```bash
echo "your-ssh-public-key-here" > ../api_key.txt
```

**Important**: Ensure the API key is on a single line with no newlines.

### 2. Verify Stanford VPN Connection

The API requires Stanford VPN access. Check connectivity:

```bash
curl http://hyperturing.stanford.edu:8000/docs
```

You should see the API documentation page.

## Quick Start

### Run Full Pipeline (Dry Run First)

```bash
# See experiment strategy without querying API
./run_part2.sh --dry-run

# Run actual experiments (will query API)
./run_part2.sh
```

### Manual Steps

```bash
# Step 1: Run experiments and fit scaling laws
python run_experiments.py --api-key-file ../api_key.txt

# Step 2: Select hyperparameters (after fitting)
python select_hyperparameters.py
```

## Experiment Strategy

**Budget**: 2e18 FLOPs for experiments

**Approach**: IsoFLOPs method
- Test 8 compute budgets: 1e15, 3e15, 6e15, 1e16, 3e16, 6e16, 1e17, 3e17, 6e17, 1e18
- 6 model sizes per budget (varied via d_model and num_layers)
- Use standard hyperparameters: batch_size=256, learning_rate=3e-4
- Total: ~48 configurations

**Target**: Predict optimal model for 1e19 FLOPs (10× largest experiment)

## API Client Features

- **Caching**: Automatically caches results to avoid re-querying
- **Budget tracking**: Monitors total FLOPs used
- **Validation**: Validates configurations before querying
- **Retry logic**: Handles transient failures
- **Sync**: Can sync with API to retrieve previous runs

## Scaling Law Fitting

**Method**: IsoFLOPs (Chinchilla approach)
1. Group runs by compute budget
2. Find optimal model (lowest loss) for each budget
3. Fit power laws in log-log space
4. Extrapolate to target budget

**Outputs**:
- Power law formulas
- R² fit quality metrics
- Predictions for target budget
- Visualizations

## Hyperparameter Selection

**Given**: Target model size (from scaling law)

**Select**:
- `d_model`, `num_layers`, `num_heads` - Architecture matching target size
- `learning_rate` - Based on model size (larger models need smaller LR)
- `batch_size` - Must be 128 or 256 per assignment rules

## Output

Results are saved to `../results/part2_scaling_laws/`:

### Files Generated:
- `run_cache.json` - Cached API queries
- `experimental_runs.json` - All completed runs
- `scaling_law_results.json` - Fitted power laws and predictions
- `scaling_laws.png` - Scaling law plots (3 subplots)
- `final_config.json` - Selected hyperparameters for target budget
- `ANALYSIS_SUMMARY.txt` - Human-readable summary

## Usage Examples

### Check Current FLOPs Usage

```python
from api_client import TrainingAPIClient, load_api_key

api_key = load_api_key('../api_key.txt')
client = TrainingAPIClient(api_key)
total_flops = client.get_total_flops_used()
print(f"Total FLOPs used: {total_flops:.2e}")
```

### Query Single Configuration

```python
config = {
    'd_model': 512,
    'num_layers': 12,
    'num_heads': 8,
    'batch_size': 256,
    'learning_rate': 3e-4,
    'train_flops': int(1e16)
}

result = client.query_loss(config)
print(f"Loss: {result['loss']:.6f}")
```

### Design Custom Strategy

```python
from experiment_design import ExperimentDesigner

designer = ExperimentDesigner(budget=2e18)
configs = designer.create_isoflops_strategy(
    num_budgets=6,
    models_per_budget=5,
    batch_size=128,
    learning_rate=1e-4
)
designer.print_strategy_summary()
```

### Fit Scaling Laws

```python
from scaling_law_fitter import ScalingLawFitter

fitter = ScalingLawFitter(runs)
fitter.fit_isoflops()
prediction = fitter.predict_optimal_config(target_flops=1e19)
fitter.plot_scaling_laws(output_dir='../results/part2_scaling_laws')
```

## Important Notes

### Budget Management
- **Hard cap**: API enforces 2e18 FLOPs limit
- **Caching**: Re-querying same config doesn't count against budget
- **Check before running**: Use `--dry-run` to see strategy first

### Hyperparameter Constraints
- `d_model`: [64, 1024]
- `num_layers`: [2, 24]
- `num_heads`: [2, 16] and must divide d_model
- `batch_size`: Must be 128 or 256
- `learning_rate`: [1e-4, 1e-3]
- `train_flops`: Specific discrete values (1e13 to 1e18)

### Best Practices
1. Always run with `--dry-run` first to review strategy
2. Check current FLOPs usage before running
3. Start with cached runs if available (`--use-cached-only`)
4. Save results frequently (auto-saved after each query)
5. Generate plots and summaries for analysis

## Deliverables

This module generates all required deliverables:

1. **Code** (this directory) - All implementation
2. **Results** - Plots, predictions, and analysis
3. **Predictions** - For Google form submission:
   - Optimal model size (parameters)
   - Hyperparameters (d_model, num_layers, num_heads, batch_size, learning_rate)
   - Predicted loss

## Troubleshooting

### API Connection Issues
- Ensure you're on Stanford VPN
- Verify API key is correct (no newlines)
- Check API status: `curl http://hyperturing.stanford.edu:8000/docs`

### Budget Exceeded
- Use cached runs: `python run_experiments.py --use-cached-only`
- Check usage: Call `/total_flops_used` endpoint

### Configuration Errors
- Ensure `d_model % num_heads == 0`
- Use only allowed discrete `train_flops` values
- Batch size must be 128 or 256

## References

- Hoffmann et al., 2022: "Training compute-optimal large language models" (Chinchilla)
- Kaplan et al., 2020: "Scaling laws for neural language models"