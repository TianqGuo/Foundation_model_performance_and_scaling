# Part 1: IsoFLOPs Scaling Law Fitting

This module implements the IsoFLOPs method from Hoffmann et al. (2022) for fitting scaling laws using synthetic training run data.

## Overview

The IsoFLOPs approach fits power laws to predict compute-optimal model size and dataset size given a compute budget:
- **Model size**: N_opt ∝ C^a
- **Dataset size**: D_opt ∝ C^b

Where C is the compute budget in FLOPs.

## Files

- `fit_scaling_laws.py` - Main implementation script
- `run_part1.sh` - Shell script to run the analysis
- `README.md` - This file

## Quick Start

```bash
# From the part1_isoflops directory
./run_part1.sh
```

## Usage

### Running with default settings:
```bash
python fit_scaling_laws.py
```

### Custom input/output paths:
```bash
python fit_scaling_laws.py \
    --data /path/to/data.json \
    --output-dir /path/to/results
```

## Output

Results are saved to `../results/part1_isoflops/`:

- `model_size_scaling_law.png` - Plot showing optimal model size vs compute budget
- `dataset_size_scaling_law.png` - Plot showing optimal dataset size vs compute budget
- `scaling_law_results.json` - Numerical results including:
  - Fitted power law parameters
  - Predictions for 10^23 and 10^24 FLOPs
  - All data points used for fitting

## Method

1. **Load data**: Read training runs from JSON file
2. **Group by budget**: Organize runs by compute budget
3. **Find optimal**: For each budget, select run with minimum loss
4. **Fit power laws**: Use least squares in log-log space
5. **Extrapolate**: Make predictions for larger budgets
6. **Visualize**: Create scaling law plots

## Requirements

- Python 3.12+
- numpy
- scipy
- matplotlib

Install dependencies:
```bash
uv pip install numpy scipy matplotlib
```

## Deliverables (Assignment Part 1)

1. **Plot 1**: Model size scaling law with extrapolation to 10^24 FLOPs
2. **Plot 2**: Dataset size scaling law with extrapolation to 10^24 FLOPs
3. **Predictions**: Optimal model and dataset sizes for 10^23 and 10^24 FLOPs

All deliverables are automatically generated when running the script.