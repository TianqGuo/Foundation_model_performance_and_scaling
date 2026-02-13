# CS336 Spring 2025 Assignment 3: Scaling Laws

Implementation of scaling laws experiments for predicting compute-optimal language model configurations.

For the full assignment description, see [cs336_spring2025_assignment3_scaling.pdf](./cs336_spring2025_assignment3_scaling.pdf)

## Assignment Overview

**Objective**: Predict the optimal model size and hyperparameters for a compute budget of **1×10^19 FLOPs**.

**Approach**:
1. **Part 1**: Reproduce IsoFLOPs scaling law fitting method using synthetic data (5 points)
2. **Part 2**: Query training API with budget of 2×10^18 FLOPs, fit scaling laws, and predict optimal configuration (50 points)

## Setup

### 1. Install Dependencies

```sh
# Install uv (if not already installed)
# Follow instructions at https://github.com/astral-sh/uv

# Install project dependencies
uv add numpy scipy matplotlib requests torch

# Or install in existing environment
uv pip install numpy scipy matplotlib requests torch
```

### 2. Run Commands with uv

```sh
# Run any command in the environment
uv run <command>

# Get Python interpreter path (for VSCode)
uv run which python
```

### 3. Setup for Part 2 (API Access)

Create API key file:
```sh
echo "your-ssh-public-key-here" > api_key.txt
```

**Important**: Ensure you're on Stanford VPN to access the training API.

## Quick Start

### Part 1: IsoFLOPs Method

```bash
cd part1_isoflops
./run_part1.sh
```

**Output**: Scaling law plots and predictions for 10^23 and 10^24 FLOPs in `results/part1_isoflops/`.

### Part 2: Full Scaling Laws Pipeline

```bash
cd part2_scaling_laws

# Test API connection first
python test_api_connection.py

# Preview strategy without querying API
./run_part2.sh --dry-run

# Run full pipeline
./run_part2.sh
```

**Output**: Scaling law plots, predictions, and selected hyperparameters in `results/part2_scaling_laws/`.

## Project Structure

```
assignment3-scaling/
├── part1_isoflops/              # Part 1: IsoFLOPs method
│   ├── fit_scaling_laws.py     # Main implementation
│   ├── run_part1.sh            # Run script
│   └── README.md               # Documentation
│
├── part2_scaling_laws/          # Part 2: Training API experiments
│   ├── api_client.py           # API client with caching
│   ├── experiment_design.py    # Experiment strategy
│   ├── scaling_law_fitter.py   # Power law fitting
│   ├── hyperparameter_selector.py  # Hyperparameter selection
│   ├── run_experiments.py      # Main orchestration
│   ├── test_api_connection.py  # Connection testing
│   ├── generate_submission.py  # Submission summary
│   ├── run_part2.sh            # Run script
│   └── README.md               # Documentation
│
├── results/                     # All outputs (organized by module)
│   ├── part1_isoflops/
│   └── part2_scaling_laws/
│
├── data/
│   └── isoflops_curves.json    # Synthetic training data
│
├── cs336_scaling/               # Model code (provided)
│   └── model.py
│
├── Requirements.md              # Assignment requirements breakdown
├── WRITEUP_TEMPLATE.md          # Template for writeup.pdf
└── README.md                    # This file
```

## Workflow

### Complete Pipeline

1. **Part 1**: Reproduce IsoFLOPs method
   ```bash
   cd part1_isoflops
   ./run_part1.sh
   ```

2. **Part 2**: Run scaling law experiments
   ```bash
   cd part2_scaling_laws
   python test_api_connection.py  # Verify API access
   ./run_part2.sh --dry-run        # Preview strategy
   ./run_part2.sh                  # Run experiments
   python generate_submission.py   # Generate submission summary
   ```

3. **Create Writeup**: Use `WRITEUP_TEMPLATE.md` as starting point

4. **Submit**:
   - `writeup.pdf` to Gradescope
   - `code.zip` to Gradescope
   - Predictions to Google form: https://forms.gle/sAUSLwCUETew2hYN6

## Key Features

### API Client (`part2_scaling_laws/api_client.py`)
- Automatic caching to avoid re-querying same configurations
- Budget tracking and monitoring
- Configuration validation before querying
- Retry logic for transient failures

### Experiment Design (`part2_scaling_laws/experiment_design.py`)
- IsoFLOPs approach: multiple model sizes at each compute budget
- Efficient use of 2×10^18 FLOPs budget
- Systematic exploration from 1e15 to 1e18 FLOPs

### Scaling Law Fitting (`part2_scaling_laws/scaling_law_fitter.py`)
- Power law fitting: N_opt ~ C^a, D_opt ~ C^b, L ~ C^c
- R² scores for fit quality assessment
- Extrapolation to target budget (1e19 FLOPs)

### Hyperparameter Selection (`part2_scaling_laws/hyperparameter_selector.py`)
- Architecture matching for target model size
- Learning rate scaling based on model size
- Batch size constraint satisfaction (128 or 256)

## Deliverables

### 1. writeup.pdf
Complete methodology and results using `WRITEUP_TEMPLATE.md`

### 2. code.zip
All implementation code (both part1_isoflops/ and part2_scaling_laws/)

### 3. Google Form
- Predicted optimal model size (parameters)
- Hyperparameters: d_model, num_layers, num_heads, batch_size, learning_rate
- Predicted training loss

Generate submission values:
```bash
cd part2_scaling_laws
python generate_submission.py
```

## Important Constraints

### Budget
- **Part 2 budget**: 2×10^18 FLOPs (hard cap enforced by API)
- **Target budget**: 1×10^19 FLOPs (for predictions)

### Hyperparameters
- `d_model`: [64, 1024], must be divisible by num_heads
- `num_layers`: [2, 24]
- `num_heads`: [2, 16]
- `batch_size`: **Must be 128 or 256** (assignment requirement)
- `learning_rate`: [1e-4, 1e-3]
- `train_flops`: Discrete values {1e13, 3e13, 6e13, ..., 1e18}

## Troubleshooting

### API Connection Issues
```bash
# Verify Stanford VPN connection
curl http://hyperturing.stanford.edu:8000/docs

# Test API connection and key
cd part2_scaling_laws
python test_api_connection.py
```

### Budget Exceeded
```bash
# Use only cached runs
cd part2_scaling_laws
./run_part2.sh --use-cached
```

### Line Ending Issues (WSL/Windows)
```bash
# Fix all shell scripts
./fix_line_endings.sh
```

## References

- Hoffmann et al., 2022: "Training compute-optimal large language models" (Chinchilla, arXiv:2203.15556)
- Kaplan et al., 2020: "Scaling laws for neural language models" (arXiv:2001.08361)
- Yang et al., 2022: "Tensor Programs V" (μP, arXiv:2203.03466)

## Additional Documentation

- `Requirements.md` - Detailed assignment requirements
- `WRITEUP_TEMPLATE.md` - Template for final writeup
- `CODING_PREFERENCES.md` - Code organization guidelines
- `part1_isoflops/README.md` - Part 1 documentation
- `part2_scaling_laws/README.md` - Part 2 documentation

## Issues

If you see any issues with the assignment handout or code, please raise a GitHub issue or open a pull request with a fix.

