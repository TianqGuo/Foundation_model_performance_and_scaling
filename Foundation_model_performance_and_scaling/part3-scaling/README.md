# Scaling Laws: Predicting Compute-Optimal Language Models

Implementation of the IsoFLOPs method (Hoffmann et al., 2022 / Chinchilla) for fitting scaling laws and predicting compute-optimal model configurations.

**Goal**: Given a fixed compute budget, what is the optimal tradeoff between model size and training tokens?

## What This Project Does

### Part 1: IsoFLOPs Scaling Law Fitting

Reproduces the Chinchilla IsoFLOPs method using synthetic training run data. For each compute budget, finds the model size that minimizes training loss, then fits power laws to predict compute-optimal configurations at much larger budgets.

**Method:**
1. For each compute budget C, find the model size N_opt(C) that minimizes loss
2. Fit power laws: N_opt ∝ C^a and D_opt ∝ C^b
3. Extrapolate to unseen compute budgets

**Fitted scaling laws** (from 9 IsoFLOP profiles across 6×10¹⁸ – 3×10²¹ FLOPs):

| Law | Formula |
|-----|---------|
| Compute-optimal model size | N_opt = 1.16 × C^0.469 |
| Compute-optimal dataset size | D_opt = 0.143 × C^0.531 |

**Extrapolated predictions:**

| Compute budget | Optimal model size | Optimal tokens |
|---|---|---|
| 1×10²³ FLOPs | ~70B parameters | ~238B tokens |
| 1×10²⁴ FLOPs | ~206B parameters | ~809B tokens |

These exponents (~0.47 / ~0.53) closely match the Chinchilla paper, confirming roughly equal scaling of model size and data with compute.

**Scaling law plots:**

![Compute-optimal model size scaling law](results/part1_isoflops/model_size_scaling_law.png)

![Compute-optimal dataset size scaling law](results/part1_isoflops/dataset_size_scaling_law.png)

### Part 2: Empirical Scaling Laws via Training API

Queries a live training API (Stanford cluster) with a budget of 2×10¹⁸ FLOPs to run real training experiments across model sizes and compute budgets, fits scaling laws from the results, and predicts the optimal model configuration for a target budget of 1×10¹⁹ FLOPs.

The pipeline includes:
- **Experiment design** — IsoFLOPs sweep across model sizes at each compute level
- **Scaling law fitting** — power law regression with R² quality assessment
- **Hyperparameter selection** — architecture matching for target model size, LR scaling

---

## Repository Layout

```
part3-scaling/
├── part1_isoflops/
│   ├── fit_scaling_laws.py      # IsoFLOPs fitting implementation
│   └── run_part1.sh             # Runner script
├── part2_scaling_laws/
│   ├── api_client.py            # API client with caching + budget tracking
│   ├── experiment_design.py     # IsoFLOPs sweep strategy
│   ├── scaling_law_fitter.py    # Power law regression
│   ├── hyperparameter_selector.py
│   ├── run_experiments.py       # Main orchestration
│   └── run_part2.sh             # Runner script
├── results/
│   └── part1_isoflops/          # Scaling law plots and fitted parameters
├── data/
│   └── isoflops_curves.json     # Synthetic training run data
└── cs336_scaling/
    └── model.py                 # Transformer model definition
```

## Setup

```bash
uv sync
```

## Running

### Part 1 — IsoFLOPs fitting on synthetic data

```bash
cd part1_isoflops && ./run_part1.sh
# → results/part1_isoflops/
```

### Part 2 — Empirical scaling laws (requires Stanford VPN + API key)

```bash
echo "your-ssh-public-key" > api_key.txt

cd part2_scaling_laws
python test_api_connection.py   # verify access
./run_part2.sh --dry-run        # preview experiment plan
./run_part2.sh                  # run experiments
```

## References

- Hoffmann et al., 2022 — [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- Kaplan et al., 2020 — [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Yang et al., 2022 — [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (μP)](https://arxiv.org/abs/2203.03466)
- Stanford CS336 Spring 2025 — [Language Models from Scratch, Part 3 (course framework)](https://github.com/stanford-cs336/assignment3-scaling)