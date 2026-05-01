#!/usr/bin/env python3
"""
IsoFLOPs Scaling Law Fitting (Part 1)

This script implements the IsoFLOPs method from Hoffmann et al. (2022) for fitting
scaling laws using training run data. It fits power laws to predict compute-optimal
model size and dataset size given a compute budget.

The method works as follows:
1. For each compute budget C_i, find the model size N_opt(C_i) that minimizes loss
2. Fit power laws: N_opt ∝ C^a and D_opt ∝ C^b
3. Extrapolate to larger compute budgets

Usage:
    python fit_scaling_laws.py --data ../data/isoflops_curves.json --output-dir ../results/part1_isoflops
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_training_data(filepath: str) -> List[Dict]:
    """Load training run data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def group_by_compute_budget(runs: List[Dict]) -> Dict[float, List[Dict]]:
    """Group training runs by compute budget."""
    grouped = {}
    for run in runs:
        budget = run['compute_budget']
        if budget not in grouped:
            grouped[budget] = []
        grouped[budget].append(run)
    return grouped


def find_optimal_runs(grouped_runs: Dict[float, List[Dict]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the optimal model size for each compute budget.

    Returns:
        compute_budgets: Array of compute budgets (C_i)
        optimal_params: Array of optimal model sizes (N_opt(C_i))
        optimal_tokens: Array of optimal dataset sizes (D_opt(C_i))
    """
    compute_budgets = []
    optimal_params = []
    optimal_tokens = []

    for budget, runs in sorted(grouped_runs.items()):
        # Find run with minimum loss for this budget
        min_loss_run = min(runs, key=lambda r: r['final_loss'])

        # Calculate dataset size D = C / (6N)
        N = min_loss_run['parameters']
        D = budget / (6 * N)

        compute_budgets.append(budget)
        optimal_params.append(N)
        optimal_tokens.append(D)

    return np.array(compute_budgets), np.array(optimal_params), np.array(optimal_tokens)


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, callable]:
    """
    Fit a power law to the data: y = a * x^b

    Returns:
        a: Coefficient
        b: Exponent
        predict_func: Function to make predictions
    """
    # Use log-log space for better fitting
    log_x = np.log(x)
    log_y = np.log(y)

    # Fit linear relationship in log space: log(y) = log(a) + b * log(x)
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    log_a = coeffs[1]
    a = np.exp(log_a)

    # Alternative: use curve_fit
    # params, _ = curve_fit(power_law, x, y, p0=[1e10, 0.5], maxfev=10000)
    # a, b = params

    def predict_func(x_new):
        return a * np.power(x_new, b)

    return a, b, predict_func


def create_scaling_law_plot(
    compute_budgets: np.ndarray,
    optimal_values: np.ndarray,
    predict_func: callable,
    ylabel: str,
    title: str,
    output_path: Path,
    extrapolate_to: float = 1e24
):
    """Create and save a scaling law plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot original data points
    ax.scatter(compute_budgets, optimal_values, s=100, alpha=0.6,
               label='Observed optimal points', zorder=3)

    # Plot fitted curve
    x_fit = np.logspace(np.log10(compute_budgets.min()),
                        np.log10(extrapolate_to), 100)
    y_fit = predict_func(x_fit)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2,
            label='Fitted power law', zorder=2)

    # Mark specific predictions
    for target_budget in [1e23, 1e24]:
        prediction = predict_func(target_budget)
        ax.scatter([target_budget], [prediction], s=200, marker='*',
                   color='gold', edgecolors='black', linewidths=1.5,
                   label=f'C={target_budget:.0e}: {ylabel}={prediction:.2e}',
                   zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fit IsoFLOPs scaling laws')
    parser.add_argument(
        '--data',
        type=str,
        default='../data/isoflops_curves.json',
        help='Path to training data JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results/part1_isoflops',
        help='Directory to save results'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IsoFLOPs Scaling Law Fitting (Chinchilla Method)")
    print("=" * 80)

    # Load data
    print(f"\n1. Loading training data from {args.data}")
    runs = load_training_data(args.data)
    print(f"   Loaded {len(runs)} training runs")

    # Group by compute budget
    print("\n2. Grouping runs by compute budget")
    grouped = group_by_compute_budget(runs)
    print(f"   Found {len(grouped)} unique compute budgets")
    for budget in sorted(grouped.keys()):
        print(f"      C = {budget:.2e}: {len(grouped[budget])} runs")

    # Find optimal runs
    print("\n3. Finding optimal model size for each compute budget")
    compute_budgets, optimal_params, optimal_tokens = find_optimal_runs(grouped)

    print("\n   Optimal points:")
    for c, n, d in zip(compute_budgets, optimal_params, optimal_tokens):
        print(f"      C = {c:.2e}: N_opt = {n:.2e}, D_opt = {d:.2e}")

    # Fit power law for model size
    print("\n4. Fitting power law for model size: N_opt = a * C^b")
    a_N, b_N, predict_N = fit_power_law(compute_budgets, optimal_params)
    print(f"   Fitted parameters: a = {a_N:.4e}, b = {b_N:.4f}")
    print(f"   Power law: N_opt = {a_N:.4e} * C^{b_N:.4f}")

    # Fit power law for dataset size
    print("\n5. Fitting power law for dataset size: D_opt = a * C^b")
    a_D, b_D, predict_D = fit_power_law(compute_budgets, optimal_tokens)
    print(f"   Fitted parameters: a = {a_D:.4e}, b = {b_D:.4f}")
    print(f"   Power law: D_opt = {a_D:.4e} * C^{b_D:.4f}")

    # Make predictions
    print("\n6. Predictions for target compute budgets")
    print("-" * 80)
    for target_C in [1e23, 1e24]:
        pred_N = predict_N(target_C)
        pred_D = predict_D(target_C)
        print(f"\n   Compute Budget: C = {target_C:.0e} FLOPs")
        print(f"      Optimal Model Size:   N_opt = {pred_N:.6e} parameters")
        print(f"      Optimal Dataset Size: D_opt = {pred_D:.6e} tokens")

    # Create plots
    print("\n7. Creating scaling law plots")

    # Model size plot
    create_scaling_law_plot(
        compute_budgets=compute_budgets,
        optimal_values=optimal_params,
        predict_func=predict_N,
        ylabel='Optimal Model Size (Parameters)',
        title='Scaling Law: Compute-Optimal Model Size',
        output_path=output_dir / 'model_size_scaling_law.png'
    )

    # Dataset size plot
    create_scaling_law_plot(
        compute_budgets=compute_budgets,
        optimal_values=optimal_tokens,
        predict_func=predict_D,
        ylabel='Optimal Dataset Size (Tokens)',
        title='Scaling Law: Compute-Optimal Dataset Size',
        output_path=output_dir / 'dataset_size_scaling_law.png'
    )

    # Save numerical results
    results = {
        'power_law_model_size': {
            'coefficient_a': float(a_N),
            'exponent_b': float(b_N),
            'formula': f'N_opt = {a_N:.4e} * C^{b_N:.4f}'
        },
        'power_law_dataset_size': {
            'coefficient_a': float(a_D),
            'exponent_b': float(b_D),
            'formula': f'D_opt = {a_D:.4e} * C^{b_D:.4f}'
        },
        'predictions': {
            '1e23_FLOPs': {
                'compute_budget': 1e23,
                'optimal_model_size': float(predict_N(1e23)),
                'optimal_dataset_size': float(predict_D(1e23))
            },
            '1e24_FLOPs': {
                'compute_budget': 1e24,
                'optimal_model_size': float(predict_N(1e24)),
                'optimal_dataset_size': float(predict_D(1e24))
            }
        },
        'data_points': {
            'compute_budgets': compute_budgets.tolist(),
            'optimal_model_sizes': optimal_params.tolist(),
            'optimal_dataset_sizes': optimal_tokens.tolist()
        }
    }

    results_file = output_dir / 'scaling_law_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   Results saved to {results_file}")

    print("\n" + "=" * 80)
    print("DONE! All results saved to:", output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()