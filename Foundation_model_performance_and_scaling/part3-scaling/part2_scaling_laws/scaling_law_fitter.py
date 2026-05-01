"""
Scaling Law Fitting

Fits power laws to experimental training data using various methods.
Supports IsoFLOPs approach and direct loss prediction.
"""

from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class ScalingLawFitter:
    """Fits scaling laws to training data."""

    def __init__(self, runs: List[Dict]):
        """
        Initialize with training runs.

        Args:
            runs: List of training run dictionaries with keys:
                  d_model, num_layers, num_heads, batch_size, learning_rate,
                  train_flops, loss, estimated_params (optional)
        """
        self.runs = runs
        self.compute_budgets = None
        self.optimal_params = None
        self.optimal_tokens = None
        self.optimal_losses = None
        self.param_predictor = None
        self.token_predictor = None
        self.loss_predictor = None

    @staticmethod
    def estimate_params(d_model: int, num_layers: int) -> int:
        """Estimate model parameters."""
        return 12 * num_layers * (d_model ** 2)

    @staticmethod
    def estimate_tokens(train_flops: float, params: int) -> float:
        """Estimate tokens from FLOPs: D = C / (6N)"""
        return train_flops / (6 * params)

    def prepare_data(self):
        """Prepare data by computing parameters and tokens for each run."""
        for run in self.runs:
            if 'estimated_params' not in run:
                run['estimated_params'] = self.estimate_params(
                    run['d_model'], run['num_layers']
                )

            if 'estimated_tokens' not in run:
                run['estimated_tokens'] = self.estimate_tokens(
                    run['train_flops'], run['estimated_params']
                )

    def fit_isoflops(self) -> Tuple[callable, callable, callable]:
        """
        Fit scaling laws using IsoFLOPs method.

        For each compute budget, finds the model with lowest loss,
        then fits power laws: N_opt ~ C^a, D_opt ~ C^b, L_opt ~ C^c

        Returns:
            (param_predictor, token_predictor, loss_predictor)
        """
        self.prepare_data()

        # Group runs by compute budget
        budget_groups = {}
        for run in self.runs:
            budget = run['train_flops']
            if budget not in budget_groups:
                budget_groups[budget] = []
            budget_groups[budget].append(run)

        # Find optimal run for each budget
        compute_budgets = []
        optimal_params = []
        optimal_tokens = []
        optimal_losses = []

        for budget in sorted(budget_groups.keys()):
            runs_at_budget = budget_groups[budget]
            # Find run with minimum loss
            best_run = min(runs_at_budget, key=lambda r: r['loss'])

            compute_budgets.append(budget)
            optimal_params.append(best_run['estimated_params'])
            optimal_tokens.append(best_run['estimated_tokens'])
            optimal_losses.append(best_run['loss'])

        self.compute_budgets = np.array(compute_budgets)
        self.optimal_params = np.array(optimal_params)
        self.optimal_tokens = np.array(optimal_tokens)
        self.optimal_losses = np.array(optimal_losses)

        # Fit power laws
        self.param_predictor = self._fit_power_law(
            self.compute_budgets, self.optimal_params
        )
        self.token_predictor = self._fit_power_law(
            self.compute_budgets, self.optimal_tokens
        )
        self.loss_predictor = self._fit_power_law(
            self.compute_budgets, self.optimal_losses
        )

        return self.param_predictor, self.token_predictor, self.loss_predictor

    @staticmethod
    def _fit_power_law(x: np.ndarray, y: np.ndarray) -> callable:
        """
        Fit power law: y = a * x^b

        Returns a function that can make predictions.
        """
        # Fit in log-log space
        log_x = np.log(x)
        log_y = np.log(y)

        # Linear fit: log(y) = log(a) + b * log(x)
        coeffs = np.polyfit(log_x, log_y, 1)
        b = coeffs[0]
        log_a = coeffs[1]
        a = np.exp(log_a)

        # Create predictor function
        def predictor(x_new):
            if isinstance(x_new, (list, np.ndarray)):
                return a * np.power(x_new, b)
            else:
                return a * (x_new ** b)

        # Store coefficients as attributes
        predictor.a = a
        predictor.b = b
        predictor.formula = f"y = {a:.4e} * x^{b:.4f}"

        return predictor

    def predict_optimal_config(self, target_flops: float) -> Dict:
        """
        Predict optimal configuration for target FLOPs.

        Args:
            target_flops: Target compute budget

        Returns:
            Dictionary with predicted values
        """
        if not self.param_predictor:
            raise ValueError("Must call fit_isoflops() first")

        pred_params = self.param_predictor(target_flops)
        pred_tokens = self.token_predictor(target_flops)
        pred_loss = self.loss_predictor(target_flops)

        return {
            'compute_budget': target_flops,
            'optimal_params': pred_params,
            'optimal_tokens': pred_tokens,
            'predicted_loss': pred_loss
        }

    def compute_fit_quality(self) -> Dict:
        """
        Compute quality metrics for the fitted scaling laws.

        Returns:
            Dictionary with R^2 values and other metrics
        """
        if not self.param_predictor:
            raise ValueError("Must call fit_isoflops() first")

        # Compute R^2 for each fit
        def compute_r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)

        pred_params = self.param_predictor(self.compute_budgets)
        pred_tokens = self.token_predictor(self.compute_budgets)
        pred_losses = self.loss_predictor(self.compute_budgets)

        r2_params = compute_r2(self.optimal_params, pred_params)
        r2_tokens = compute_r2(self.optimal_tokens, pred_tokens)
        r2_losses = compute_r2(self.optimal_losses, pred_losses)

        return {
            'r2_params': r2_params,
            'r2_tokens': r2_tokens,
            'r2_losses': r2_losses,
            'num_data_points': len(self.compute_budgets)
        }

    def plot_scaling_laws(self, output_dir: str, target_budget: float = 1e19):
        """
        Create plots of the fitted scaling laws.

        Args:
            output_dir: Directory to save plots
            target_budget: Target budget to highlight
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Extrapolation range
        x_min = self.compute_budgets.min()
        x_max = max(self.compute_budgets.max(), target_budget * 1.2)
        x_range = np.logspace(np.log10(x_min), np.log10(x_max), 100)

        # Plot 1: Model Size
        ax = axes[0]
        ax.scatter(self.compute_budgets, self.optimal_params,
                  s=100, alpha=0.6, label='Observed optimal', zorder=3)
        ax.plot(x_range, self.param_predictor(x_range),
               'r-', linewidth=2, label='Fitted power law', zorder=2)
        pred_params = self.param_predictor(target_budget)
        ax.scatter([target_budget], [pred_params],
                  s=200, marker='*', color='gold', edgecolors='black',
                  linewidths=1.5, label=f'Prediction at C={target_budget:.0e}',
                  zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Compute Budget (FLOPs)', fontsize=11)
        ax.set_ylabel('Optimal Model Size (Parameters)', fontsize=11)
        ax.set_title('Model Size Scaling Law', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Plot 2: Dataset Size
        ax = axes[1]
        ax.scatter(self.compute_budgets, self.optimal_tokens,
                  s=100, alpha=0.6, label='Observed optimal', zorder=3)
        ax.plot(x_range, self.token_predictor(x_range),
               'r-', linewidth=2, label='Fitted power law', zorder=2)
        pred_tokens = self.token_predictor(target_budget)
        ax.scatter([target_budget], [pred_tokens],
                  s=200, marker='*', color='gold', edgecolors='black',
                  linewidths=1.5, label=f'Prediction at C={target_budget:.0e}',
                  zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Compute Budget (FLOPs)', fontsize=11)
        ax.set_ylabel('Optimal Dataset Size (Tokens)', fontsize=11)
        ax.set_title('Dataset Size Scaling Law', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Plot 3: Loss
        ax = axes[2]
        ax.scatter(self.compute_budgets, self.optimal_losses,
                  s=100, alpha=0.6, label='Observed optimal loss', zorder=3)
        ax.plot(x_range, self.loss_predictor(x_range),
               'r-', linewidth=2, label='Fitted power law', zorder=2)
        pred_loss = self.loss_predictor(target_budget)
        ax.scatter([target_budget], [pred_loss],
                  s=200, marker='*', color='gold', edgecolors='black',
                  linewidths=1.5, label=f'Prediction at C={target_budget:.0e}',
                  zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Compute Budget (FLOPs)', fontsize=11)
        ax.set_ylabel('Optimal Training Loss', fontsize=11)
        ax.set_title('Loss Scaling Law', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        plt.tight_layout()
        plot_file = output_path / 'scaling_laws.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scaling law plots saved to {plot_file}")

    def print_summary(self, target_budget: float = 1e19):
        """Print summary of fitted scaling laws."""
        if not self.param_predictor:
            raise ValueError("Must call fit_isoflops() first")

        print("\n" + "=" * 80)
        print("SCALING LAW FITTING RESULTS")
        print("=" * 80)

        # Power law formulas
        print("\nFitted Power Laws:")
        print(f"  Model Size:   {self.param_predictor.formula}")
        print(f"  Dataset Size: {self.token_predictor.formula}")
        print(f"  Loss:         {self.loss_predictor.formula}")

        # Fit quality
        quality = self.compute_fit_quality()
        print(f"\nFit Quality (R² scores):")
        print(f"  Model Size:   {quality['r2_params']:.4f}")
        print(f"  Dataset Size: {quality['r2_tokens']:.4f}")
        print(f"  Loss:         {quality['r2_losses']:.4f}")
        print(f"  Data Points:  {quality['num_data_points']}")

        # Prediction for target budget
        prediction = self.predict_optimal_config(target_budget)
        print(f"\nPrediction for Target Budget (C = {target_budget:.2e} FLOPs):")
        print(f"  Optimal Model Size:   {prediction['optimal_params']:.6e} parameters")
        print(f"  Optimal Dataset Size: {prediction['optimal_tokens']:.6e} tokens")
        print(f"  Predicted Loss:       {prediction['predicted_loss']:.6f}")

        print("=" * 80)

        return prediction