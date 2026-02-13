"""
Hyperparameter Selection

Selects optimal hyperparameters (d_model, num_layers, num_heads, learning_rate, batch_size)
for a target model size based on experimental data and best practices.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class HyperparameterSelector:
    """Selects hyperparameters for a target model size."""

    @staticmethod
    def estimate_params(d_model: int, num_layers: int) -> int:
        """Estimate model parameters: 12 * num_layers * d_model^2"""
        return 12 * num_layers * (d_model ** 2)

    def find_architecture(
        self,
        target_params: float,
        prefer_depth: bool = True
    ) -> List[Dict]:
        """
        Find (d_model, num_layers, num_heads) combinations for target parameters.

        Args:
            target_params: Target number of parameters
            prefer_depth: If True, prefer deeper (more layers) over wider (larger d_model)

        Returns:
            List of candidate architectures sorted by closeness to target
        """
        candidates = []

        # Valid ranges
        d_models = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640,
                   704, 768, 832, 896, 960, 1024]
        num_layers_options = list(range(2, 25))

        for d_model in d_models:
            for num_layers in num_layers_options:
                # Find valid num_heads (must divide d_model)
                valid_heads = [h for h in range(2, 17) if d_model % h == 0]
                if not valid_heads:
                    continue

                # Choose num_heads that gives good head dimension (64 or 128)
                num_heads = self._choose_num_heads(d_model, valid_heads)

                # Estimate parameters
                params = self.estimate_params(d_model, num_layers)

                # Compute error
                relative_error = abs(params - target_params) / target_params

                candidates.append({
                    'd_model': d_model,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'estimated_params': params,
                    'relative_error': relative_error,
                    'head_dim': d_model // num_heads
                })

        # Sort by relative error
        candidates.sort(key=lambda x: x['relative_error'])

        # If prefer_depth, among similar errors, prefer more layers
        if prefer_depth and len(candidates) > 0:
            # Group candidates with error < 5%
            good_candidates = [c for c in candidates if c['relative_error'] < 0.05]
            if good_candidates:
                # Sort by num_layers descending
                good_candidates.sort(key=lambda x: -x['num_layers'])
                # Re-combine: good candidates first, then rest
                other_candidates = [c for c in candidates if c['relative_error'] >= 0.05]
                candidates = good_candidates + other_candidates

        return candidates[:10]  # Return top 10

    @staticmethod
    def _choose_num_heads(d_model: int, valid_heads: List[int]) -> int:
        """Choose number of heads that gives reasonable head dimension."""
        # Prefer head dimensions of 64, then 128, then 32
        for target_dim in [64, 128, 32, 96]:
            target_heads = d_model // target_dim
            if target_heads in valid_heads:
                return target_heads

        # Otherwise return middle option
        return valid_heads[len(valid_heads) // 2]

    def select_learning_rate(
        self,
        d_model: int,
        runs: Optional[List[Dict]] = None
    ) -> float:
        """
        Select learning rate based on model size and experimental data.

        Args:
            d_model: Model dimension
            runs: Optional list of previous runs to analyze

        Returns:
            Recommended learning rate
        """
        # If we have experimental data, analyze it
        if runs:
            # Group by d_model, find best learning rate
            lr_by_dmodel = {}
            for run in runs:
                dm = run.get('d_model')
                lr = run.get('learning_rate')
                loss = run.get('loss')

                if dm and lr and loss:
                    if dm not in lr_by_dmodel:
                        lr_by_dmodel[dm] = []
                    lr_by_dmodel[dm].append((lr, loss))

            # Find best LR for each d_model
            best_lrs = {}
            for dm, lr_losses in lr_by_dmodel.items():
                best_lr = min(lr_losses, key=lambda x: x[1])[0]
                best_lrs[dm] = best_lr

            # If we have data for this d_model, use it
            if d_model in best_lrs:
                return best_lrs[d_model]

            # Otherwise, interpolate from nearby d_models
            if best_lrs:
                sorted_dms = sorted(best_lrs.keys())
                # Find closest d_models
                closest = min(sorted_dms, key=lambda x: abs(x - d_model))
                return best_lrs[closest]

        # Default heuristic: larger models need smaller learning rates
        # Use μP-like scaling: lr ~ 1 / sqrt(d_model)
        base_lr = 1e-3
        scaled_lr = base_lr / np.sqrt(d_model / 512)

        # Clip to valid range
        return max(1e-4, min(1e-3, scaled_lr))

    def select_batch_size(
        self,
        model_params: float,
        constraint: List[int] = [128, 256]
    ) -> int:
        """
        Select batch size (must be 128 or 256 per assignment rules).

        Args:
            model_params: Number of model parameters
            constraint: Allowed batch sizes

        Returns:
            Recommended batch size
        """
        # For larger models, prefer larger batch size (better GPU utilization)
        # For smaller models, smaller batch size is fine

        if model_params > 5e8:  # > 500M parameters
            return max(constraint)
        else:
            return min(constraint)

    def create_config(
        self,
        target_params: float,
        runs: Optional[List[Dict]] = None,
        batch_size_constraint: List[int] = [128, 256]
    ) -> Dict:
        """
        Create complete configuration for target model size.

        Args:
            target_params: Target number of parameters
            runs: Optional previous runs for learning rate selection
            batch_size_constraint: Allowed batch sizes

        Returns:
            Complete configuration dictionary
        """
        # Find architecture
        candidates = self.find_architecture(target_params, prefer_depth=True)

        if not candidates:
            raise ValueError(f"Could not find architecture for {target_params} parameters")

        # Select best candidate
        best = candidates[0]

        # Select learning rate
        learning_rate = self.select_learning_rate(best['d_model'], runs)

        # Select batch size
        batch_size = self.select_batch_size(target_params, batch_size_constraint)

        config = {
            'd_model': best['d_model'],
            'num_layers': best['num_layers'],
            'num_heads': best['num_heads'],
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'estimated_params': best['estimated_params'],
            'head_dim': best['head_dim'],
            'target_params': target_params,
            'relative_error': best['relative_error']
        }

        return config

    def print_config_summary(self, config: Dict):
        """Print summary of selected configuration."""
        print("\n" + "=" * 80)
        print("SELECTED HYPERPARAMETERS")
        print("=" * 80)
        print(f"Target Parameters:    {config['target_params']:.2e}")
        print(f"Actual Parameters:    {config['estimated_params']:.2e}")
        print(f"Relative Error:       {config['relative_error']:.2%}")
        print()
        print("Architecture:")
        print(f"  d_model:            {config['d_model']}")
        print(f"  num_layers:         {config['num_layers']}")
        print(f"  num_heads:          {config['num_heads']}")
        print(f"  head_dim:           {config['head_dim']}")
        print()
        print("Training:")
        print(f"  batch_size:         {config['batch_size']}")
        print(f"  learning_rate:      {config['learning_rate']:.6f}")
        print("=" * 80)


def select_hyperparameters_from_prediction(
    prediction: Dict,
    runs: Optional[List[Dict]] = None
) -> Dict:
    """
    Select hyperparameters based on scaling law prediction.

    Args:
        prediction: Prediction dictionary with 'optimal_params' key
        runs: Optional previous runs for learning rate selection

    Returns:
        Complete configuration
    """
    target_params = prediction['optimal_params']

    selector = HyperparameterSelector()
    config = selector.create_config(
        target_params=target_params,
        runs=runs,
        batch_size_constraint=[128, 256]
    )

    selector.print_config_summary(config)

    return config