"""
Experiment Design Strategy

Designs experiment configurations to query within the FLOPs budget.
Uses IsoFLOPs approach with multiple compute budgets and model sizes.
"""

import itertools
from typing import Dict, List, Tuple

import numpy as np


class ExperimentDesigner:
    """Designs experiments for scaling law fitting."""

    # Available compute budgets from API
    AVAILABLE_FLOPS = [
        int(1e13), int(3e13), int(6e13),
        int(1e14), int(3e14), int(6e14),
        int(1e15), int(3e15), int(6e15),
        int(1e16), int(3e16), int(6e16),
        int(1e17), int(3e17), int(6e17),
        int(1e18),
    ]

    def __init__(self, budget: float = 2e18):
        """
        Initialize experiment designer.

        Args:
            budget: Total FLOPs budget for experiments
        """
        self.budget = budget
        self.configs = []
        self.estimated_cost = 0

    @staticmethod
    def estimate_model_size(d_model: int, num_layers: int) -> int:
        """Estimate model parameters."""
        return 12 * num_layers * (d_model ** 2)

    def generate_isoflops_configs(
        self,
        compute_budgets: List[int],
        model_sizes_per_budget: int = 5,
        batch_size: int = 256,
        learning_rate: float = 3e-4
    ) -> List[Dict]:
        """
        Generate IsoFLOPs configurations.

        Strategy: For each compute budget, test multiple model sizes.

        Args:
            compute_budgets: List of compute budgets to test
            model_sizes_per_budget: Number of different model sizes per budget
            batch_size: Batch size to use
            learning_rate: Learning rate to use

        Returns:
            List of configuration dictionaries
        """
        configs = []

        for train_flops in compute_budgets:
            # Generate multiple model sizes for this budget
            # Strategy: Vary d_model and num_layers to get different model sizes

            # Target model sizes (parameters) for this budget
            # Rough heuristic: optimal size scales as ~C^0.5
            base_size = (train_flops ** 0.5) / 100
            size_multipliers = np.logspace(-0.5, 0.5, model_sizes_per_budget)
            target_sizes = [base_size * mult for mult in size_multipliers]

            for target_size in target_sizes:
                # Find a good (d_model, num_layers) combination
                config = self._find_model_config(target_size, batch_size, learning_rate, train_flops)
                if config:
                    configs.append(config)

        return configs

    def _find_model_config(
        self,
        target_params: float,
        batch_size: int,
        learning_rate: float,
        train_flops: int
    ) -> Dict:
        """
        Find model configuration that approximately matches target parameters.

        Args:
            target_params: Target number of parameters
            batch_size: Batch size
            learning_rate: Learning rate
            train_flops: Training FLOPs budget

        Returns:
            Configuration dictionary or None if not found
        """
        # Try different (d_model, num_layers) combinations
        d_models = [64, 128, 192, 256, 384, 512, 640, 768, 896, 1024]
        num_layers_options = list(range(2, 25))

        best_config = None
        best_diff = float('inf')

        for d_model in d_models:
            for num_layers in num_layers_options:
                # Find valid num_heads (must divide d_model)
                valid_heads = [h for h in range(2, 17) if d_model % h == 0]
                if not valid_heads:
                    continue

                # Prefer heads that give reasonable head dimension
                # Head dimension should be 32, 64, or 128 ideally
                num_heads = self._choose_num_heads(d_model, valid_heads)

                # Estimate parameters
                params = self.estimate_model_size(d_model, num_layers)

                # Check if this is closer to target
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = {
                        'd_model': d_model,
                        'num_layers': num_layers,
                        'num_heads': num_heads,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'train_flops': train_flops,
                        'estimated_params': params
                    }

        return best_config

    @staticmethod
    def _choose_num_heads(d_model: int, valid_heads: List[int]) -> int:
        """Choose number of heads that gives reasonable head dimension."""
        # Prefer head dimensions of 64 or 128
        for target_dim in [64, 128, 32]:
            target_heads = d_model // target_dim
            if target_heads in valid_heads:
                return target_heads

        # Otherwise return middle option
        return valid_heads[len(valid_heads) // 2]

    def create_isoflops_strategy(
        self,
        num_budgets: int = 8,
        models_per_budget: int = 6,
        batch_size: int = 256,
        learning_rate: float = 3e-4
    ) -> List[Dict]:
        """
        Create IsoFLOPs experiment strategy.

        Args:
            num_budgets: Number of compute budgets to test
            models_per_budget: Number of model sizes per budget
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            List of configurations
        """
        # Select compute budgets - spread across available range
        # Focus on larger budgets that are closer to our target (1e19)
        # Target is 1e19, so select budgets up to 1e18
        selected_budgets = [
            int(1e15),  # 1e15
            int(3e15),  # 3e15
            int(6e15),  # 6e15
            int(1e16),  # 1e16
            int(3e16),  # 3e16
            int(6e16),  # 6e16
            int(1e17),  # 1e17
            int(3e17),  # 3e17
            int(6e17),  # 6e17
            int(1e18),  # 1e18 (our largest available, 1/10 of target)
        ][:num_budgets]

        configs = self.generate_isoflops_configs(
            compute_budgets=selected_budgets,
            model_sizes_per_budget=models_per_budget,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Estimate total cost
        self.configs = configs
        self.estimated_cost = sum(c['train_flops'] for c in configs)

        return configs

    def add_hyperparameter_sweep(
        self,
        base_config: Dict,
        param_to_sweep: str,
        values: List
    ) -> List[Dict]:
        """
        Add hyperparameter sweep configurations.

        Args:
            base_config: Base configuration
            param_to_sweep: Parameter name to sweep
            values: Values to test

        Returns:
            List of sweep configurations
        """
        sweep_configs = []

        for value in values:
            config = base_config.copy()
            config[param_to_sweep] = value
            sweep_configs.append(config)

        return sweep_configs

    def get_budget_summary(self) -> Dict:
        """Get summary of experiment budget."""
        return {
            'num_configs': len(self.configs),
            'estimated_cost': self.estimated_cost,
            'budget': self.budget,
            'budget_utilization': self.estimated_cost / self.budget,
            'remaining_budget': self.budget - self.estimated_cost
        }

    def print_strategy_summary(self):
        """Print summary of experiment strategy."""
        summary = self.get_budget_summary()

        print("\n" + "=" * 80)
        print("EXPERIMENT STRATEGY SUMMARY")
        print("=" * 80)
        print(f"Total FLOPs Budget:     {self.budget:.2e}")
        print(f"Estimated Cost:         {summary['estimated_cost']:.2e}")
        print(f"Budget Utilization:     {summary['budget_utilization']:.1%}")
        print(f"Remaining Budget:       {summary['remaining_budget']:.2e}")
        print(f"Number of Configs:      {summary['num_configs']}")
        print("=" * 80)

        # Group by compute budget
        budget_groups = {}
        for config in self.configs:
            budget = config['train_flops']
            if budget not in budget_groups:
                budget_groups[budget] = []
            budget_groups[budget].append(config)

        print("\nConfigurations by Compute Budget:")
        print("-" * 80)
        for budget in sorted(budget_groups.keys()):
            configs = budget_groups[budget]
            print(f"\nC = {budget:.2e} FLOPs ({len(configs)} models):")
            for i, cfg in enumerate(configs, 1):
                params = cfg['estimated_params']
                print(f"    {i}. d_model={cfg['d_model']}, "
                      f"num_layers={cfg['num_layers']}, "
                      f"num_heads={cfg['num_heads']}, "
                      f"params={params:.2e}")
        print("=" * 80)


def create_default_strategy(budget: float = 2e18) -> ExperimentDesigner:
    """
    Create default experiment strategy.

    Strategy:
    - Test 8 compute budgets from 1e15 to 1e18
    - 6 model sizes per budget (varied via d_model and num_layers)
    - Use standard batch_size=256, learning_rate=3e-4

    Total: ~48 configurations, but actual cost depends on flops per config.
    """
    designer = ExperimentDesigner(budget=budget)

    # Create IsoFLOPs strategy
    configs = designer.create_isoflops_strategy(
        num_budgets=8,
        models_per_budget=6,
        batch_size=256,
        learning_rate=3e-4
    )

    return designer