#!/usr/bin/env python3
"""
Run Scaling Law Experiments (Part 2)

Main script for querying the training API, fitting scaling laws,
and making predictions for the target compute budget.

Usage:
    python run_experiments.py --api-key-file ../api_key.txt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from api_client import TrainingAPIClient, load_api_key
from experiment_design import ExperimentDesigner, create_default_strategy
from scaling_law_fitter import ScalingLawFitter


def run_experiments(
    api_key: str,
    budget: float,
    output_dir: str,
    dry_run: bool = False
) -> List[Dict]:
    """
    Run experiments by querying the API.

    Args:
        api_key: API key for authentication
        budget: Total FLOPs budget
        output_dir: Directory to save results
        dry_run: If True, only print strategy without querying API

    Returns:
        List of completed runs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize API client
    cache_file = output_path / 'run_cache.json'
    client = TrainingAPIClient(api_key, cache_file=str(cache_file))

    # Try to sync with API to get any existing runs
    print("\nChecking for existing runs from API...")
    try:
        client.sync_cache_with_api()
        existing_flops = client.get_total_flops_used()
        print(f"Already used: {existing_flops:.2e} FLOPs")
    except Exception as e:
        print(f"Could not sync with API: {e}")
        existing_flops = 0

    # Create experiment strategy
    print("\n" + "=" * 80)
    print("DESIGNING EXPERIMENT STRATEGY")
    print("=" * 80)

    designer = create_default_strategy(budget=budget)
    designer.print_strategy_summary()

    if dry_run:
        print("\nDRY RUN - Not querying API")
        return []

    # Confirm before running
    summary = designer.get_budget_summary()
    print(f"\nReady to query {summary['num_configs']} configurations")
    print(f"Estimated cost: {summary['estimated_cost']:.2e} FLOPs")

    # Check current usage
    try:
        current_usage = client.get_total_flops_used()
        remaining = budget - current_usage
        print(f"Current usage: {current_usage:.2e} FLOPs")
        print(f"Remaining budget: {remaining:.2e} FLOPs")

        if current_usage >= budget:
            print("\nWARNING: Budget already exceeded!")
            print("Skipping API queries. Using cached runs only.")
            previous_runs = client.get_previous_runs()
            return previous_runs

    except Exception as e:
        print(f"Could not check current usage: {e}")
        current_usage = 0

    # Run experiments
    print("\n" + "=" * 80)
    print("QUERYING TRAINING API")
    print("=" * 80)

    runs = []
    configs = designer.configs

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Querying configuration:")
        print(f"    d_model={config['d_model']}, num_layers={config['num_layers']}, "
              f"num_heads={config['num_heads']}, batch_size={config['batch_size']}")
        print(f"    lr={config['learning_rate']}, train_flops={config['train_flops']:.2e}")
        print(f"    Estimated params: {config['estimated_params']:.2e}")

        try:
            # Query API
            result = client.query_loss(config, use_cache=True)

            # Store result
            run = {**config, 'loss': result['loss']}
            runs.append(run)

            # Print result
            cached_str = " [CACHED]" if result.get('cached', False) else ""
            print(f"    → Loss: {result['loss']:.6f}{cached_str}")
            print(f"    → Total FLOPs used: {result['total_flops_used']:.2e}")

            # Check if we're approaching budget
            if result['total_flops_used'] > budget * 0.9:
                print(f"\nWARNING: Approaching budget limit (90% used)")

            if result['total_flops_used'] >= budget:
                print(f"\nBudget limit reached! Stopping experiments.")
                break

            # Small delay to avoid overwhelming API
            if not result.get('cached', False):
                time.sleep(0.5)

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    print("\n" + "=" * 80)
    print(f"COMPLETED {len(runs)} EXPERIMENTS")
    print("=" * 80)

    # Save runs to file
    runs_file = output_path / 'experimental_runs.json'
    with open(runs_file, 'w') as f:
        json.dump(runs, f, indent=2)
    print(f"\nRuns saved to {runs_file}")

    return runs


def fit_scaling_laws(
    runs: List[Dict],
    target_budget: float,
    output_dir: str
) -> Dict:
    """
    Fit scaling laws to experimental data.

    Args:
        runs: List of training runs
        target_budget: Target compute budget for prediction
        output_dir: Directory to save results

    Returns:
        Prediction dictionary
    """
    print("\n" + "=" * 80)
    print("FITTING SCALING LAWS")
    print("=" * 80)

    if len(runs) == 0:
        print("ERROR: No runs available for fitting!")
        return {}

    # Create fitter
    fitter = ScalingLawFitter(runs)

    # Fit using IsoFLOPs method
    print("\nFitting power laws using IsoFLOPs method...")
    fitter.fit_isoflops()

    # Print summary
    prediction = fitter.print_summary(target_budget=target_budget)

    # Create plots
    print("\nGenerating plots...")
    fitter.plot_scaling_laws(output_dir, target_budget=target_budget)

    # Save results
    output_path = Path(output_dir)
    results = {
        'power_laws': {
            'model_size': {
                'coefficient_a': float(fitter.param_predictor.a),
                'exponent_b': float(fitter.param_predictor.b),
                'formula': fitter.param_predictor.formula
            },
            'dataset_size': {
                'coefficient_a': float(fitter.token_predictor.a),
                'exponent_b': float(fitter.token_predictor.b),
                'formula': fitter.token_predictor.formula
            },
            'loss': {
                'coefficient_a': float(fitter.loss_predictor.a),
                'exponent_b': float(fitter.loss_predictor.b),
                'formula': fitter.loss_predictor.formula
            }
        },
        'fit_quality': fitter.compute_fit_quality(),
        'prediction': prediction,
        'data_points': {
            'compute_budgets': fitter.compute_budgets.tolist(),
            'optimal_params': fitter.optimal_params.tolist(),
            'optimal_tokens': fitter.optimal_tokens.tolist(),
            'optimal_losses': fitter.optimal_losses.tolist()
        }
    }

    results_file = output_path / 'scaling_law_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description='Run scaling law experiments and fit power laws'
    )
    parser.add_argument(
        '--api-key-file',
        type=str,
        default='../api_key.txt',
        help='Path to file containing API key'
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=2e18,
        help='Total FLOPs budget for experiments (default: 2e18)'
    )
    parser.add_argument(
        '--target-budget',
        type=float,
        default=1e19,
        help='Target budget for prediction (default: 1e19)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results/part2_scaling_laws',
        help='Directory to save results'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only print strategy without querying API'
    )
    parser.add_argument(
        '--use-cached-only',
        action='store_true',
        help='Only use cached runs, do not query API'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CS336 ASSIGNMENT 3 - SCALING LAWS (PART 2)")
    print("=" * 80)

    # Load API key
    if not args.dry_run and not args.use_cached_only:
        print(f"\nLoading API key from {args.api_key_file}")
        try:
            api_key = load_api_key(args.api_key_file)
            print("API key loaded successfully")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return
    else:
        api_key = "dummy_key"

    # Run experiments
    if args.use_cached_only:
        print("\nUsing cached runs only...")
        cache_file = Path(args.output_dir) / 'experimental_runs.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                runs = json.load(f)
            print(f"Loaded {len(runs)} cached runs")
        else:
            print(f"ERROR: Cache file not found: {cache_file}")
            return
    else:
        runs = run_experiments(
            api_key=api_key,
            budget=args.budget,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )

    if args.dry_run:
        print("\nDry run complete. No API queries made.")
        return

    # Fit scaling laws
    if len(runs) > 0:
        prediction = fit_scaling_laws(
            runs=runs,
            target_budget=args.target_budget,
            output_dir=args.output_dir
        )

        print("\n" + "=" * 80)
        print("FINAL PREDICTION SUMMARY")
        print("=" * 80)
        print(f"Target Budget:        {args.target_budget:.2e} FLOPs")
        print(f"Optimal Model Size:   {prediction['optimal_params']:.2e} parameters")
        print(f"Optimal Dataset Size: {prediction['optimal_tokens']:.2e} tokens")
        print(f"Predicted Loss:       {prediction['predicted_loss']:.6f}")
        print("=" * 80)
    else:
        print("\nNo runs available. Cannot fit scaling laws.")

    print("\n✓ Done! Check results in:", args.output_dir)


if __name__ == '__main__':
    main()