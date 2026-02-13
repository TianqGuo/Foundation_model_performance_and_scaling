#!/usr/bin/env python3
"""
Generate Submission Summary

Creates a human-readable summary of the final predictions for submission.
"""

import json
import sys
from pathlib import Path


def generate_submission_summary(results_dir: str = "../results/part2_scaling_laws"):
    """Generate submission summary from results."""

    results_path = Path(results_dir)

    # Load scaling law results
    scaling_law_file = results_path / "scaling_law_results.json"
    if not scaling_law_file.exists():
        print(f"ERROR: Scaling law results not found: {scaling_law_file}")
        return

    with open(scaling_law_file, 'r') as f:
        scaling_results = json.load(f)

    # Load final config
    config_file = results_path / "final_config.json"
    if not config_file.exists():
        print(f"ERROR: Final config not found: {config_file}")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract key information
    prediction = scaling_results['prediction']
    power_laws = scaling_results['power_laws']
    fit_quality = scaling_results['fit_quality']

    # Generate summary
    print("\n" + "=" * 80)
    print("SUBMISSION SUMMARY - CS336 Assignment 3: Scaling Laws")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("FITTED SCALING LAWS")
    print("-" * 80)
    print(f"\nModel Size:   {power_laws['model_size']['formula']}")
    print(f"Dataset Size: {power_laws['dataset_size']['formula']}")
    print(f"Loss:         {power_laws['loss']['formula']}")

    print(f"\nFit Quality (R² scores):")
    print(f"  Model Size:   {fit_quality['r2_params']:.4f}")
    print(f"  Dataset Size: {fit_quality['r2_tokens']:.4f}")
    print(f"  Loss:         {fit_quality['r2_losses']:.4f}")

    print("\n" + "-" * 80)
    print("PREDICTIONS FOR TARGET BUDGET (1e19 FLOPs)")
    print("-" * 80)
    print(f"\nOptimal Model Size:   {prediction['optimal_params']:.6e} parameters")
    print(f"                      ≈ {prediction['optimal_params']/1e9:.2f}B parameters")
    print(f"\nOptimal Dataset Size: {prediction['optimal_tokens']:.6e} tokens")
    print(f"                      ≈ {prediction['optimal_tokens']/1e9:.2f}B tokens")
    print(f"\nPredicted Loss:       {prediction['predicted_loss']:.6f}")

    print("\n" + "-" * 80)
    print("SELECTED HYPERPARAMETERS")
    print("-" * 80)
    print(f"\nArchitecture:")
    print(f"  d_model:      {config['d_model']}")
    print(f"  num_layers:   {config['num_layers']}")
    print(f"  num_heads:    {config['num_heads']}")
    print(f"  head_dim:     {config['head_dim']}")
    print(f"\nEstimated Parameters: {config['estimated_params']:.6e}")
    print(f"                      ≈ {config['estimated_params']/1e9:.2f}B parameters")
    print(f"Relative Error:       {config['relative_error']:.2%}")

    print(f"\nTraining:")
    print(f"  batch_size:   {config['batch_size']}")
    print(f"  learning_rate: {config['learning_rate']:.6f}")

    print("\n" + "-" * 80)
    print("GOOGLE FORM SUBMISSION")
    print("-" * 80)
    print("\nCopy the following to the Google form:")
    print("\n1. Predicted optimal model size:")
    print(f"   {config['estimated_params']:.0f}")

    print("\n2. Training hyperparameters:")
    print(f"   d_model: {config['d_model']}")
    print(f"   num_layers: {config['num_layers']}")
    print(f"   num_heads: {config['num_heads']}")
    print(f"   batch_size: {config['batch_size']}")
    print(f"   learning_rate: {config['learning_rate']}")

    print("\n3. Predicted training loss:")
    print(f"   {prediction['predicted_loss']:.6f}")

    print("\n" + "=" * 80)

    # Save summary to text file
    summary_file = results_path / "SUBMISSION_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SUBMISSION SUMMARY - CS336 Assignment 3: Scaling Laws\n")
        f.write("=" * 80 + "\n\n")

        f.write("FITTED SCALING LAWS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Size:   {power_laws['model_size']['formula']}\n")
        f.write(f"Dataset Size: {power_laws['dataset_size']['formula']}\n")
        f.write(f"Loss:         {power_laws['loss']['formula']}\n\n")

        f.write("PREDICTIONS FOR TARGET BUDGET (1e19 FLOPs)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Optimal Model Size:   {prediction['optimal_params']:.6e} parameters\n")
        f.write(f"Optimal Dataset Size: {prediction['optimal_tokens']:.6e} tokens\n")
        f.write(f"Predicted Loss:       {prediction['predicted_loss']:.6f}\n\n")

        f.write("SELECTED HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"d_model:      {config['d_model']}\n")
        f.write(f"num_layers:   {config['num_layers']}\n")
        f.write(f"num_heads:    {config['num_heads']}\n")
        f.write(f"batch_size:   {config['batch_size']}\n")
        f.write(f"learning_rate: {config['learning_rate']}\n")
        f.write(f"Estimated Parameters: {config['estimated_params']:.6e}\n\n")

        f.write("GOOGLE FORM SUBMISSION\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. Model size: {config['estimated_params']:.0f}\n")
        f.write(f"2. Hyperparameters: d_model={config['d_model']}, ")
        f.write(f"num_layers={config['num_layers']}, ")
        f.write(f"num_heads={config['num_heads']}, ")
        f.write(f"batch_size={config['batch_size']}, ")
        f.write(f"learning_rate={config['learning_rate']}\n")
        f.write(f"3. Predicted loss: {prediction['predicted_loss']:.6f}\n")

    print(f"\nSummary saved to: {summary_file}")
    print()


if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results/part2_scaling_laws"
    generate_submission_summary(results_dir)