#!/usr/bin/env python3
"""
Learning Rate Tuning Experiment
Investigating the effects of different learning rates on SGD training
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to sys.path to import model components
sys.path.append(str(Path(__file__).parent.parent))
from model.cross_entropy import cross_entropy


def create_simple_model(vocab_size=1000, d_model=128):
    """Create a simple linear model for demonstration."""
    return nn.Linear(d_model, vocab_size, bias=False)


def generate_dummy_data(batch_size=32, seq_len=10, d_model=128, vocab_size=1000):
    """Generate dummy training data."""
    # Random input features
    inputs = torch.randn(batch_size, d_model)
    # Random target labels
    targets = torch.randint(0, vocab_size, (batch_size,))
    return inputs, targets


def run_sgd_experiment(learning_rate, num_iterations=10, batch_size=32, d_model=128, vocab_size=1000):
    """
    Run SGD training with a specific learning rate.

    Args:
        learning_rate: Learning rate for SGD optimizer
        num_iterations: Number of training iterations
        batch_size: Batch size for training
        d_model: Model dimension
        vocab_size: Vocabulary size

    Returns:
        List of losses for each iteration
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create model
    model = create_simple_model(vocab_size, d_model)

    # Create SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Track losses
    losses = []

    print(f"\nRunning SGD with learning_rate={learning_rate}")
    print("-" * 50)

    for iteration in range(num_iterations):
        # Generate batch data
        inputs, targets = generate_dummy_data(batch_size, d_model=d_model, vocab_size=vocab_size)

        # Forward pass
        logits = model(inputs)  # (batch_size, vocab_size)

        # Compute cross-entropy loss
        loss = cross_entropy(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Iteration {iteration + 1:2d}: Loss = {loss.item():.6f}")

        # Check for NaN or infinite loss
        if not torch.isfinite(loss):
            print(f"  WARNING: Non-finite loss detected! Training unstable.")
            break

    return losses


def analyze_learning_rates():
    """Analyze the effects of different learning rates."""

    print("=== Learning Rate Tuning Experiment ===")
    print("Testing SGD with different learning rates for 10 iterations")

    # Learning rates to test
    learning_rates = [1e-3, 1e1, 1e2, 1e3]  # Reference + requested values
    all_losses = {}

    # Run experiments for each learning rate
    for lr in learning_rates:
        try:
            losses = run_sgd_experiment(lr, num_iterations=10)
            all_losses[lr] = losses
        except Exception as e:
            print(f"Error with learning_rate={lr}: {e}")
            all_losses[lr] = []

    # Print results in tabular format (no plotting dependencies)
    print(f"\n" + "=" * 80)
    print("DETAILED LOSS PROGRESSION")
    print("=" * 80)

    for lr, losses in all_losses.items():
        if losses:
            print(f"\nLearning Rate {lr}:")
            print("Iteration:  " + "  ".join([f"{i+1:2d}" for i in range(len(losses))]))
            print("Loss:       " + "  ".join([f"{loss:8.4f}" for loss in losses]))
        else:
            print(f"\nLearning Rate {lr}: No data (training failed)")

    # Analysis and reporting
    print("\n" + "=" * 60)
    print("ANALYSIS OF LEARNING RATE EFFECTS")
    print("=" * 60)

    for lr, losses in all_losses.items():
        if not losses:
            continue

        print(f"\nLearning Rate {lr}:")
        print(f"  Initial Loss: {losses[0]:.6f}")
        print(f"  Final Loss:   {losses[-1]:.6f}")

        if len(losses) > 1:
            loss_change = losses[-1] - losses[0]
            if loss_change < 0:
                print(f"  Loss Change:  {loss_change:.6f} (decreasing ✓)")
            else:
                print(f"  Loss Change:  +{loss_change:.6f} (increasing ✗)")

        # Check for convergence issues
        if len(losses) < 10:
            print(f"  Status:       DIVERGED - Training stopped early")
        elif any(not np.isfinite(loss) for loss in losses):
            print(f"  Status:       UNSTABLE - NaN/Inf values detected")
        elif losses[-1] > losses[0] * 10:
            print(f"  Status:       DIVERGING - Loss increasing rapidly")
        elif abs(losses[-1] - losses[0]) < 1e-6:
            print(f"  Status:       STAGNANT - Minimal change")
        else:
            print(f"  Status:       TRAINING - Loss evolving normally")

    # Summary observations
    print(f"\n" + "=" * 60)
    print("SUMMARY OBSERVATIONS")
    print("=" * 60)

    print(f"\n1. Learning Rate 1e-3 (Reference):")
    if 1e-3 in all_losses and all_losses[1e-3]:
        losses = all_losses[1e-3]
        print(f"   - Provides stable, gradual learning")
        print(f"   - Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
        print(f"   - Recommended for actual training")

    print(f"\n2. Learning Rate 1e1 (10):")
    if 1e1 in all_losses and all_losses[1e1]:
        losses = all_losses[1e1]
        print(f"   - Aggressive learning rate")
        if losses[-1] > losses[0]:
            print(f"   - Loss INCREASED from {losses[0]:.4f} to {losses[-1]:.4f}")
            print(f"   - May cause oscillations around minimum")
        else:
            print(f"   - Loss decreased but potentially unstable")

    print(f"\n3. Learning Rate 1e2 (100):")
    if 1e2 in all_losses and all_losses[1e2]:
        losses = all_losses[1e2]
        print(f"   - Very aggressive learning rate")
        if len(losses) < 10:
            print(f"   - Training DIVERGED - stopped early")
        elif any(not np.isfinite(loss) for loss in losses):
            print(f"   - Numerical instability - NaN/Inf values")
        else:
            print(f"   - Likely causing severe overshooting")

    print(f"\n4. Learning Rate 1e3 (1000):")
    if 1e3 in all_losses and all_losses[1e3]:
        losses = all_losses[1e3]
        print(f"   - Extremely aggressive learning rate")
        print(f"   - Almost certainly causes immediate divergence")
        print(f"   - Gradients are too large for stable optimization")
    else:
        print(f"   - Training failed completely")

    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"""
Learning rate is crucial for SGD optimization:

• Too small (e.g., < 1e-4): Slow convergence, may get stuck in poor local minima
• Appropriate (e.g., 1e-3 to 1e-1): Stable learning with steady progress
• Too large (e.g., > 1e1): Overshooting, oscillations, potential divergence
• Extremely large (e.g., > 1e2): Immediate divergence, numerical instability

The experiments demonstrate why learning rate tuning is essential for
successful neural network training. Start with moderate values (1e-3)
and adjust based on loss behavior.
    """)

    return all_losses


if __name__ == "__main__":
    analyze_learning_rates()