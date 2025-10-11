#!/usr/bin/env python3
"""
Learning Rate Schedule and Gradient Clipping Demonstration
Shows practical usage of cosine annealing with warmup and gradient clipping
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to sys.path to import optimizer components
sys.path.append(str(Path(__file__).parent.parent))
from optimizer.lr_schedule import get_lr_cosine_schedule
from optimizer.gradient_clipping import clip_gradients


def demonstrate_lr_schedule():
    """Demonstrate the cosine annealing learning rate schedule."""
    print("=== Cosine Learning Rate Schedule Demonstration ===")

    # Schedule parameters
    max_lr = 1.0
    min_lr = 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    print(f"Schedule Parameters:")
    print(f"  Max learning rate: {max_lr}")
    print(f"  Min learning rate: {min_lr}")
    print(f"  Warmup iterations: {warmup_iters}")
    print(f"  Cosine cycle iterations: {cosine_cycle_iters}")

    print(f"\nLearning Rate Schedule:")
    print(f"{'Iteration':<10} {'Learning Rate':<15} {'Phase'}")
    print("-" * 40)

    learning_rates = []
    for it in range(25):
        lr = get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        learning_rates.append(lr)

        # Determine phase
        if it < warmup_iters:
            phase = "Warmup"
        elif it <= cosine_cycle_iters:
            phase = "Cosine"
        else:
            phase = "Post-annealing"

        print(f"{it:<10} {lr:<15.6f} {phase}")

    # Verify against expected values from test
    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    print(f"\nVerification against expected values:")
    all_close = np.allclose(learning_rates, expected_lrs, atol=1e-10)
    print(f"All values match expected: {all_close}")

    if not all_close:
        print("Differences found:")
        for i, (actual, expected) in enumerate(zip(learning_rates, expected_lrs)):
            if abs(actual - expected) > 1e-10:
                print(f"  Iteration {i}: actual={actual:.10f}, expected={expected:.10f}")

    return learning_rates


def demonstrate_gradient_clipping():
    """Demonstrate gradient clipping functionality."""
    print("\n" + "=" * 60)
    print("GRADIENT CLIPPING DEMONSTRATION")
    print("=" * 60)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create some dummy data and compute loss
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, y)

    # Backward pass to generate gradients
    loss.backward()

    # Compute gradient norms before clipping
    total_norm_before = 0.0
    grad_norms_before = []
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm().item()
            grad_norms_before.append(param_norm)
            total_norm_before += param_norm ** 2
    total_norm_before = total_norm_before ** 0.5

    print(f"\nBefore gradient clipping:")
    print(f"  Total gradient norm: {total_norm_before:.6f}")
    print(f"  Individual parameter norms: {[f'{norm:.4f}' for norm in grad_norms_before]}")

    # Test different clipping thresholds
    max_norms = [0.5, 1.0, 2.0, 10.0]

    for max_norm in max_norms:
        # Create fresh computation for each test
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        print(f"\nClipping with max_norm = {max_norm}:")

        # Apply gradient clipping
        clip_gradients(model.parameters(), max_norm)

        # Compute gradient norms after clipping
        total_norm_after = 0.0
        grad_norms_after = []
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm().item()
                grad_norms_after.append(param_norm)
                total_norm_after += param_norm ** 2
        total_norm_after = total_norm_after ** 0.5

        print(f"  Total gradient norm after: {total_norm_after:.6f}")
        print(f"  Individual parameter norms: {[f'{norm:.4f}' for norm in grad_norms_after]}")

        # Check if clipping was applied
        was_clipped = total_norm_before > max_norm
        print(f"  Was clipped: {was_clipped}")

        if was_clipped:
            expected_norm = min(max_norm, total_norm_before)
            print(f"  Expected norm: ≈{expected_norm:.6f}")

    # Demonstrate edge cases
    print(f"\n" + "=" * 40)
    print("EDGE CASES")
    print("=" * 40)

    # Test with no gradients
    model.zero_grad()
    print(f"\nTesting with no gradients:")
    try:
        clip_gradients(model.parameters(), 1.0)
        print("  ✓ Handles empty gradients correctly")
    except Exception as e:
        print(f"  ✗ Error with empty gradients: {e}")

    # Test with very small threshold
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    print(f"\nTesting with very small threshold (0.001):")
    clip_gradients(model.parameters(), 0.001)
    final_norm = sum(p.grad.data.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  Final norm: {final_norm:.6f}")
    print(f"  Should be ≈0.001: {abs(final_norm - 0.001) < 1e-4}")


def training_simulation():
    """Simulate a training loop using both learning rate scheduling and gradient clipping."""
    print("\n" + "=" * 60)
    print("TRAINING SIMULATION")
    print("=" * 60)

    # Create model
    model = nn.Linear(20, 1)
    criterion = nn.MSELoss()

    # Scheduler parameters
    max_lr = 0.01
    min_lr = 0.001
    warmup_iters = 5
    cosine_cycle_iters = 20
    max_grad_norm = 1.0

    print(f"Training configuration:")
    print(f"  Max LR: {max_lr}, Min LR: {min_lr}")
    print(f"  Warmup: {warmup_iters} iters, Cosine cycle: {cosine_cycle_iters} iters")
    print(f"  Max gradient norm: {max_grad_norm}")

    print(f"\n{'Step':<5} {'LR':<10} {'Loss':<10} {'Grad Norm':<12} {'Clipped'}")
    print("-" * 50)

    for step in range(25):
        # Get learning rate for this step
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_iters, cosine_cycle_iters)

        # Generate random data
        x = torch.randn(32, 20)
        y = torch.randn(32, 1)

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Check gradient norm before clipping
        grad_norm_before = sum(p.grad.data.norm() ** 2 for p in model.parameters()) ** 0.5

        # Apply gradient clipping
        clip_gradients(model.parameters(), max_grad_norm)

        # Check if clipping was applied
        grad_norm_after = sum(p.grad.data.norm() ** 2 for p in model.parameters()) ** 0.5
        was_clipped = grad_norm_before > max_grad_norm

        # Simulate optimizer step (manual SGD)
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad.data

        print(f"{step:<5} {lr:<10.6f} {loss.item():<10.4f} {grad_norm_after:<12.6f} {'Yes' if was_clipped else 'No'}")


def main():
    """Run all demonstrations."""
    print("Learning Rate Schedule and Gradient Clipping Demonstrations")
    print("Following CS336 Assignment Requirements")

    # Demonstrate learning rate schedule
    learning_rates = demonstrate_lr_schedule()

    # Demonstrate gradient clipping
    demonstrate_gradient_clipping()

    # Show them working together in training
    training_simulation()

    print(f"\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()