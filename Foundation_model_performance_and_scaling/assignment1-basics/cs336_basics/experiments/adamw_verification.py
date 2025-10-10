#!/usr/bin/env python3
"""
AdamW Verification Test
Quick test to verify our AdamW implementation works correctly
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to sys.path to import optimizer
sys.path.append(str(Path(__file__).parent.parent))
from optimizer.adamw import AdamW


def test_adamw_basic():
    """Test basic AdamW functionality."""
    print("=== AdamW Basic Functionality Test ===")

    # Create a simple model
    torch.manual_seed(42)
    model = nn.Linear(10, 1, bias=False)

    # Create AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    print(f"Initial weight norm: {model.weight.norm().item():.6f}")

    # Simple optimization loop
    for step in range(5):
        # Generate dummy data
        x = torch.randn(32, 10)
        target = torch.randn(32, 1)

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}: Loss = {loss.item():.6f}, Weight norm = {model.weight.norm().item():.6f}")

    print("✓ AdamW optimization completed successfully!")


def test_adamw_vs_pytorch():
    """Compare our AdamW with PyTorch's AdamW on the same problem."""
    print("\n=== AdamW vs PyTorch AdamW Comparison ===")

    # Set up identical problems
    torch.manual_seed(123)
    model1 = nn.Linear(5, 2, bias=False)
    model2 = nn.Linear(5, 2, bias=False)

    # Copy weights to ensure identical starting points
    with torch.no_grad():
        model2.weight.copy_(model1.weight)

    # Create optimizers
    our_optimizer = AdamW(
        model1.parameters(),
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )

    pytorch_optimizer = torch.optim.AdamW(
        model2.parameters(),
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )

    print("Comparing optimization trajectories...")

    # Run identical optimization steps
    for step in range(3):
        # Same random data for both
        torch.manual_seed(456 + step)
        x = torch.randn(16, 5)
        target = torch.randn(16, 2)

        # Our implementation
        output1 = model1(x)
        loss1 = nn.MSELoss()(output1, target)
        our_optimizer.zero_grad()
        loss1.backward()
        our_optimizer.step()

        # PyTorch implementation
        torch.manual_seed(456 + step)  # Reset for identical data
        x = torch.randn(16, 5)
        target = torch.randn(16, 2)
        output2 = model2(x)
        loss2 = nn.MSELoss()(output2, target)
        pytorch_optimizer.zero_grad()
        loss2.backward()
        pytorch_optimizer.step()

        # Compare weights
        weight_diff = (model1.weight - model2.weight).abs().max().item()
        print(f"Step {step + 1}: Weight difference = {weight_diff:.8f}")

        if weight_diff < 1e-6:
            print(f"  ✓ Weights are very similar (diff < 1e-6)")
        elif weight_diff < 1e-4:
            print(f"  ✓ Weights are reasonably similar (diff < 1e-4)")
        else:
            print(f"  ! Weights differ significantly (diff >= 1e-4)")

    print("✓ Comparison completed!")


def test_adamw_parameters():
    """Test AdamW with different parameter configurations."""
    print("\n=== AdamW Parameter Configuration Test ===")

    model = nn.Linear(3, 1, bias=False)

    # Test different configurations
    configs = [
        {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01, "name": "Standard"},
        {"lr": 1e-2, "betas": (0.9, 0.95), "weight_decay": 0.1, "name": "Large LM"},
        {"lr": 1e-4, "betas": (0.8, 0.999), "weight_decay": 0.001, "name": "Conservative"},
    ]

    for config in configs:
        name = config.pop("name")
        print(f"\nTesting {name} configuration: {config}")

        try:
            optimizer = AdamW(model.parameters(), **config)
            print(f"  ✓ {name} configuration created successfully")

            # Quick optimization step
            x = torch.randn(8, 3)
            target = torch.randn(8, 1)
            output = model(x)
            loss = nn.MSELoss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  ✓ {name} optimization step completed")

        except Exception as e:
            print(f"  ✗ {name} configuration failed: {e}")


if __name__ == "__main__":
    test_adamw_basic()
    test_adamw_vs_pytorch()
    test_adamw_parameters()
    print("\n=== All AdamW Tests Completed ===")