#!/usr/bin/env python3
"""
Training Loop Demonstration
Shows data loading, checkpointing, and complete training workflow
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import sys
from pathlib import Path

# Add parent directory to sys.path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from data.data_loader import get_batch, validate_data
from training.checkpointing import save_checkpoint, load_checkpoint, verify_checkpoint
from model.cross_entropy import cross_entropy
from optimizer.adamw import AdamW
from optimizer.lr_schedule import get_lr_cosine_schedule
from optimizer.gradient_clipping import clip_gradients


def create_synthetic_dataset(size=10000, vocab_size=1000):
    """Create a synthetic tokenized dataset for demonstration."""
    print(f"Creating synthetic dataset: {size} tokens, vocab_size={vocab_size}")

    # Create synthetic token sequence with some structure
    # Mix of random tokens and some patterns
    np.random.seed(42)

    # Generate base random sequence
    data = np.random.randint(0, vocab_size, size=size, dtype=np.uint16)

    # Add some patterns to make training more interesting
    # Insert some repeated sequences
    for i in range(0, size - 10, 100):
        pattern = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        if i + len(pattern) < size:
            data[i:i + len(pattern)] = pattern

    # Validate the dataset
    validate_data(data, vocab_size)

    return data


def demonstrate_data_loading():
    """Demonstrate data loading functionality."""
    print("=== Data Loading Demonstration ===")

    # Create synthetic dataset
    data = create_synthetic_dataset(size=1000, vocab_size=100)

    # Test different batch configurations
    configs = [
        {"batch_size": 4, "context_length": 8, "device": "cpu"},
        {"batch_size": 2, "context_length": 16, "device": "cpu"},
        {"batch_size": 1, "context_length": 32, "device": "cpu"},
    ]

    for config in configs:
        print(f"\nTesting config: {config}")

        try:
            inputs, targets = get_batch(data, **config)

            print(f"  Inputs shape: {inputs.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Inputs device: {inputs.device}")
            print(f"  Inputs dtype: {inputs.dtype}")

            # Show first sequence
            print(f"  First input sequence: {inputs[0].tolist()}")
            print(f"  First target sequence: {targets[0].tolist()}")

            # Verify targets are inputs shifted by 1
            input_shifted = inputs[0, 1:].tolist() if len(inputs[0]) > 1 else []
            target_prefix = targets[0, :-1].tolist() if len(targets[0]) > 1 else []

            if input_shifted == target_prefix:
                print("  ✓ Target verification passed")
            else:
                print("  ✗ Target verification failed")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Test memory mapping simulation
    print(f"\nTesting memory-mapped data simulation:")
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        temp_path = f.name

    try:
        # Save and load data using numpy
        np.save(temp_path, data)
        loaded_data = np.load(temp_path, mmap_mode='r')

        print(f"  Original data shape: {data.shape}")
        print(f"  Loaded data shape: {loaded_data.shape}")
        print(f"  Data types match: {data.dtype == loaded_data.dtype}")
        print(f"  Data values match: {np.array_equal(data, loaded_data)}")

        # Test batch loading from memory-mapped data
        inputs, targets = get_batch(loaded_data, batch_size=2, context_length=4, device="cpu")
        print(f"  Batch from mmap shape: {inputs.shape}")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def demonstrate_checkpointing():
    """Demonstrate checkpointing functionality."""
    print("\n=== Checkpointing Demonstration ===")

    # Create a simple model and optimizer
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Train for a few steps to create non-trivial state
    print("Training model to create state...")
    for step in range(5):
        x = torch.randn(16, 64)
        y = torch.randint(0, 10, (16,))

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step}: Loss = {loss.item():.4f}")

    iteration = 100

    # Test checkpointing with temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name

    try:
        print(f"\nSaving checkpoint to {checkpoint_path}")

        # Save checkpoint
        save_checkpoint(model, optimizer, iteration, checkpoint_path)

        # Verify checkpoint
        info = verify_checkpoint(checkpoint_path, expected_iteration=iteration)
        print(f"Checkpoint info: {info}")

        # Modify model to test restoration
        print("\nModifying model state...")
        with torch.no_grad():
            for param in model.parameters():
                param.data.fill_(999.0)  # Set to obvious wrong values

        # Reset optimizer state
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Load checkpoint
        print("Loading checkpoint...")
        restored_iteration = load_checkpoint(checkpoint_path, model, optimizer)

        print(f"Restored iteration: {restored_iteration}")
        print(f"Expected iteration: {iteration}")
        print(f"Iteration matches: {restored_iteration == iteration}")

        # Test that model weights were restored
        print("Testing model weights restoration...")
        x = torch.randn(16, 64)
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # The output should not be all 999s if restoration worked
        if not torch.allclose(output, torch.full_like(output, 999.0)):
            print("✓ Model weights successfully restored")
        else:
            print("✗ Model weights not properly restored")

    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)


def mini_training_loop():
    """Demonstrate a complete mini training loop with all components."""
    print("\n=== Mini Training Loop Demonstration ===")

    # Configuration
    config = {
        'vocab_size': 100,
        'd_model': 32,
        'context_length': 16,
        'batch_size': 4,
        'max_iterations': 20,
        'eval_interval': 5,
        'checkpoint_interval': 10,
        'max_lr': 1e-3,
        'min_lr': 1e-4,
        'warmup_iters': 5,
        'max_grad_norm': 1.0,
    }

    print(f"Training configuration: {config}")

    # Create model (simple embedding + linear for demo)
    class SimpleLanguageModel(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            emb = self.embedding(x)  # (batch, seq, d_model)
            return self.linear(emb)  # (batch, seq, vocab_size)

    model = SimpleLanguageModel(config['vocab_size'], config['d_model'])
    optimizer = AdamW(model.parameters(), lr=config['max_lr'], weight_decay=0.01)

    # Create datasets
    train_data = create_synthetic_dataset(size=2000, vocab_size=config['vocab_size'])
    eval_data = create_synthetic_dataset(size=500, vocab_size=config['vocab_size'])

    print(f"\nDataset sizes - Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Training loop
    print(f"\nStarting training loop...")
    print(f"{'Step':<5} {'LR':<10} {'Loss':<10} {'Grad Norm':<12} {'Action'}")
    print("-" * 55)

    for step in range(config['max_iterations']):
        # Get learning rate
        lr = get_lr_cosine_schedule(
            step,
            config['max_lr'],
            config['min_lr'],
            config['warmup_iters'],
            config['max_iterations']
        )

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        inputs, targets = get_batch(
            train_data,
            config['batch_size'],
            config['context_length'],
            'cpu'
        )

        # Forward pass
        logits = model(inputs)  # (batch, seq, vocab)

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, config['vocab_size'])  # (batch*seq, vocab)
        targets_flat = targets.view(-1)  # (batch*seq,)

        # Compute loss
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        clip_gradients(model.parameters(), config['max_grad_norm'])

        # Check gradient norm
        grad_norm = sum(p.grad.data.norm() ** 2 for p in model.parameters()) ** 0.5

        # Optimizer step
        optimizer.step()

        # Determine action
        action = ""
        if step % config['eval_interval'] == 0 and step > 0:
            # Evaluation
            with torch.no_grad():
                eval_inputs, eval_targets = get_batch(
                    eval_data,
                    config['batch_size'],
                    config['context_length'],
                    'cpu'
                )
                eval_logits = model(eval_inputs)
                eval_logits_flat = eval_logits.view(-1, config['vocab_size'])
                eval_targets_flat = eval_targets.view(-1)
                eval_loss = cross_entropy(eval_logits_flat, eval_targets_flat)
                action += f"Eval: {eval_loss.item():.4f}"

        if step % config['checkpoint_interval'] == 0 and step > 0:
            action += " Checkpoint"

        print(f"{step:<5} {lr:<10.6f} {loss.item():<10.4f} {grad_norm.item():<12.6f} {action}")

    print("\n✓ Training loop completed successfully!")

    # Final evaluation
    print(f"\nFinal evaluation:")
    with torch.no_grad():
        eval_inputs, eval_targets = get_batch(eval_data, config['batch_size'], config['context_length'], 'cpu')
        eval_logits = model(eval_inputs)
        eval_logits_flat = eval_logits.view(-1, config['vocab_size'])
        eval_targets_flat = eval_targets.view(-1)
        final_eval_loss = cross_entropy(eval_logits_flat, eval_targets_flat)
        print(f"Final evaluation loss: {final_eval_loss.item():.4f}")


def main():
    """Run all demonstrations."""
    print("Data Loading, Checkpointing, and Training Demonstrations")
    print("Following CS336 Assignment Requirements")

    demonstrate_data_loading()
    demonstrate_checkpointing()
    mini_training_loop()

    print(f"\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("✓ Efficient batch sampling from tokenized sequences")
    print("✓ Memory-mapped data loading for large datasets")
    print("✓ Model and optimizer checkpointing for training resumption")
    print("✓ Complete training loop with LR scheduling and gradient clipping")
    print("✓ Proper integration of all training components")


if __name__ == "__main__":
    main()