#!/usr/bin/env python3
"""
Run Experiment with Tracking

This script wraps the existing training loop and adds experiment tracking capabilities.
It integrates ExperimentLogger with cs336_basics/train.py
"""

import argparse
import json
import sys
from pathlib import Path
import time
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.transformer_training.model.transformer_lm import TransformerLM
from cs336_basics.transformer_training.model.cross_entropy import cross_entropy
from cs336_basics.transformer_training.optimizer.adamw import AdamW
from cs336_basics.transformer_training.optimizer.lr_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_training.optimizer.gradient_clipping import clip_gradients
from cs336_basics.data.data_loader import get_batch, load_data_memmap, validate_data
from cs336_basics.transformer_training import save_checkpoint_with_metadata, load_checkpoint

from cs336_basics.basics.experiment_tracker import ExperimentConfig, create_experiment_config
from cs336_basics.basics.experiment_logger import ExperimentLogger


def get_device(device_str=None):
    """Get appropriate device for training."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, eval_iters, device):
    """Evaluate model on validation data."""
    model.eval()
    losses = []

    for _ in range(eval_iters):
        inputs, targets = get_batch(data, batch_size, context_length, str(device))
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    ppl = np.exp(avg_loss)
    return avg_loss, ppl


def train_step(model, optimizer, inputs, targets, grad_clip=None):
    """Perform single training step."""
    model.train()

    # Forward pass
    logits = model(inputs)
    loss = cross_entropy(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    if grad_clip is not None:
        clip_gradients(model.parameters(), grad_clip)

    # Optimizer step
    optimizer.step()

    return loss.item()


def run_experiment(
    experiment_name: str,
    config_dict: dict,
    output_dir: Path,
    resume_from: str = None,
):
    """
    Run training experiment with full tracking.

    Args:
        experiment_name: Name for this experiment
        config_dict: Dictionary with all configuration parameters
        output_dir: Directory to save experiment results
        resume_from: Optional checkpoint to resume from
    """
    # Create experiment configuration
    exp_config = create_experiment_config(
        experiment_name=experiment_name,
        description=config_dict.get('description', ''),
        **{k: v for k, v in config_dict.items() if k != 'description'}
    )

    # Initialize experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=experiment_name,
        config=exp_config,
        output_dir=output_dir,
        resume=resume_from is not None,
    )

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    print(f"Description: {exp_config.description}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Set random seed
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    # Get device
    device = get_device(exp_config.device)
    print(f"Using device: {device}")

    # Load data
    print(f"Loading training data from: {exp_config.train_data_path}")
    train_data = load_data_memmap(exp_config.train_data_path, dtype=np.uint16)
    print(f"Training data loaded: {len(train_data):,} tokens")
    validate_data(train_data, exp_config.vocab_size)

    val_data = None
    if exp_config.val_data_path:
        print(f"Loading validation data from: {exp_config.val_data_path}")
        val_data = load_data_memmap(exp_config.val_data_path, dtype=np.uint16)
        print(f"Validation data loaded: {len(val_data):,} tokens")
        validate_data(val_data, exp_config.vocab_size)

    # Initialize model
    print("\nInitializing model...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(exp_config.dtype, torch.float32)

    model = TransformerLM(
        vocab_size=exp_config.vocab_size,
        context_length=exp_config.context_length,
        d_model=exp_config.d_model,
        num_layers=exp_config.num_layers,
        num_heads=exp_config.num_heads,
        d_ff=exp_config.d_ff,
        rope_theta=exp_config.rope_theta,
        device=device,
        dtype=torch_dtype,
    )

    num_params = count_parameters(model)
    print(f"Model initialized with {num_params:,} trainable parameters")

    # Enable performance optimizations
    if device.type == "cuda":
        print("\n" + "="*80)
        print("ENABLING PERFORMANCE OPTIMIZATIONS")
        print("="*80)

        # Enable TF32 for Ampere+ GPUs (RTX 30/40 series, A100, H100)
        # This gives 2-3x speedup with minimal accuracy impact
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for matmul and cuDNN operations")

        # Compile model for additional speedup
        # Note: Requires Triton to be installed
        print("\nAttempting to compile model with torch.compile()...")
        try:
            # Try to compile - will fail if Triton is not installed
            model = torch.compile(model, mode="default")
            print("✓ Model compiled successfully!")
            print("  Note: First few iterations will be slower during compilation")
        except RuntimeError as e:
            if "triton" in str(e).lower():
                print("⚠️  Triton not installed - skipping compilation")
                print("  To install: pip install triton")
                print("  Training will be slower without compilation (2-3× slower)")
            else:
                print(f"⚠️  Compilation failed: {e}")
                print("  Continuing without compilation")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")
            print("  Continuing without compilation")

        print("="*80 + "\n")
    elif device.type == "mps":
        print("\nUsing MPS device (Apple Silicon)")
        print("Attempting to compile model with aot_eager backend...")
        try:
            model = torch.compile(model, backend="aot_eager")
            print("✓ Model compiled for MPS")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")
            print("  Continuing without compilation")
    elif device.type == "cpu":
        print("\nUsing CPU device")
        print("Attempting to compile model...")
        try:
            model = torch.compile(model)
            print("✓ Model compiled for CPU")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")
            print("  Continuing without compilation")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=exp_config.learning_rate,
        betas=(exp_config.beta1, exp_config.beta2),
        weight_decay=exp_config.weight_decay,
    )

    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        start_iteration = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iteration}")

    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "="*80)
    print("Starting training loop")
    print("="*80)

    best_val_loss = float('inf')

    for iteration in range(start_iteration, exp_config.max_iterations):
        iter_start_time = time.time()

        # Update learning rate
        current_lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=exp_config.learning_rate,
            min_learning_rate=exp_config.min_learning_rate,
            warmup_iters=exp_config.warmup_iters,
            cosine_cycle_iters=exp_config.max_iterations,
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Get training batch
        inputs, targets = get_batch(
            train_data, exp_config.batch_size, exp_config.context_length, str(device)
        )

        # Training step
        train_loss = train_step(model, optimizer, inputs, targets, exp_config.grad_clip)

        iter_time = time.time() - iter_start_time
        tokens_per_sec = (exp_config.batch_size * exp_config.context_length) / iter_time

        # Log training metrics
        if iteration % exp_config.log_interval == 0:
            exp_logger.log_training_step(
                step=iteration,
                train_loss=train_loss,
                learning_rate=current_lr,
                tokens_per_sec=tokens_per_sec,
            )

            print(
                f"Iter {iteration:6d}/{exp_config.max_iterations} | "
                f"Loss: {train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )

        # Periodic validation
        if val_data is not None and (iteration % exp_config.eval_interval == 0 or iteration == exp_config.max_iterations - 1):
            val_loss, val_ppl = evaluate(
                model, val_data, exp_config.batch_size,
                exp_config.context_length, 20, device
            )

            exp_logger.log_validation_step(
                step=iteration,
                val_loss=val_loss,
                val_perplexity=val_ppl,
            )

            print(
                f"\nValidation | Loss: {val_loss:.4f} | "
                f"Perplexity: {val_ppl:.2f}\n"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_dir / "best_model.pt"
                save_checkpoint_with_metadata(
                    model, optimizer, iteration, best_model_path,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                    is_best=True
                )
                print(f"✓ New best model saved")

        # Periodic checkpointing
        if iteration > 0 and iteration % exp_config.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
            save_checkpoint_with_metadata(
                model, optimizer, iteration, checkpoint_path,
                train_loss=train_loss,
                learning_rate=current_lr
            )

    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
    save_checkpoint_with_metadata(
        model, optimizer, exp_config.max_iterations, final_checkpoint_path,
        train_loss=train_loss,
        learning_rate=current_lr
    )

    # Finalize experiment
    exp_logger.finalize()

    print(f"\n✓ Experiment '{experiment_name}' completed successfully!")
    print(f"Results saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment with tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: assignment_experiments/runs/<name>)")
    parser.add_argument("--resume-from", type=str, help="Checkpoint to resume from")
    parser.add_argument("--description", type=str, default="", help="Experiment description")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Add description if provided
    if args.description:
        config['description'] = args.description

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "runs" / args.name

    # Run experiment
    run_experiment(
        experiment_name=args.name,
        config_dict=config,
        output_dir=output_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
