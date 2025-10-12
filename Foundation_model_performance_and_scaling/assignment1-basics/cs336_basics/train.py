#!/usr/bin/env python3
"""
Transformer Language Model Training Script

Complete training loop implementing CS336 Assignment 1 requirements:
- Configurable model and optimizer hyperparameters via CLI
- Memory-efficient loading with np.memmap
- Checkpoint serialization
- Periodic training and validation logging
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# Import custom modules
from cs336_basics.model.transformer_lm import TransformerLM
from cs336_basics.model.cross_entropy import cross_entropy, perplexity
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.optimizer.lr_schedule import get_lr_cosine_schedule
from cs336_basics.optimizer.gradient_clipping import clip_gradients
from cs336_basics.data.data_loader import get_batch, load_data_memmap, validate_data
from cs336_basics.training.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_checkpoint_with_metadata,
)


def setup_logging(log_dir: str, log_to_file: bool = True) -> logging.Logger:
    """
    Set up logging to console and optionally to file.

    Args:
        log_dir: Directory for log files
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("transformer_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device_str: Device string ('cpu', 'cuda', 'mps', or None for auto-detect)

    Returns:
        torch.device instance
    """
    if device_str:
        return torch.device(device_str)

    # Auto-detect best available device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on validation data.

    Args:
        model: The model to evaluate
        data: Validation data array
        batch_size: Batch size for evaluation
        context_length: Context length
        eval_iters: Number of evaluation iterations
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    losses = []
    start_time = time.time()

    for _ in range(eval_iters):
        # Get batch
        inputs, targets = get_batch(data, batch_size, context_length, str(device))

        # Forward pass
        logits = model(inputs)

        # Compute loss
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())

    elapsed_time = time.time() - start_time

    avg_loss = np.mean(losses)
    ppl = np.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "time": elapsed_time,
    }


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    grad_clip: Optional[float] = None,
) -> float:
    """
    Perform a single training step.

    Args:
        model: The model to train
        optimizer: The optimizer
        inputs: Input token IDs
        targets: Target token IDs
        grad_clip: Gradient clipping threshold (None to disable)

    Returns:
        Loss value for this step
    """
    model.train()

    # Forward pass
    logits = model(inputs)

    # Compute loss
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


def train(
    # Model hyperparameters
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float = 10000.0,

    # Training hyperparameters
    batch_size: int = 32,
    max_iterations: int = 10000,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    warmup_iters: int = 100,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    grad_clip: Optional[float] = 1.0,

    # Data paths
    train_data_path: str = None,
    val_data_path: Optional[str] = None,

    # Checkpointing
    checkpoint_dir: str = "./checkpoints",
    checkpoint_interval: int = 1000,
    resume_from: Optional[str] = None,

    # Logging
    log_interval: int = 10,
    eval_interval: int = 100,
    eval_iters: int = 20,

    # System
    device: Optional[str] = None,
    dtype: str = "float32",
    compile_model: bool = False,
    seed: int = 42,
) -> None:
    """
    Main training loop for transformer language model.

    Implements all CS336 requirements:
    - Configurable hyperparameters
    - Memory-efficient data loading with np.memmap
    - Checkpoint serialization
    - Periodic logging of training and validation metrics
    """

    # Set up logging
    logger = setup_logging(checkpoint_dir, log_to_file=True)
    logger.info("=" * 80)
    logger.info("Starting Transformer Language Model Training")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get device
    device_obj = get_device(device)
    logger.info(f"Using device: {device_obj}")

    # Set dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    logger.info(f"Using dtype: {torch_dtype}")

    # Load training data with memory mapping
    logger.info(f"Loading training data from: {train_data_path}")
    train_data = load_data_memmap(train_data_path, dtype=np.uint16)
    logger.info(f"Training data loaded: {len(train_data):,} tokens")
    validate_data(train_data, vocab_size)

    # Load validation data if provided
    val_data = None
    if val_data_path:
        logger.info(f"Loading validation data from: {val_data_path}")
        val_data = load_data_memmap(val_data_path, dtype=np.uint16)
        logger.info(f"Validation data loaded: {len(val_data):,} tokens")
        validate_data(val_data, vocab_size)

    # Log hyperparameters
    logger.info("\nModel Hyperparameters:")
    logger.info(f"  vocab_size: {vocab_size}")
    logger.info(f"  context_length: {context_length}")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  num_layers: {num_layers}")
    logger.info(f"  num_heads: {num_heads}")
    logger.info(f"  d_ff: {d_ff}")
    logger.info(f"  rope_theta: {rope_theta}")

    logger.info("\nTraining Hyperparameters:")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  max_iterations: {max_iterations}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  min_learning_rate: {min_learning_rate}")
    logger.info(f"  warmup_iters: {warmup_iters}")
    logger.info(f"  weight_decay: {weight_decay}")
    logger.info(f"  beta1: {beta1}, beta2: {beta2}")
    logger.info(f"  grad_clip: {grad_clip}")

    # Initialize model
    logger.info("\nInitializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device_obj,
        dtype=torch_dtype,
    )

    num_params = count_parameters(model)
    logger.info(f"Model initialized with {num_params:,} trainable parameters")

    # Compile model if requested (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Initialize optimizer
    logger.info("Initializing optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_iteration = load_checkpoint(resume_from, model, optimizer)
        logger.info(f"Resumed from iteration {start_iteration}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting training loop")
    logger.info("=" * 80)

    best_val_loss = float('inf')
    training_start_time = time.time()

    for iteration in range(start_iteration, max_iterations):
        iter_start_time = time.time()

        # Update learning rate with schedule
        current_lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iterations,
        )

        # Set learning rate for all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Get training batch
        inputs, targets = get_batch(train_data, batch_size, context_length, str(device_obj))

        # Training step
        train_loss = train_step(model, optimizer, inputs, targets, grad_clip)

        iter_time = time.time() - iter_start_time

        # Periodic logging
        if iteration % log_interval == 0 or iteration == max_iterations - 1:
            tokens_per_sec = (batch_size * context_length) / iter_time
            logger.info(
                f"Iter {iteration:6d}/{max_iterations} | "
                f"Loss: {train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {iter_time:.3f}s | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )

        # Periodic validation
        if val_data is not None and (iteration % eval_interval == 0 or iteration == max_iterations - 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating at iteration {iteration}")

            eval_metrics = evaluate(
                model, val_data, batch_size, context_length, eval_iters, device_obj
            )

            logger.info(
                f"Validation | Loss: {eval_metrics['loss']:.4f} | "
                f"Perplexity: {eval_metrics['perplexity']:.2f} | "
                f"Time: {eval_metrics['time']:.2f}s"
            )

            # Save best model
            if eval_metrics['loss'] < best_val_loss:
                best_val_loss = eval_metrics['loss']
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_checkpoint_with_metadata(
                    model, optimizer, iteration, best_model_path,
                    val_loss=eval_metrics['loss'],
                    val_perplexity=eval_metrics['perplexity'],
                    is_best=True
                )
                logger.info(f"✓ New best model saved to {best_model_path}")

            logger.info(f"{'='*60}\n")

        # Periodic checkpointing
        if iteration > 0 and iteration % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint_with_metadata(
                model, optimizer, iteration, checkpoint_path,
                train_loss=train_loss,
                learning_rate=current_lr
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    save_checkpoint_with_metadata(
        model, optimizer, max_iterations, final_checkpoint_path,
        train_loss=train_loss,
        learning_rate=current_lr
    )
    logger.info(f"Final checkpoint saved to {final_checkpoint_path}")

    # Training summary
    total_time = time.time() - training_start_time
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Total training time: {total_time / 3600:.2f} hours")
    logger.info(f"Average time per iteration: {total_time / max_iterations:.3f}s")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    if val_data is not None:
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 80)


def create_sample_config(output_path: str = "sample_config.json") -> None:
    """Create a sample configuration file."""
    config = {
        # Model hyperparameters
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "rope_theta": 10000.0,

        # Training hyperparameters
        "batch_size": 32,
        "max_iterations": 10000,
        "learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "warmup_iters": 100,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "grad_clip": 1.0,

        # Data paths
        "train_data_path": "path/to/train_data.npy",
        "val_data_path": "path/to/val_data.npy",

        # Checkpointing
        "checkpoint_dir": "./checkpoints",
        "checkpoint_interval": 1000,
        "resume_from": None,

        # Logging
        "log_interval": 10,
        "eval_interval": 100,
        "eval_iters": 20,

        # System
        "device": None,  # Auto-detect
        "dtype": "float32",
        "compile_model": False,
        "seed": 42,
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Sample configuration saved to: {output_path}")
    print("\nEdit this file with your settings, then run:")
    print(f"  python train.py --config {output_path}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample config
  python train.py --create_sample_config

  # Train with config file (recommended)
  python train.py --config config.json

  # Train with command-line arguments
  python train.py --train_data_path data/train.npy --vocab_size 10000 --d_model 512

  # Resume from checkpoint
  python train.py --config config.json --resume_from checkpoints/checkpoint_1000.pt
        """
    )

    # Config file option
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--create_sample_config", action="store_true",
                       help="Create a sample configuration file and exit")

    # Model hyperparameters
    model_group = parser.add_argument_group("Model Hyperparameters")
    model_group.add_argument("--vocab_size", type=int, help="Vocabulary size")
    model_group.add_argument("--context_length", type=int, default=512, help="Context length")
    model_group.add_argument("--d_model", type=int, default=512, help="Model dimension")
    model_group.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    model_group.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    model_group.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    model_group.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Training hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_group.add_argument("--max_iterations", type=int, default=10000, help="Maximum iterations")
    train_group.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    train_group.add_argument("--min_learning_rate", type=float, default=3e-5, help="Minimum learning rate")
    train_group.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations")
    train_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    train_group.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    train_group.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    train_group.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")

    # Data paths
    data_group = parser.add_argument_group("Data Paths")
    data_group.add_argument("--train_data_path", type=str, help="Path to training data (.npy)")
    data_group.add_argument("--val_data_path", type=str, help="Path to validation data (.npy)")

    # Checkpointing
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                                 help="Directory for checkpoints")
    checkpoint_group.add_argument("--checkpoint_interval", type=int, default=1000,
                                 help="Save checkpoint every N iterations")
    checkpoint_group.add_argument("--resume_from", type=str, help="Path to checkpoint to resume from")

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
    log_group.add_argument("--eval_interval", type=int, default=100, help="Evaluate every N iterations")
    log_group.add_argument("--eval_iters", type=int, default=20, help="Number of evaluation iterations")

    # System
    system_group = parser.add_argument_group("System")
    system_group.add_argument("--device", type=str, help="Device (cpu, cuda, mps)")
    system_group.add_argument("--dtype", type=str, default="float32",
                             choices=["float32", "float16", "bfloat16"], help="Data type")
    system_group.add_argument("--compile_model", action="store_true", help="Use torch.compile()")
    system_group.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Handle special commands
    if args.create_sample_config:
        create_sample_config()
        return

    # Load config from file if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {args.config}")

    # Override config with command-line arguments (command-line takes precedence)
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'create_sample_config']:
            config[key] = value

    # Validate required arguments
    if "train_data_path" not in config or config["train_data_path"] is None:
        parser.error("--train_data_path is required (or specify in config file)")

    if "vocab_size" not in config or config["vocab_size"] is None:
        parser.error("--vocab_size is required (or specify in config file)")

    # Start training
    train(**config)


if __name__ == "__main__":
    main()
