"""
Model checkpointing utilities for training resumption and model persistence.

Implements save/load functionality for model weights, optimizer state, and training metadata.
"""

import torch
import torch.nn as nn
from typing import Union, BinaryIO, IO
import os


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
) -> None:
    """
    Save a training checkpoint containing model, optimizer, and iteration state.

    The checkpoint contains all information needed to resume training from the exact
    same state, including:
    - Model weights and buffers
    - Optimizer state (momentum, running averages, etc.)
    - Current iteration number

    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer to save (with its internal state)
        iteration: Current training iteration number
        out: Output destination - either a file path or file-like object

    Example:
        model = TransformerLM(...)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        iteration = 1000

        # Save to file path
        save_checkpoint(model, optimizer, iteration, 'checkpoint_1000.pt')

        # Save to file object
        with open('checkpoint.pt', 'wb') as f:
            save_checkpoint(model, optimizer, iteration, f)
    """
    # Create checkpoint dictionary containing all necessary state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'model_class': model.__class__.__name__,  # Helpful for debugging
    }

    # Save checkpoint using PyTorch's built-in serialization
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load a training checkpoint and restore model/optimizer state.

    Restores the model and optimizer to their exact state when the checkpoint
    was saved, allowing training to resume seamlessly.

    Args:
        src: Source to load from - either a file path or file-like object
        model: PyTorch model to restore (must have same architecture as saved model)
        optimizer: PyTorch optimizer to restore (must be same type as saved optimizer)

    Returns:
        The iteration number that was saved in the checkpoint

    Example:
        model = TransformerLM(...)  # Same architecture as saved model
        optimizer = AdamW(model.parameters(), lr=1e-3)  # Same optimizer type

        # Load from file path
        iteration = load_checkpoint('checkpoint_1000.pt', model, optimizer)

        # Load from file object
        with open('checkpoint.pt', 'rb') as f:
            iteration = load_checkpoint(f, model, optimizer)

        # Continue training from iteration + 1
        for step in range(iteration + 1, max_iterations):
            # ... training loop continues
    """
    # Load checkpoint dictionary
    checkpoint = torch.load(src, map_location='cpu')  # Load to CPU first for device flexibility

    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract and return iteration number
    iteration = checkpoint['iteration']

    return iteration


def save_checkpoint_with_metadata(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    **metadata
) -> None:
    """
    Extended checkpoint saving with additional metadata.

    Allows saving additional training information like loss values, learning rates,
    model configuration, etc.

    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer to save
        iteration: Current training iteration number
        out: Output destination
        **metadata: Additional key-value pairs to save in checkpoint

    Example:
        save_checkpoint_with_metadata(
            model, optimizer, iteration, 'checkpoint.pt',
            loss=2.4, learning_rate=1e-4, epoch=10,
            config={'d_model': 512, 'num_layers': 6}
        )
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'model_class': model.__class__.__name__,
        **metadata  # Include any additional metadata
    }

    torch.save(checkpoint, out)


def verify_checkpoint(
    checkpoint_path: Union[str, os.PathLike],
    expected_iteration: int = None
) -> dict:
    """
    Verify checkpoint integrity and return metadata.

    Useful for debugging checkpoint issues or inspecting checkpoint contents
    without fully loading into model/optimizer.

    Args:
        checkpoint_path: Path to checkpoint file
        expected_iteration: Optional iteration number to verify against

    Returns:
        Dictionary with checkpoint metadata

    Example:
        info = verify_checkpoint('checkpoint_1000.pt', expected_iteration=1000)
        print(f"Checkpoint contains iteration {info['iteration']}")
        print(f"Model class: {info['model_class']}")
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        metadata = {
            'iteration': checkpoint.get('iteration', 'Unknown'),
            'model_class': checkpoint.get('model_class', 'Unknown'),
            'has_model_state': 'model_state_dict' in checkpoint,
            'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
        }

        # Add any additional keys found in checkpoint
        additional_keys = set(checkpoint.keys()) - {
            'model_state_dict', 'optimizer_state_dict', 'iteration', 'model_class'
        }
        if additional_keys:
            metadata['additional_keys'] = list(additional_keys)

        # Verify expected iteration if provided
        if expected_iteration is not None:
            actual_iteration = checkpoint.get('iteration')
            if actual_iteration != expected_iteration:
                metadata['iteration_mismatch'] = True
                metadata['expected_iteration'] = expected_iteration
                metadata['actual_iteration'] = actual_iteration

        return metadata

    except Exception as e:
        return {'error': str(e), 'valid': False}