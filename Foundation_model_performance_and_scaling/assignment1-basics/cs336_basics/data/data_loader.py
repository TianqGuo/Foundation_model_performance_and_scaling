"""
Data loading utilities for transformer training.

Implements efficient batch sampling from tokenized sequences.
"""

import numpy as np
import torch
from typing import Tuple


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of training sequences from tokenized data.

    Takes a single long sequence of tokens and samples batch_size subsequences
    of length context_length, along with their corresponding next-token targets.

    Args:
        data: 1D numpy array of token IDs, shape (sequence_length,)
        batch_size: Number of sequences to sample in the batch
        context_length: Length of each sequence (both input and target)
        device: PyTorch device string ('cpu', 'cuda:0', 'mps', etc.)

    Returns:
        Tuple of (inputs, targets) where:
        - inputs: Tensor of shape (batch_size, context_length) with input token IDs
        - targets: Tensor of shape (batch_size, context_length) with target token IDs

        For each sequence i, targets[i] = inputs[i] shifted by one position.
        E.g., if inputs[i] = [x_j, x_{j+1}, ..., x_{j+m-1}]
              then targets[i] = [x_{j+1}, x_{j+2}, ..., x_{j+m}]

    Example:
        For data = [1, 2, 3, 4, 5, 6, 7, 8], batch_size=2, context_length=3:
        Possible output:
        inputs =  [[2, 3, 4],     # starting at index 1
                   [5, 6, 7]]     # starting at index 4
        targets = [[3, 4, 5],     # inputs shifted right by 1
                   [6, 7, 8]]     # inputs shifted right by 1
    """
    if len(data) < context_length + 1:
        raise ValueError(f"Data length ({len(data)}) must be at least context_length + 1 ({context_length + 1})")

    # Maximum valid starting index for a sequence of length context_length
    # We need context_length + 1 tokens total (input + next token for each position)
    max_start_idx = len(data) - context_length

    # Randomly sample starting indices for each sequence in the batch
    # Each index i gives us tokens [i, i+1, ..., i+context_length]
    # where [i, i+1, ..., i+context_length-1] are inputs and [i+1, i+2, ..., i+context_length] are targets
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    # Extract input sequences: each sequence starts at start_indices[i] and has length context_length
    inputs = np.array([data[idx:idx + context_length] for idx in start_indices])

    # Extract target sequences: same as inputs but shifted right by 1 position
    targets = np.array([data[idx + 1:idx + context_length + 1] for idx in start_indices])

    # Convert to PyTorch tensors and move to specified device
    inputs_tensor = torch.from_numpy(inputs).long().to(device)
    targets_tensor = torch.from_numpy(targets).long().to(device)

    return inputs_tensor, targets_tensor


def load_data_memmap(
    file_path: str,
    dtype: np.dtype = np.uint16
) -> np.ndarray:
    """
    Load tokenized data using memory mapping for efficient large dataset handling.

    Memory mapping allows us to work with datasets larger than RAM by loading
    data on-demand as it's accessed.

    Args:
        file_path: Path to the saved numpy array file (.npy)
        dtype: Data type of the saved array (default: np.uint16 for token IDs)

    Returns:
        Memory-mapped numpy array that can be used like a regular array

    Example:
        # Save tokenized data
        tokens = np.array([1, 2, 3, ...], dtype=np.uint16)
        np.save('tokens.npy', tokens)

        # Load with memory mapping
        data = load_data_memmap('tokens.npy', dtype=np.uint16)
        batch = get_batch(data, batch_size=32, context_length=128)
    """
    try:
        # Try loading with memory mapping using np.load
        data = np.load(file_path, mmap_mode='r')

        # Verify the data type matches expectations
        if data.dtype != dtype:
            print(f"Warning: Expected dtype {dtype}, but loaded {data.dtype}")

        return data

    except Exception as e:
        # Fallback to direct memory mapping if np.load fails
        print(f"np.load failed: {e}")
        print(f"Attempting direct memory mapping...")

        # Determine file size to calculate array length
        import os
        file_size = os.path.getsize(file_path)
        array_length = file_size // np.dtype(dtype).itemsize

        # Create memory-mapped array
        data = np.memmap(file_path, dtype=dtype, mode='r', shape=(array_length,))
        return data


def validate_data(data: np.ndarray, vocab_size: int) -> bool:
    """
    Validate that tokenized data contains only valid token IDs.

    Args:
        data: Array of token IDs to validate
        vocab_size: Maximum valid token ID + 1

    Returns:
        True if all token IDs are valid (0 <= token_id < vocab_size)

    Raises:
        ValueError: If invalid token IDs are found
    """
    min_token = data.min()
    max_token = data.max()

    if min_token < 0:
        raise ValueError(f"Found negative token ID: {min_token}")

    if max_token >= vocab_size:
        raise ValueError(f"Found token ID {max_token} >= vocab_size {vocab_size}")

    print(f"Data validation passed: {len(data)} tokens, range [{min_token}, {max_token}]")
    return True