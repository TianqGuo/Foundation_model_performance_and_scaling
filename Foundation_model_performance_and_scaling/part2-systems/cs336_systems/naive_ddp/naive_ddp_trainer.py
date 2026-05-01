#!/usr/bin/env python3
"""
Naïve Distributed Data Parallel (DDP) Training Implementation

This module implements vanilla DDP by all-reducing individual parameter gradients
after the backward pass. This is "naïve" because it all-reduces each gradient
tensor separately, causing high communication overhead.

Key properties:
- Full model replication: Each device holds complete copy of parameters
- Full optimizer replication: Each device holds complete optimizer states
- Individual gradient all-reduce: Each parameter gradient is all-reduced separately
- No parameter/optimizer sharding: Memory inefficient

Usage:
    from naive_ddp_trainer import NaiveDDPTrainer

    trainer = NaiveDDPTrainer(model, optimizer, rank, world_size)
    trainer.train_step(local_inputs, local_targets)
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDPTrainer:
    """Naïve Distributed Data Parallel trainer.

    Implements vanilla DDP by all-reducing individual parameter gradients
    after backward pass.

    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        rank: Global rank of this process
        world_size: Total number of processes
        device: Device to use (e.g., 'cuda:0')
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rank: int,
        world_size: int,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Broadcast initial parameters from rank 0 to all other ranks
        # This ensures all processes start with identical model
        self._broadcast_parameters()

        # Statistics for tracking communication overhead
        self.total_comm_time = 0.0
        self.num_comm_ops = 0

    def _broadcast_parameters(self):
        """Broadcast model parameters from rank 0 to all other ranks.

        This ensures all processes start with the same initial model state.
        """
        with torch.no_grad():
            for param in self.model.parameters():
                # Broadcast from rank 0 to all ranks
                dist.broadcast(param.data, src=0)

    def _allreduce_gradients(self) -> float:
        """All-reduce gradients across all processes (naïve implementation).

        NAÏVE APPROACH: All-reduce each parameter gradient individually.
        This causes high communication overhead because:
        1. Many small all-reduce operations instead of few large ones
        2. Cannot overlap communication with computation
        3. Higher latency from multiple kernel launches

        Returns:
            Time spent on gradient communication (seconds)
        """
        comm_start = time.perf_counter()

        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # NAÏVE: All-reduce each gradient tensor individually
                    # This is inefficient - optimized DDP would bucket gradients
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

                    # Average gradients (all-reduce gives sum)
                    param.grad.data /= self.world_size

                    self.num_comm_ops += 1

        # Synchronize to get accurate timing
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        comm_time = time.perf_counter() - comm_start
        self.total_comm_time += comm_time

        return comm_time

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Execute one training step with naïve DDP.

        Steps:
        1. Forward pass on local data
        2. Backward pass to compute local gradients
        3. All-reduce gradients (naïve: individual all-reduces)
        4. Optimizer step with synchronized gradients

        Args:
            inputs: Local batch inputs (already sharded across devices)
            targets: Local batch targets (already sharded across devices)
            loss_fn: Loss function (default: nn.CrossEntropyLoss)

        Returns:
            Dictionary with:
                - loss: Training loss value
                - comm_time: Time spent on gradient communication
                - compute_time: Time spent on forward + backward
                - total_time: Total iteration time
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Timing: start iteration
        iter_start = time.perf_counter()

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward + backward (computation)
        compute_start = time.perf_counter()

        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        compute_time = time.perf_counter() - compute_start

        # All-reduce gradients (communication)
        comm_time = self._allreduce_gradients()

        # Optimizer step
        self.optimizer.step()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        total_time = time.perf_counter() - iter_start

        return {
            "loss": loss.item(),
            "comm_time": comm_time,
            "compute_time": compute_time,
            "total_time": total_time,
            "comm_fraction": comm_time / total_time if total_time > 0 else 0,
        }

    def get_communication_stats(self) -> Dict[str, float]:
        """Get accumulated communication statistics.

        Returns:
            Dictionary with total communication time and number of operations
        """
        return {
            "total_comm_time": self.total_comm_time,
            "num_comm_ops": self.num_comm_ops,
            "avg_comm_time": self.total_comm_time / self.num_comm_ops if self.num_comm_ops > 0 else 0,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state (for checkpointing)."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state (for checkpointing)."""
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

        # Re-broadcast to ensure all ranks are synchronized
        self._broadcast_parameters()


def setup_distributed(rank: int, world_size: int, backend: str = "nccl", master_port: str = "29500"):
    """Initialize distributed process group.

    Args:
        rank: Global rank of this process
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        master_port: Port for master process
    """
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set device for this rank
    if backend == "nccl":
        torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def shard_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    rank: int,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shard a batch across processes for data parallel training.

    Divides batch into world_size equal chunks and returns the chunk
    corresponding to this rank.

    Args:
        batch: Tuple of (inputs, targets) with batch_size as first dimension
        rank: Global rank of this process
        world_size: Total number of processes

    Returns:
        Tuple of (local_inputs, local_targets) for this rank

    Example:
        Global batch: [0, 1, 2, 3, 4, 5, 6, 7] with 4 processes
        Rank 0 gets: [0, 1]
        Rank 1 gets: [2, 3]
        Rank 2 gets: [4, 5]
        Rank 3 gets: [6, 7]
    """
    inputs, targets = batch

    batch_size = inputs.shape[0]
    assert batch_size % world_size == 0, (
        f"Batch size ({batch_size}) must be divisible by world size ({world_size})"
    )

    local_batch_size = batch_size // world_size
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size

    local_inputs = inputs[start_idx:end_idx]
    local_targets = targets[start_idx:end_idx]

    return local_inputs, local_targets