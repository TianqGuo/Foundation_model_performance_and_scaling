#!/usr/bin/env python3
# ==============================================================================
# Flat DDP Trainer - Batched Gradient Communication
# ==============================================================================
#
# DESCRIPTION:
#   Distributed data parallel trainer that flattens all gradients into a single
#   tensor before all-reduce to minimize communication overhead.
#
# DIFFERENCES FROM NAIVE DDP:
#   - Naive: All-reduces each parameter gradient individually
#   - Flat: Flattens all gradients into one tensor, single all-reduce call
#
# USAGE:
#   from flat_ddp_trainer import FlatDDPTrainer
#   trainer = FlatDDPTrainer(model, optimizer, rank, world_size)
#   step_info = trainer.train_step(inputs, targets)
#
# ==============================================================================

import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


class FlatDDPTrainer:
    """Distributed data parallel trainer with flattened gradient all-reduce.

    Instead of all-reducing each parameter gradient individually, this trainer
    flattens all gradients into a single tensor and performs one batched
    all-reduce operation. This reduces communication overhead by minimizing
    the number of communication calls.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rank: int,
        world_size: int,
        device: str = "cuda",
    ):
        """Initialize flat DDP trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer for updating parameters
            rank: Global rank of this process
            world_size: Total number of processes
            device: Device to use (default: "cuda")
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

        # Communication statistics
        self.num_comm_ops = 0
        self.total_comm_time = 0.0

    def _allreduce_gradients_flat(self) -> float:
        """All-reduce gradients using a single flattened tensor.

        Flattens all parameter gradients into a single contiguous tensor,
        performs one all-reduce operation, then unflatters back to individual
        parameter gradients.

        Returns:
            Time spent on communication (seconds)
        """
        comm_start = time.perf_counter()

        with torch.no_grad():
            # Collect all gradients that need to be reduced
            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.data)

            if len(gradients) == 0:
                return 0.0

            # Flatten all gradients into a single tensor
            flat_grads = torch._utils._flatten_dense_tensors(gradients)

            # Single all-reduce operation
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            flat_grads /= self.world_size
            self.num_comm_ops += 1

            # Unflatten back to individual parameter gradients
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, gradients)

            # Copy unflattened gradients back to parameters
            for param_grad, unflat_grad in zip(gradients, unflat_grads):
                param_grad.copy_(unflat_grad)

        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        comm_time = time.perf_counter() - comm_start
        self.total_comm_time += comm_time

        return comm_time

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> dict:
        """Execute one training step.

        Args:
            inputs: Input tensor [batch_size, seq_len]
            targets: Target tensor [batch_size] or [batch_size, seq_len]
            loss_fn: Optional loss function (default: CrossEntropyLoss)

        Returns:
            Dictionary with step statistics:
                - loss: Training loss
                - comm_time: Time spent on gradient communication
                - total_time: Total time for the step
        """
        if loss_fn is None:
            loss_fn = self.loss_fn

        step_start = time.perf_counter()

        # Move to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # All-reduce gradients (flattened)
        comm_time = self._allreduce_gradients_flat()

        # Optimizer step
        self.optimizer.step()

        # Total time
        torch.cuda.synchronize()
        total_time = time.perf_counter() - step_start

        return {
            "loss": loss.item(),
            "comm_time": comm_time,
            "total_time": total_time,
        }

    def get_communication_stats(self) -> dict:
        """Get communication statistics.

        Returns:
            Dictionary with communication statistics:
                - num_comm_ops: Number of communication operations
                - total_comm_time: Total time spent on communication
        """
        return {
            "num_comm_ops": self.num_comm_ops,
            "total_comm_time": self.total_comm_time,
        }