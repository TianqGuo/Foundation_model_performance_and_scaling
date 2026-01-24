#!/usr/bin/env python3
# ==============================================================================
# DDP with Overlapping Computation and Communication (Section 2.3.2)
# ==============================================================================
#
# DESCRIPTION:
#   Distributed Data Parallel wrapper that overlaps backward computation with
#   gradient communication by using backward hooks and asynchronous all-reduce.
#
# KEY FEATURES:
#   - Uses register_post_accumulate_grad_hook() to trigger communication
#   - Asynchronous all-reduce (async_op=True) for each parameter
#   - Overlaps communication with remaining backward computation
#
# USAGE:
#   from ddp import DDP
#
#   model = ToyModel().to(device)
#   ddp_model = DDP(model)
#
#   for batch in dataloader:
#       logits = ddp_model(inputs)
#       loss = loss_fn(logits, targets)
#       loss.backward()
#       ddp_model.finish_gradient_synchronization()
#       optimizer.step()
#
# ==============================================================================

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """Distributed Data Parallel wrapper with computation/communication overlap.

    This implementation overlaps backward pass computation with gradient
    communication by:
    1. Registering backward hooks on each parameter
    2. When a gradient is ready, immediately launch async all-reduce
    3. Continue backward pass while communication happens
    4. Wait for all communication to finish before optimizer step
    """

    def __init__(self, module: nn.Module):
        """Construct DDP container to handle gradient synchronization.

        Args:
            module: PyTorch nn.Module to be parallelized
        """
        super().__init__()
        self.module = module

        # Get distributed info
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before creating DDP")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Track async communication handles
        self.comm_handles = []

        # Broadcast parameters from rank 0 to ensure same initial state
        self._broadcast_parameters()

        # Register backward hooks for each parameter
        self._register_hooks()

    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all ranks."""
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def _register_hooks(self):
        """Register post-accumulate-grad hooks on all parameters.

        These hooks are called automatically during backward() when each
        parameter's gradient is ready.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                # Register hook that triggers async all-reduce
                param.register_post_accumulate_grad_hook(
                    self._make_allreduce_hook(param)
                )

    def _make_allreduce_hook(self, param: nn.Parameter):
        """Create a hook function for async all-reduce of a parameter.

        Args:
            param: Parameter to create hook for

        Returns:
            Hook function that will be called when gradient is ready
        """
        def hook(grad: torch.Tensor) -> Optional[torch.Tensor]:
            """Hook called when gradient is accumulated.

            Args:
                grad: The gradient tensor

            Returns:
                None (in-place modification)
            """
            if grad is not None:
                # Launch asynchronous all-reduce
                # This returns immediately without blocking
                handle = dist.all_reduce(
                    grad.data,
                    op=dist.ReduceOp.SUM,
                    async_op=True  # KEY: Asynchronous operation
                )

                # Store handle to wait on later
                self.comm_handles.append((handle, grad))

            return None  # In-place modification

        return hook

    def finish_gradient_synchronization(self):
        """Wait for asynchronous communication calls to be queued on GPU.

        This must be called after backward() and before optimizer.step()
        to ensure all gradients are synchronized.
        """
        # Wait for all async all-reduce operations to complete
        for handle, grad in self.comm_handles:
            handle.wait()  # Block until operation is queued on GPU

            # Average the gradient (all-reduce gives sum)
            grad.data /= self.world_size

        # Clear handles for next iteration
        self.comm_handles.clear()

    def forward(self, *inputs, **kwargs):
        """Call wrapped module's forward() with provided arguments.

        Args:
            *inputs: Positional arguments to forward()
            **kwargs: Keyword arguments to forward()

        Returns:
            Output from wrapped module
        """
        return self.module(*inputs, **kwargs)