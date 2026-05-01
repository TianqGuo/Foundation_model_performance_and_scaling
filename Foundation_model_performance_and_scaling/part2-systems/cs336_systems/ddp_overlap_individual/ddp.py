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

from typing import Optional

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

        # Track which gradient storages we've already communicated
        # (important for tied weights where multiple params share same gradient)
        self.communicated_grad_storages = set()

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
                grad: Gradient tensor provided by autograd

            Returns:
                None (communication happens in-place)
            """
            if grad is None:
                return None

            # Autograd sometimes passes a different tensor than param.grad. Make
            # sure we operate on the actual optimizer gradient so that the
            # upcoming optimizer step observes the synchronized values.
            grad_tensor = param.grad if param.grad is not None else grad

            # Get unique identifier for this gradient's storage. This handles
            # tied weights that share the same grad buffer.
            grad_ptr = grad_tensor.data_ptr()

            if grad_ptr in self.communicated_grad_storages:
                return None

            self.communicated_grad_storages.add(grad_ptr)

            # Launch asynchronous all-reduce. This queues the op immediately and
            # allows the backward pass to continue executing.
            handle = dist.all_reduce(
                grad_tensor,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

            # Store handle (and tensor reference) so we can wait/average later.
            self.comm_handles.append((handle, grad_tensor))
            return None

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

        # Clear handles and storage tracking for next iteration
        self.comm_handles.clear()
        self.communicated_grad_storages.clear()

    def forward(self, *inputs, **kwargs):
        """Call wrapped module's forward() with provided arguments.

        Args:
            *inputs: Positional arguments to forward()
            **kwargs: Keyword arguments to forward()

        Returns:
            Output from wrapped module
        """
        return self.module(*inputs, **kwargs)
