#!/usr/bin/env python3
# ==============================================================================
# DDP with Bucketed Gradients and Overlapping Computation (Section 2.3.3)
# ==============================================================================
#
# DESCRIPTION:
#   Distributed Data Parallel wrapper that organizes parameters into buckets
#   and overlaps backward computation with gradient communication by using
#   backward hooks and asynchronous all-reduce on buckets.
#
# KEY FEATURES:
#   - Buckets parameters in reverse order (gradients ready in reverse)
#   - Uses register_post_accumulate_grad_hook() to trigger communication
#   - Asynchronous all-reduce (async_op=True) for each bucket
#   - Overlaps communication with remaining backward computation
#   - Reduces communication overhead vs individual parameter all-reduce
#
# USAGE:
#   from ddp import DDP
#
#   model = ToyModel().to(device)
#   ddp_model = DDP(model, bucket_size_mb=25.0)
#
#   for batch in dataloader:
#       logits = ddp_model(inputs)
#       loss = loss_fn(logits, targets)
#       loss.backward()
#       ddp_model.finish_gradient_synchronization()
#       optimizer.step()
#
# ==============================================================================

from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """Distributed Data Parallel wrapper with bucketed gradients and overlap.

    This implementation:
    1. Groups parameters into buckets (reverse order)
    2. Registers backward hooks on each parameter
    3. When all parameters in a bucket are ready, launches async all-reduce
    4. Continues backward pass while communication happens
    5. Waits for all communication to finish before optimizer step
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        """Construct DDP container with bucketed gradient synchronization.

        Args:
            module: PyTorch nn.Module to be parallelized
            bucket_size_mb: Maximum bucket size in megabytes
        """
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        # Get distributed info
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before creating DDP")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Bucketing data structures
        self.buckets: List[List[nn.Parameter]] = []  # List of buckets (each bucket is list of params)
        self.param_to_bucket: Dict[nn.Parameter, int] = {}  # Map param to bucket index
        self.bucket_ready_params: List[Set[nn.Parameter]] = []  # Track ready params per bucket

        # Communication tracking
        self.bucket_handles: List = []  # Async handles for buckets
        self.bucket_flattened_grads: Dict[int, torch.Tensor] = {}  # Flattened grads for each bucket
        self.bucket_grad_shapes: Dict[int, List] = {}  # Original shapes for unflattening

        # Track communicated gradient storages (for tied weights)
        self.communicated_grad_storages: Set[int] = set()

        # Broadcast parameters from rank 0 to ensure same initial state
        self._broadcast_parameters()

        # Create buckets and register hooks
        self._create_buckets()
        self._register_hooks()

    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all ranks."""
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def _create_buckets(self):
        """Create parameter buckets in reverse order.

        Groups parameters into buckets of max bucket_size_mb MB.
        Reverse order because gradients become ready in reverse during backward.
        """
        # Get all parameters with gradients in reverse order
        params_reversed = list(reversed([p for p in self.module.parameters() if p.requires_grad]))

        if not params_reversed:
            return

        # Convert MB to bytes
        bucket_size_bytes = self.bucket_size_mb * 1024 * 1024

        current_bucket = []
        current_bucket_size = 0

        for param in params_reversed:
            param_size = param.numel() * param.element_size()

            # Start new bucket if adding this param would exceed bucket size
            # (unless current bucket is empty - always add at least one param)
            if current_bucket and (current_bucket_size + param_size) > bucket_size_bytes:
                # Finalize current bucket
                bucket_idx = len(self.buckets)
                self.buckets.append(current_bucket)
                self.bucket_ready_params.append(set())

                # Map params to this bucket
                for p in current_bucket:
                    self.param_to_bucket[p] = bucket_idx

                # Start new bucket
                current_bucket = [param]
                current_bucket_size = param_size
            else:
                current_bucket.append(param)
                current_bucket_size += param_size

        # Add final bucket if non-empty
        if current_bucket:
            bucket_idx = len(self.buckets)
            self.buckets.append(current_bucket)
            self.bucket_ready_params.append(set())

            for p in current_bucket:
                self.param_to_bucket[p] = bucket_idx

    def _register_hooks(self):
        """Register post-accumulate-grad hooks on all parameters.

        These hooks are called automatically during backward() when each
        parameter's gradient is ready.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                # Register hook that marks param as ready
                param.register_post_accumulate_grad_hook(
                    self._make_allreduce_hook(param)
                )

    def _make_allreduce_hook(self, param: nn.Parameter):
        """Create a hook function for tracking bucket readiness.

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

            # Get actual optimizer gradient (not hook's temporary tensor)
            grad_tensor = param.grad if param.grad is not None else grad

            # Get unique identifier for this gradient's storage (tied weights)
            grad_ptr = grad_tensor.data_ptr()

            if grad_ptr in self.communicated_grad_storages:
                return None

            self.communicated_grad_storages.add(grad_ptr)

            # Mark this parameter as ready in its bucket
            bucket_idx = self.param_to_bucket.get(param)
            if bucket_idx is None:
                return None

            self.bucket_ready_params[bucket_idx].add(param)

            # Check if all parameters in bucket are ready
            bucket_params = self.buckets[bucket_idx]
            if len(self.bucket_ready_params[bucket_idx]) == len(bucket_params):
                # All params ready - launch async all-reduce for this bucket
                self._allreduce_bucket(bucket_idx)

            return None

        return hook

    def _allreduce_bucket(self, bucket_idx: int):
        """Flatten and all-reduce a bucket when all its parameters are ready.

        Args:
            bucket_idx: Index of the bucket to all-reduce
        """
        bucket_params = self.buckets[bucket_idx]

        # Collect gradients for this bucket
        grads = []
        shapes = []

        for param in bucket_params:
            if param.grad is not None:
                grads.append(param.grad.data)
                shapes.append(param.grad.shape)

        if not grads:
            return

        # Flatten all gradients in bucket into single tensor
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        # Launch asynchronous all-reduce on flattened bucket
        handle = dist.all_reduce(
            flat_grads,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        # Store handle and metadata for later
        self.bucket_handles.append((handle, flat_grads, bucket_idx))
        self.bucket_flattened_grads[bucket_idx] = flat_grads
        self.bucket_grad_shapes[bucket_idx] = (grads, shapes)

    def finish_gradient_synchronization(self):
        """Wait for asynchronous bucket all-reduces and copy back to parameters.

        This must be called after backward() and before optimizer.step()
        to ensure all gradients are synchronized.
        """
        # Wait for all async all-reduce operations to complete
        for handle, flat_grads, bucket_idx in self.bucket_handles:
            handle.wait()  # Block until operation is queued on GPU

            # Average the gradient (all-reduce gives sum)
            flat_grads /= self.world_size

            # Unflatten back to individual parameter gradients
            grads, shapes = self.bucket_grad_shapes[bucket_idx]
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)

            # Copy synchronized gradients back to parameters
            for param_grad, unflat_grad in zip(grads, unflat_grads):
                param_grad.copy_(unflat_grad)

        # Clear state for next iteration
        self.bucket_handles.clear()
        self.bucket_flattened_grads.clear()
        self.bucket_grad_shapes.clear()
        self.communicated_grad_storages.clear()

        # Reset bucket ready tracking
        for ready_set in self.bucket_ready_params:
            ready_set.clear()

    def forward(self, *inputs, **kwargs):
        """Call wrapped module's forward() with provided arguments.

        Args:
            *inputs: Positional arguments to forward()
            **kwargs: Keyword arguments to forward()

        Returns:
            Output from wrapped module
        """
        return self.module(*inputs, **kwargs)