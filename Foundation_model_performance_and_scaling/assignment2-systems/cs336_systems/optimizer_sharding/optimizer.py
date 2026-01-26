"""
Sharded optimizer implementation for memory-efficient distributed training.

This module implements optimizer state sharding (similar to ZeRO Stage 1),
where optimizer states are partitioned across ranks to reduce per-rank memory usage.
"""

from typing import Any, Callable, Iterable, Optional, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    A wrapper optimizer that shards optimizer state across distributed ranks.

    This implementation partitions model parameters across ranks, with each rank
    maintaining optimizer states for only ~1/world_size of the parameters. After
    each optimizer step, updated parameters are broadcasted to synchronize across ranks.

    This approach (similar to ZeRO Stage 1) reduces per-rank optimizer memory by
    world_size factor while maintaining identical training dynamics to non-sharded training.

    Args:
        params: Model parameters (or parameter groups) to optimize
        optimizer_cls: The optimizer class to wrap (e.g., torch.optim.AdamW)
        **kwargs: Keyword arguments forwarded to the wrapped optimizer constructor

    Example:
        >>> model = MyModel()
        >>> sharded_opt = ShardedOptimizer(
        ...     model.parameters(),
        ...     torch.optim.AdamW,
        ...     lr=1e-3,
        ...     weight_decay=0.01
        ... )
        >>> # Training loop
        >>> for batch in dataloader:
        ...     sharded_opt.zero_grad()
        ...     loss = model(batch)
        ...     loss.backward()
        ...     sharded_opt.step()  # Updates and broadcasts this rank's parameters
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls: Type[Optimizer],
        **kwargs: Any
    ):
        """
        Initialize the sharded optimizer.

        Args:
            params: Model parameters or parameter groups to optimize
            optimizer_cls: Optimizer class to wrap (e.g., torch.optim.AdamW)
            **kwargs: Additional arguments passed to optimizer_cls constructor
        """
        # Get distributed info
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized before ShardedOptimizer")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Store the optimizer class and kwargs for creating the wrapped optimizer
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # Track all parameter groups (will be populated by add_param_group)
        # This tracks ALL parameters across all ranks (for broadcasting)
        self.all_param_groups = []

        # Create empty defaults dict for the super().__init__ call
        # The super class will call add_param_group for each param group
        defaults = kwargs.copy()

        # Call super().__init__ with params - this will trigger add_param_group() calls
        super().__init__(params, defaults)

        # Now create the wrapped optimizer with only this rank's parameters
        # Collect this rank's parameters from all param groups
        self.rank_param_groups = []
        for group_info in self.all_param_groups:
            rank_params = [p for p, owner_rank in group_info['params_with_ranks']
                          if owner_rank == self.rank and p.requires_grad]
            if rank_params:
                # Create param group for wrapped optimizer with same hyperparameters
                rank_group = {k: v for k, v in group_info['group'].items() if k != 'params'}
                rank_group['params'] = rank_params
                self.rank_param_groups.append(rank_group)

        # Create the wrapped optimizer with only this rank's parameters
        self.wrapped_optimizer = optimizer_cls(self.rank_param_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Add a parameter group to the optimizer.

        This method partitions the parameters in the group across ranks and stores
        the mapping. It's called by the Optimizer super-class during initialization
        and can be called during training (e.g., for layer unfreezing).

        Args:
            param_group: Dictionary with 'params' key and optional hyperparameters
        """
        # Extract parameters from the param group
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)

        # Filter to only parameters that require grad
        params = [p for p in params if p.requires_grad]

        # Assign each parameter to a rank using round-robin distribution
        # This ensures roughly equal distribution across ranks
        params_with_ranks = []
        for idx, param in enumerate(params):
            owner_rank = idx % self.world_size
            params_with_ranks.append((param, owner_rank))

        # Store the param group info (we'll use this when creating wrapped optimizer)
        group_info = {
            'params_with_ranks': params_with_ranks,
            'group': param_group.copy()  # Store original param group config
        }
        self.all_param_groups.append(group_info)

        # Call super().add_param_group() with all params (even though we only optimize subset)
        # This is needed to maintain the param_groups structure expected by the Optimizer base class
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, **kwargs) -> Optional[float]:
        """
        Perform a single optimization step on this rank's parameters and broadcast updates.

        Args:
            closure: Optional closure to re-evaluate the model (used by some optimizers)
            **kwargs: Additional arguments passed to wrapped optimizer's step()

        Returns:
            Loss value if closure is provided, otherwise None
        """
        # Call the wrapped optimizer's step (updates only this rank's parameters)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update this rank's parameters
        self.wrapped_optimizer.step(**kwargs)

        # Synchronize parameters across ranks by broadcasting each parameter from its owner
        for group_info in self.all_param_groups:
            for param, owner_rank in group_info['params_with_ranks']:
                # Each parameter is broadcast from its owner rank to all other ranks
                dist.broadcast(param.data, src=owner_rank)

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Clear gradients of all parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        # Zero gradients on the wrapped optimizer (handles this rank's parameters)
        self.wrapped_optimizer.zero_grad(set_to_none=set_to_none)

        # Also zero gradients for parameters owned by other ranks
        # (they may have gradients from backward pass but won't be updated by this rank)
        for group_info in self.all_param_groups:
            for param, owner_rank in group_info['params_with_ranks']:
                if owner_rank != self.rank and param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()

    def state_dict(self) -> dict:
        """
        Return the state of the optimizer as a dict.

        Only returns state for this rank's parameters.
        """
        return self.wrapped_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the optimizer state.

        Args:
            state_dict: Optimizer state dictionary
        """
        self.wrapped_optimizer.load_state_dict(state_dict)