"""
Softmax function implementation with numerical stability.

ALTERNATIVE: You can use torch.softmax(x, dim=dim) directly
============================================================

PyTorch's built-in torch.softmax is equivalent to our implementation and also
includes numerical stability (max subtraction trick). The built-in version is:

    import torch.nn.functional as F
    result = F.softmax(x, dim=dim)
    # or equivalently:
    result = torch.softmax(x, dim=dim)

Reasons to use our custom implementation:
1. Educational purposes - understanding the numerical stability trick
2. Explicit control over the implementation details
3. Consistency with assignment requirements

Reasons to use PyTorch's built-in:
1. Highly optimized C++/CUDA kernels
2. More thoroughly tested across edge cases
3. Better integration with autograd and JIT compilation
4. Supports additional parameters like dtype casting

Example comparison:
    # Our implementation
    from cs336_basics.model.softmax import softmax
    result1 = softmax(x, dim=-1)

    # PyTorch built-in (equivalent)
    result2 = torch.softmax(x, dim=-1)

    # Both should give identical results within numerical precision
"""

import torch
from torch import Tensor
from jaxtyping import Float


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Apply softmax operation on a tensor with numerical stability.

    Uses the trick of subtracting the maximum value along the specified dimension
    to avoid numerical overflow issues (exp(large_number) -> inf).

    Args:
        x: Input tensor of any shape
        dim: Dimension along which to apply softmax

    Returns:
        Output tensor with softmax applied along the specified dimension
    """
    # Subtract the maximum value along the specified dimension for numerical stability
    # keepdim=True ensures the result can be broadcast back to original shape
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max

    # Compute exponentials
    exp_x = torch.exp(x_shifted)

    # Compute sum along the specified dimension
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    # Return normalized probabilities
    return exp_x / sum_exp_x