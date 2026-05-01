# Implementation Notes and Known Issues

This document tracks important implementation details, gotchas, and solutions for the Section 2.3.2 overlap DDP module.

## Table of Contents

1. [Tied Weights and Gradient Deduplication](#tied-weights-and-gradient-deduplication)
2. [Future Issues](#future-issues)

---

## Tied Weights and Gradient Deduplication

### Issue Description

**Symptom**: Test failures with `ToyModelWithTiedWeights`, where DDP model parameters don't match non-parallel baseline after training.

**Root Cause**: When multiple parameters share the same gradient storage (tied/shared weights), naive iteration over parameters causes multiple gradient synchronization operations on the same gradient tensor.

### Example: Tied Weights

```python
class ModelWithTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.linear = nn.Linear(64, 100)

        # Tie weights: both parameters point to same storage
        self.linear.weight = self.embed.weight  # Tied!
```

In this case:
- `self.embed.weight` and `self.linear.weight` are the **same tensor**
- Both have `requires_grad=True`
- They share the **same gradient storage**
- During backward, gradients accumulate into the **same `grad` tensor**

### Why This Affects Different DDP Implementations

#### Naive DDP (Synchronous, After Backward)

```python
# PROBLEMATIC: Iterates over all parameters
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad.data)  # Multiple all-reduces on same gradient!
        param.grad.data /= world_size
```

**Problem**: If `param1` and `param2` share a gradient, this loop does:
1. First iteration: `all_reduce(shared_grad)` → `shared_grad = sum / world_size`
2. Second iteration: `all_reduce(shared_grad)` → `shared_grad = sum / world_size` **AGAIN!**

Result: Gradient is incorrectly averaged twice (or N times for N tied parameters).

#### Flat DDP (Batched, After Backward)

Flat DDP typically **flattens all gradients into one tensor**, which naturally deduplicates:

```python
# Collect unique gradient tensors
gradients = []
for param in model.parameters():
    if param.grad is not None:
        gradients.append(param.grad.data)

# Flatten into single tensor (deduplicates automatically)
flat_grads = torch._utils._flatten_dense_tensors(gradients)
dist.all_reduce(flat_grads)
```

**Why it works**: The list `gradients` might contain the same tensor multiple times, but when PyTorch flattens them, it handles the deduplication implicitly (or the test might not have caught the bug).

#### Overlap DDP (Async Hooks, During Backward)

```python
# PROBLEMATIC: Each parameter gets its own hook
for param in model.parameters():
    if param.requires_grad:
        param.register_post_accumulate_grad_hook(
            self._make_allreduce_hook(param)
        )
```

**Problem**: When tied weights exist:
1. Two parameters share `grad` tensor
2. Both get hooks registered
3. During backward, when `grad` is ready, **both hooks fire**
4. Both hooks call `dist.all_reduce(grad.data, async_op=True)`
5. Same gradient gets all-reduced **twice in parallel**

Result: Gradient is summed twice across ranks, then divided by `world_size` twice → incorrect values.

### The Solution: Gradient Deduplication

Track which gradient storages have already been communicated using their memory address, but make sure to operate on `param.grad` (the tensor the optimizer will later mutate) rather than whatever autograd passes into the hook:

```python
class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.comm_handles = []

        # Track communicated gradients by storage pointer
        self.communicated_grad_storages = set()

        # ... rest of init

    def _make_allreduce_hook(self, param):
        def hook(grad):
            if grad is None:
                return None

            grad_tensor = param.grad if param.grad is not None else grad
            grad_ptr = grad_tensor.data_ptr()

            if grad_ptr not in self.communicated_grad_storages:
                self.communicated_grad_storages.add(grad_ptr)

                handle = dist.all_reduce(grad_tensor, async_op=True)
                self.comm_handles.append((handle, grad_tensor))

            return None
        return hook

    def finish_gradient_synchronization(self):
        # Wait for all-reduce, average, then clear tracking
        for handle, grad in self.comm_handles:
            handle.wait()
            grad.data /= self.world_size

        self.comm_handles.clear()
        self.communicated_grad_storages.clear()  # Reset for next iteration
```

### Key Insights

1. **Storage Pointers are Unique**: `grad_tensor.data_ptr()` returns the memory address of the gradient's underlying storage. Multiple parameter tensors sharing the same storage will have the **same data pointer**.

2. **Hook Input ≠ Optimizer Tensor**: Autograd may supply a temporary tensor to the hook. Always fall back to `param.grad` when it exists so that the values the optimizer sees are the synchronized ones.

3. **Hook Timing**: `register_post_accumulate_grad_hook()` fires when a parameter's gradient is **ready**, not when the parameter is used. For tied weights, the gradient is shared, so all hooks fire at the same time.

4. **Async Makes it Obvious**: With async hooks, the issue is more apparent because hooks are independent callbacks. But the bug exists in any implementation that doesn't deduplicate.

5. **Must Clear Tracking**: The `communicated_grad_storages` set must be cleared after each iteration, otherwise gradients won't be communicated in subsequent iterations.

### Testing for Tied Weights

The test suite includes `ToyModelWithTiedWeights` specifically to catch this issue:

```python
@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_DistributedDataParallelIndividualParameters(model_class):
    # ... test code that verifies DDP matches non-parallel baseline
```

This ensures that DDP implementations handle tied weights correctly.

### Recommended Practice

**Always deduplicate gradients** when iterating over parameters for communication:

```python
# Bad: May communicate same gradient multiple times
for param in model.parameters():
    if param.grad is not None:
        communicate(param.grad)

# Good: Track unique gradients by storage pointer
communicated = set()
for param in model.parameters():
    if param.grad is not None:
        grad_ptr = param.grad.data_ptr()
        if grad_ptr not in communicated:
            communicated.add(grad_ptr)
            communicate(param.grad)
```

---

## Future Issues

*This section will be updated as new issues are discovered and resolved.*

### Template for Adding New Issues

When documenting a new issue, include:

1. **Issue Description**: What symptom did you observe?
2. **Root Cause**: Why does this happen?
3. **Affected Components**: Which implementations are affected?
4. **Solution**: How was it fixed?
5. **Testing**: How can this be tested/prevented?

---

## Contributing

When you discover a new issue or implementation detail worth documenting:

1. Add a new section under the appropriate category
2. Follow the documentation format above
3. Include code examples where helpful
4. Link to relevant test cases or benchmarks

---

**Last Updated**: 2026-01-24
**Maintainers**: CS336 Assignment 2 Contributors
