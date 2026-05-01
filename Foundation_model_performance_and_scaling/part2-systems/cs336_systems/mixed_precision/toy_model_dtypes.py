"""
Analyze data types in a toy model under FP16 mixed precision autocasting.

This script demonstrates how torch.autocast selectively casts operations
and why layer normalization is treated differently from linear layers.

Usage:
    python -m cs336_systems.mixed_precision.toy_model_dtypes
"""

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    """Simple model with linear layers, ReLU, and layer normalization."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def analyze_dtypes():
    """Analyze data types throughout model with FP16 autocasting."""

    print("=" * 80)
    print("ToyModel Data Type Analysis with FP16 Mixed Precision")
    print("=" * 80)
    print()

    # Create model and input (FP32 by default)
    model = ToyModel(in_features=5, out_features=3).cuda()
    x = torch.randn(2, 5).cuda()  # batch_size=2, in_features=5

    print("Initial Setup:")
    print(f"  Model parameter dtype (fc1.weight): {model.fc1.weight.dtype}")
    print(f"  Input dtype: {x.dtype}")
    print()

    # Run forward pass WITHOUT autocast
    print("-" * 80)
    print("Forward Pass WITHOUT Autocast (baseline - all FP32):")
    print("-" * 80)

    # Hook to capture intermediate outputs
    dtypes_no_autocast = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                dtypes_no_autocast[name] = output.dtype
        return hook

    # Register hooks
    handle1 = model.fc1.register_forward_hook(make_hook("fc1_output"))
    handle2 = model.ln.register_forward_hook(make_hook("ln_output"))
    handle3 = model.fc2.register_forward_hook(make_hook("fc2_output"))

    # Forward pass
    logits_no_autocast = model(x)
    loss_no_autocast = logits_no_autocast.sum()
    loss_no_autocast.backward()

    print(f"  fc1 output dtype: {dtypes_no_autocast['fc1_output']}")
    print(f"  ln output dtype: {dtypes_no_autocast['ln_output']}")
    print(f"  fc2 output (logits) dtype: {dtypes_no_autocast['fc2_output']}")
    print(f"  Loss dtype: {loss_no_autocast.dtype}")
    print(f"  fc1.weight.grad dtype: {model.fc1.weight.grad.dtype}")
    print()

    # Clean up hooks and gradients
    handle1.remove()
    handle2.remove()
    handle3.remove()
    model.zero_grad()

    # Run forward pass WITH FP16 autocast
    print("-" * 80)
    print("Forward Pass WITH FP16 Autocast:")
    print("-" * 80)

    # Capture dtypes within autocast
    dtypes_with_autocast = {}
    param_dtypes_in_autocast = {}

    def make_hook_autocast(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                dtypes_with_autocast[name] = output.dtype
            # Also capture parameter dtype during forward
            if hasattr(module, 'weight'):
                param_dtypes_in_autocast[name] = module.weight.dtype
        return hook

    # Register hooks
    handle1 = model.fc1.register_forward_hook(make_hook_autocast("fc1"))
    handle2 = model.ln.register_forward_hook(make_hook_autocast("ln"))
    handle3 = model.fc2.register_forward_hook(make_hook_autocast("fc2"))

    # Forward pass with autocast
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits_with_autocast = model(x)
        loss_with_autocast = logits_with_autocast.sum()

    # Backward (outside autocast)
    loss_with_autocast.backward()

    print(f"  Model parameters dtype (fc1.weight): {param_dtypes_in_autocast.get('fc1', 'N/A')}")
    print(f"    → Parameters remain FP32 (not cast by autocast)")
    print()
    print(f"  fc1 output dtype: {dtypes_with_autocast['fc1']}")
    print(f"    → Linear layer output cast to FP16 for faster matmul")
    print()
    print(f"  ln output dtype: {dtypes_with_autocast['ln']}")
    print(f"    → Layer norm kept in FP32 for numerical stability")
    print()
    print(f"  fc2 output (logits) dtype: {dtypes_with_autocast['fc2']}")
    print(f"    → Linear layer output cast to FP16")
    print()
    print(f"  Loss dtype: {loss_with_autocast.dtype}")
    print(f"    → Loss cast to FP16 (sum operation accepts FP16 input)")
    print()
    print(f"  fc1.weight.grad dtype: {model.fc1.weight.grad.dtype}")
    print(f"    → Gradients kept in FP32 (backward pass outside autocast)")
    print()

    # Clean up
    handle1.remove()
    handle2.remove()
    handle3.remove()

    print("=" * 80)
    print("Summary - Data Types with FP16 Autocast:")
    print("=" * 80)
    print()
    print("1. Model parameters:        FP32 (unchanged)")
    print("2. fc1 output:              FP16 (cast by autocast)")
    print("3. ln output:               FP32 (preserved for stability)")
    print("4. fc2 output (logits):     FP16 (cast by autocast)")
    print("5. Loss:                    FP16 (computed from FP16 logits)")
    print("6. Gradients:               FP32 (backward outside autocast)")
    print()

    print("=" * 80)
    print("Why Layer Normalization is Different:")
    print("=" * 80)
    print()
    print("Layer normalization involves:")
    print("  1. Computing mean: E[x]")
    print("  2. Computing variance: E[(x - E[x])²]")
    print("  3. Normalization: (x - mean) / sqrt(var + eps)")
    print()
    print("These operations are sensitive to precision because:")
    print("  - Variance computation involves subtractions (x - mean)")
    print("  - Catastrophic cancellation can occur in FP16")
    print("  - Division by small variance can cause numerical issues")
    print("  - eps (typically 1e-5) may underflow in FP16")
    print()
    print("FP16 range: ~6e-5 to 65504")
    print("  → eps=1e-5 rounds to 0 in FP16!")
    print()
    print("BF16 range: ~1e-38 to 3.4e38 (same exponent range as FP32)")
    print("  → eps=1e-5 is representable in BF16")
    print()
    print("However, BF16 has only 7 bits of mantissa (vs 10 in FP16, 23 in FP32):")
    print("  → Reduced precision can still cause issues in normalization")
    print("  → PyTorch still uses FP32 for layer norm even with BF16 autocast")
    print()
    print("Conclusion:")
    print("  - Layer norm needs FP32 for both FP16 and BF16 mixed precision")
    print("  - Autocast automatically handles this for numerical stability")
    print("  - Linear layers can use FP16/BF16 safely (matmul-heavy, not sensitive)")
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        exit(1)

    analyze_dtypes()
