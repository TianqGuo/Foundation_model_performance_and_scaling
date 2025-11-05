"""
Script to calculate activation tensor sizes for part (d).

This script computes the theoretical size of a single activation tensor
in the Transformer residual stream for the 2.7B model.

Usage:
    python -m cs336_systems.memory_profiling.analyze_activation_size
"""

from cs336_systems.profiling_benchmarking.benchmark import MODEL_CONFIGS


def calculate_activation_size(
    batch_size: int = 4,
    context_length: int = 512,
    d_model: int = 2560,
    bytes_per_element: int = 4,
) -> dict:
    """
    Calculate the size of a single activation tensor in the residual stream.

    A residual stream activation has shape: (batch_size, context_length, d_model)

    Args:
        batch_size: Batch size
        context_length: Sequence length
        d_model: Model dimension
        bytes_per_element: 4 for FP32, 2 for FP16/BF16

    Returns:
        Dictionary with size information
    """
    # Number of elements in activation tensor
    num_elements = batch_size * context_length * d_model

    # Size in bytes
    size_bytes = num_elements * bytes_per_element

    # Convert to MB (using 1024^2)
    size_mb = size_bytes / (1024 ** 2)

    return {
        "shape": (batch_size, context_length, d_model),
        "num_elements": num_elements,
        "bytes_per_element": bytes_per_element,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
    }


def main():
    # Get 2.7B model configuration
    config = MODEL_CONFIGS["2.7B"]

    # Reference hyperparameters
    batch_size = 4
    context_length = 512

    print("=" * 80)
    print("Activation Tensor Size Analysis - Part (d)")
    print("=" * 80)
    print()
    print(f"Model: {config.name}")
    print(f"Reference hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  d_model: {config.d_model}")
    print()

    # Calculate for single precision (FP32)
    print("-" * 80)
    print("Single-Precision (FP32) Activation Tensor Size")
    print("-" * 80)

    fp32_info = calculate_activation_size(
        batch_size=batch_size,
        context_length=context_length,
        d_model=config.d_model,
        bytes_per_element=4,
    )

    print(f"Tensor shape: {fp32_info['shape']}")
    print(f"  = (batch_size, context_length, d_model)")
    print(f"  = ({batch_size}, {context_length}, {config.d_model})")
    print()
    print(f"Number of elements: {fp32_info['num_elements']:,}")
    print(f"Bytes per element (FP32): {fp32_info['bytes_per_element']}")
    print()
    print(f"Total size:")
    print(f"  {fp32_info['size_bytes']:,} bytes")
    print(f"  {fp32_info['size_mb']:.4f} MB")
    print()

    # Derivation explanation
    print("-" * 80)
    print("Derivation:")
    print("-" * 80)
    print()
    print("A single activation tensor in the Transformer residual stream has shape:")
    print(f"  (B, T, d_model) = ({batch_size}, {context_length}, {config.d_model})")
    print()
    print("Number of elements:")
    print(f"  B × T × d_model = {batch_size} × {context_length} × {config.d_model}")
    print(f"                  = {fp32_info['num_elements']:,} elements")
    print()
    print("Size in bytes (single precision = 4 bytes per float):")
    print(f"  {fp32_info['num_elements']:,} × 4 bytes = {fp32_info['size_bytes']:,} bytes")
    print()
    print("Size in MB (dividing by 1024²):")
    print(f"  {fp32_info['size_bytes']:,} / (1024²) = {fp32_info['size_mb']:.4f} MB")
    print()

    # Compare with mixed precision
    print("-" * 80)
    print("For comparison: Mixed Precision (BF16/FP16)")
    print("-" * 80)
    print()

    fp16_info = calculate_activation_size(
        batch_size=batch_size,
        context_length=context_length,
        d_model=config.d_model,
        bytes_per_element=2,
    )

    print(f"BF16/FP16 activation size: {fp16_info['size_mb']:.4f} MB")
    print(f"Memory savings vs FP32: {(1 - fp16_info['size_mb'] / fp32_info['size_mb']) * 100:.1f}%")
    print()

    # Context about why this matters
    print("-" * 80)
    print("Context:")
    print("-" * 80)
    print()
    print(f"The 2.7B model has {config.num_layers} layers.")
    print()
    print("During the forward pass, we need to store activations at each layer")
    print("for use in the backward pass. This means storing approximately:")
    print(f"  {config.num_layers} layers × {fp32_info['size_mb']:.4f} MB/layer")
    print(f"  ≈ {config.num_layers * fp32_info['size_mb']:.2f} MB for residual stream activations")
    print()
    print("Additional memory is needed for:")
    print("  - Attention intermediate values (Q, K, V, attention scores)")
    print("  - Feed-forward intermediate activations")
    print("  - Gradients (same size as activations)")
    print("  - Model parameters")
    print("  - Optimizer states")
    print()
    print("This explains why the peak memory from part (b) is much larger than")
    print("just the activation sizes.")
    print()

    print("=" * 80)
    print("ANSWER FOR WRITEUP (Part d):")
    print("=" * 80)
    print()
    print(f"At the reference hyperparameters (batch size {batch_size}, context length {context_length}),")
    print(f"a single activation tensor in the Transformer residual stream has shape")
    print(f"({batch_size}, {context_length}, {config.d_model}) with {fp32_info['num_elements']:,} elements.")
    print(f"In single precision (FP32), this requires {fp32_info['size_bytes']:,} bytes,")
    print(f"which equals {fp32_info['size_mb']:.4f} MB (dividing by 1024²).")
    print()


if __name__ == "__main__":
    main()
