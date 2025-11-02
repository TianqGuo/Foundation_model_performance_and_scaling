"""
Comparison of different accumulation strategies with mixed precision.

This script demonstrates why keeping accumulations in FP32 is important
even when the values being accumulated are in lower precision (FP16).

Usage:
    python -m cs336_systems.mixed_precision.accumulation_comparison
"""

import torch


def run_accumulation_experiments():
    """Run four accumulation variants and print results."""

    print("=" * 70)
    print("Mixed Precision Accumulation Comparison")
    print("=" * 70)
    print()
    print("Task: Accumulate 0.01 for 1000 iterations (expected result: 10.0)")
    print()

    # Variant 1: Full FP32 (baseline - most accurate)
    print("Variant 1: FP32 accumulator + FP32 values")
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(f"  Result: {s.item():.10f}")
    print(f"  Error:  {abs(s.item() - 10.0):.10e}")
    print()

    # Variant 2: Full FP16 (worst accuracy - accumulator loses precision)
    print("Variant 2: FP16 accumulator + FP16 values")
    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(f"  Result: {s.item():.10f}")
    print(f"  Error:  {abs(s.item() - 10.0):.10e}")
    print("  ⚠️  Large error! FP16 accumulator loses precision as sum grows.")
    print()

    # Variant 3: FP32 accumulator + FP16 values (implicit cast during add)
    print("Variant 3: FP32 accumulator + FP16 values (implicit cast)")
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(f"  Result: {s.item():.10f}")
    print(f"  Error:  {abs(s.item() - 10.0):.10e}")
    print("  ⚠️  Moderate error! FP16->FP32 cast happens after rounding to FP16.")
    print()

    # Variant 4: FP32 accumulator + explicit cast to FP32
    print("Variant 4: FP32 accumulator + FP16 values (explicit cast)")
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(f"  Result: {s.item():.10f}")
    print(f"  Error:  {abs(s.item() - 10.0):.10e}")
    print("  ⚠️  Same as Variant 3 - explicit cast doesn't help here.")
    print()

    print("=" * 70)
    print("Key Insight:")
    print("=" * 70)
    print()
    print("Variant 2 (FP16 accumulator) is worst because:")
    print("  - As the sum grows, FP16's limited precision causes rounding errors")
    print("  - Adding 0.01 to ~10.0 in FP16 can round to the original value")
    print("  - This is why torch.autocast keeps accumulations in FP32")
    print()
    print("Variants 3 & 4 have moderate error because:")
    print("  - 0.01 cannot be exactly represented in FP16")
    print("  - It rounds to ~0.00999755859375 in FP16")
    print("  - Converting to FP32 doesn't recover the lost precision")
    print("  - Accumulated error: 1000 * 0.00000244140625 ≈ 0.00244")
    print()
    print("Variant 1 (full FP32) is most accurate because:")
    print("  - FP32 can represent 0.01 more precisely")
    print("  - Higher precision maintained throughout accumulation")
    print()


if __name__ == "__main__":
    run_accumulation_experiments()
