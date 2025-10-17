#!/usr/bin/env python3
"""
Quick test to verify ablation models can be instantiated and run forward passes.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from cs336_basics.assignment_experiments.ablations.modified_transformer_block import FlexibleTransformerBlock, SiLUFFN
from cs336_basics.assignment_experiments.ablations.modified_transformer_lm import FlexibleTransformerLM


def test_flexible_block():
    """Test FlexibleTransformerBlock with various configurations."""
    print("Testing FlexibleTransformerBlock...")

    configs = [
        ("baseline", {"use_layer_norm": True, "post_norm": False, "use_swiglu": True, "use_rope": True}),
        ("no_layer_norm", {"use_layer_norm": False, "post_norm": False, "use_swiglu": True, "use_rope": True}),
        ("post_norm", {"use_layer_norm": True, "post_norm": True, "use_swiglu": True, "use_rope": True}),
        ("no_rope", {"use_layer_norm": True, "post_norm": False, "use_swiglu": True, "use_rope": False}),
        ("silu", {"use_layer_norm": True, "post_norm": False, "use_swiglu": False, "use_rope": True}),
    ]

    for name, kwargs in configs:
        print(f"  {name}...", end=" ")
        block = FlexibleTransformerBlock(
            d_model=64,
            num_heads=4,
            d_ff=256,
            max_seq_len=128,
            **kwargs
        )

        # Forward pass
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, d_model=64
        positions = torch.arange(10).unsqueeze(0).expand(2, 10)

        output = block(x, token_positions=positions)

        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print("✓")

    print("All block configurations passed!\n")


def test_flexible_lm():
    """Test FlexibleTransformerLM with various configurations."""
    print("Testing FlexibleTransformerLM...")

    configs = [
        ("baseline", {"use_layer_norm": True, "post_norm": False, "use_swiglu": True, "use_rope": True}),
        ("no_layer_norm", {"use_layer_norm": False, "post_norm": False, "use_swiglu": True, "use_rope": True}),
        ("post_norm", {"use_layer_norm": True, "post_norm": True, "use_swiglu": True, "use_rope": True}),
        ("no_rope", {"use_layer_norm": True, "post_norm": False, "use_swiglu": True, "use_rope": False}),
        ("silu", {"use_layer_norm": True, "post_norm": False, "use_swiglu": False, "use_rope": True}),
    ]

    for name, kwargs in configs:
        print(f"  {name}...", end=" ")
        model = FlexibleTransformerLM(
            vocab_size=1000,
            context_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            **kwargs
        )

        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 10))  # batch=2, seq=10
        logits = model(input_ids)

        assert logits.shape == (2, 10, 1000), f"Shape mismatch: {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN in logits"
        print("✓")

    print("All model configurations passed!\n")


def test_silu_ffn():
    """Test SiLU FFN implementation."""
    print("Testing SiLUFFN...")

    ffn = SiLUFFN(d_model=64, d_ff=256)
    x = torch.randn(2, 10, 64)
    output = ffn(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    assert not torch.isnan(output).any(), "NaN in output"

    print("  SiLUFFN passed! ✓\n")


def test_parameter_counts():
    """Verify SwiGLU and SiLU have similar parameter counts."""
    print("Testing parameter count matching...")

    d_model = 512

    # SwiGLU with d_ff ≈ (8/3) * d_model
    d_ff_swiglu = ((int(8/3 * d_model) + 31) // 64) * 64
    model_swiglu = FlexibleTransformerLM(
        vocab_size=10000, context_length=256, d_model=d_model,
        num_layers=4, num_heads=16, d_ff=d_ff_swiglu,
        use_swiglu=True
    )

    # SiLU with d_ff = 4 * d_model
    d_ff_silu = 4 * d_model
    model_silu = FlexibleTransformerLM(
        vocab_size=10000, context_length=256, d_model=d_model,
        num_layers=4, num_heads=16, d_ff=d_ff_silu,
        use_swiglu=False
    )

    params_swiglu = sum(p.numel() for p in model_swiglu.parameters())
    params_silu = sum(p.numel() for p in model_silu.parameters())

    diff_pct = abs(params_swiglu - params_silu) / params_swiglu * 100

    print(f"  SwiGLU parameters: {params_swiglu:,} (d_ff={d_ff_swiglu})")
    print(f"  SiLU parameters: {params_silu:,} (d_ff={d_ff_silu})")
    print(f"  Difference: {diff_pct:.2f}%")

    assert diff_pct < 5, f"Parameter mismatch too large: {diff_pct:.2f}%"
    print("  Parameter counts match! ✓\n")


def main():
    """Run all tests."""
    print("="*60)
    print("ABLATION MODEL TESTS")
    print("="*60 + "\n")

    test_silu_ffn()
    test_flexible_block()
    test_flexible_lm()
    test_parameter_counts()

    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)


if __name__ == "__main__":
    main()
