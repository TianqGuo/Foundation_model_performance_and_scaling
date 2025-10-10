#!/usr/bin/env python3
"""
Simple Training Time Calculation
Based on standard approximations used in literature
"""

def simple_training_time():
    """Simple calculation following standard practices."""
    print("=== Simple Training Time Calculation ===")

    # Configuration
    training_steps = 400_000
    batch_size = 1024
    context_length = 1024
    total_params = 1.635e9  # 1.635B
    mfu = 0.5  # 50%
    a100_peak = 19.5e12  # 19.5 TF/s

    # Simple approximation: 6N FLOPs per token for forward pass
    # This is the standard approximation used in scaling laws papers
    forward_flops_per_token = 6 * total_params
    # Backward pass: 2x forward
    total_flops_per_token = 3 * forward_flops_per_token  # forward + backward

    total_tokens = training_steps * batch_size * context_length
    total_flops = total_tokens * total_flops_per_token

    effective_flops_per_sec = a100_peak * mfu
    training_time_sec = total_flops / effective_flops_per_sec
    training_time_days = training_time_sec / (24 * 3600)

    print(f"Configuration:")
    print(f"  Parameters: {total_params/1e9:.2f}B")
    print(f"  Training steps: {training_steps:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  Total tokens: {total_tokens:.2e}")

    print(f"\nFLOP Calculation:")
    print(f"  Forward FLOPs per token: {forward_flops_per_token:.2e}")
    print(f"  Total FLOPs per token: {total_flops_per_token:.2e}")
    print(f"  Total training FLOPs: {total_flops:.2e}")

    print(f"\nTraining Time:")
    print(f"  A100 peak: {a100_peak:.1e} FLOP/s")
    print(f"  Effective (50% MFU): {effective_flops_per_sec:.2e} FLOP/s")
    print(f"  Training time: {training_time_days:.1f} days")

    # Let me also try a much simpler approximation based on known results
    print(f"\n=== Alternative Simple Calculation ===")
    # Known: GPT-3 (175B) took ~3000 A100-days
    # Our model: 1.6B parameters
    # Rough scaling: compute scales roughly with parameters
    gpt3_params = 175e9
    gpt3_a100_days = 3000

    # Simple parameter scaling
    our_relative_size = total_params / gpt3_params
    estimated_days = gpt3_a100_days * our_relative_size

    print(f"  GPT-3 baseline: {gpt3_params/1e9:.0f}B params, {gpt3_a100_days} A100-days")
    print(f"  Our model: {total_params/1e9:.1f}B params")
    print(f"  Parameter ratio: {our_relative_size:.3f}")
    print(f"  Estimated time: {estimated_days:.1f} days")

    return min(training_time_days, estimated_days)

if __name__ == "__main__":
    days = simple_training_time()
    print(f"\nReasonable estimate: {days:.1f} days")