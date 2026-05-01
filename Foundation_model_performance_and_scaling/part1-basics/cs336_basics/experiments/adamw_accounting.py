#!/usr/bin/env python3
"""
AdamW Resource Accounting Analysis
Comprehensive analysis of memory, FLOPs, and training time for AdamW optimization
"""

import math


def analyze_adamw_memory():
    """
    Problem (a): Calculate peak memory usage for AdamW training.
    All calculations assume float32 (4 bytes per parameter).
    """
    print("=" * 80)
    print("PROBLEM (a): AdamW Memory Usage Analysis")
    print("=" * 80)

    print("\nModel Architecture Assumptions:")
    print("- d_ff = 4 × d_model")
    print("- Float32 precision (4 bytes per parameter)")
    print("- Activation calculations include only specified components")

    print(f"\n{'Component':<20} {'Memory Formula':<60} {'Description'}")
    print("-" * 120)

    # 1. PARAMETERS
    print("\n1. PARAMETERS:")

    # Token embedding: vocab_size × d_model
    token_emb = "vocab_size × d_model"

    # Each transformer block contains:
    # - Multi-head attention: 4 × d_model × d_model (Q, K, V, O projections)
    # - Feed-forward: 2 × d_model × d_ff = 2 × d_model × (4 × d_model) = 8 × d_model²
    # - RMSNorm: 2 × d_model (attention + feedforward)
    transformer_block = "num_layers × (4 × d_model² + 8 × d_model² + 2 × d_model)"
    transformer_block_simplified = "num_layers × (12 × d_model² + 2 × d_model)"

    # Final RMSNorm: d_model
    final_norm = "d_model"

    # Output embedding (language modeling head): d_model × vocab_size
    output_emb = "d_model × vocab_size"

    parameters_total = f"vocab_size × d_model + {transformer_block_simplified} + d_model + d_model × vocab_size"
    parameters_simplified = "2 × vocab_size × d_model + num_layers × (12 × d_model² + 2 × d_model) + d_model"

    print(f"{'Token Embedding':<20} {token_emb:<60}")
    print(f"{'Transformer Blocks':<20} {transformer_block_simplified:<60}")
    print(f"{'Final RMSNorm':<20} {final_norm:<60}")
    print(f"{'Output Embedding':<20} {output_emb:<60}")
    print(f"{'TOTAL':<20} {parameters_simplified:<60}")

    # 2. ACTIVATIONS
    print("\n2. ACTIVATIONS:")

    print("Per transformer block:")
    # RMSNorm outputs: batch_size × context_length × d_model
    rmsnorm_act = "2 × batch_size × context_length × d_model"

    # QKV projections: 3 × batch_size × context_length × d_model
    qkv_act = "3 × batch_size × context_length × d_model"

    # Q^T K attention scores: batch_size × num_heads × context_length × context_length
    attention_scores = "batch_size × num_heads × context_length²"

    # Softmax output (same shape as attention scores)
    softmax_act = "batch_size × num_heads × context_length²"

    # Weighted sum of values: batch_size × context_length × d_model
    weighted_values = "batch_size × context_length × d_model"

    # Output projection: batch_size × context_length × d_model
    output_proj = "batch_size × context_length × d_model"

    # Feed-forward intermediate: batch_size × context_length × d_ff
    ff_intermediate = "batch_size × context_length × 4 × d_model"

    # Feed-forward output: batch_size × context_length × d_model
    ff_output = "batch_size × context_length × d_model"

    block_activations = "num_layers × (11 × batch_size × context_length × d_model + 2 × batch_size × num_heads × context_length²)"

    print(f"{'RMSNorms':<20} {rmsnorm_act:<60}")
    print(f"{'QKV Projections':<20} {qkv_act:<60}")
    print(f"{'Attention Scores':<20} {attention_scores:<60}")
    print(f"{'Softmax':<20} {softmax_act:<60}")
    print(f"{'Weighted Values':<20} {weighted_values:<60}")
    print(f"{'Output Projection':<20} {output_proj:<60}")
    print(f"{'FF Intermediate':<20} {ff_intermediate:<60}")
    print(f"{'FF Output':<20} {ff_output:<60}")

    # Final components
    final_rmsnorm = "batch_size × context_length × d_model"
    output_embedding = "batch_size × context_length × vocab_size"
    cross_entropy = "batch_size × context_length"  # scalar per token

    print(f"\nFinal components:")
    print(f"{'Final RMSNorm':<20} {final_rmsnorm:<60}")
    print(f"{'Output Embedding':<20} {output_embedding:<60}")
    print(f"{'Cross-entropy':<20} {cross_entropy:<60}")

    activations_total = f"{block_activations} + batch_size × context_length × d_model + batch_size × context_length × vocab_size + batch_size × context_length"
    activations_simplified = "num_layers × (11 × batch_size × context_length × d_model + 2 × batch_size × num_heads × context_length²) + batch_size × context_length × (d_model + vocab_size + 1)"

    print(f"{'TOTAL':<20} {activations_simplified:<60}")

    # 3. GRADIENTS
    print("\n3. GRADIENTS:")
    print("Gradients have the same memory footprint as parameters")
    gradients_total = parameters_simplified
    print(f"{'TOTAL':<20} {gradients_total:<60}")

    # 4. OPTIMIZER STATE (AdamW)
    print("\n4. OPTIMIZER STATE (AdamW):")
    print("AdamW maintains first moment (m) and second moment (v) for each parameter")
    print("Total optimizer state = 2 × parameters")
    optimizer_state = f"2 × ({parameters_simplified})"
    print(f"{'TOTAL':<20} {optimizer_state:<60}")

    # 5. TOTAL MEMORY
    print("\n5. TOTAL PEAK MEMORY:")
    total_memory = f"4 × ({parameters_simplified}) + {activations_simplified}"
    print(f"Parameters + Gradients + Optimizer State: 4 × parameters")
    print(f"Activations: {activations_simplified}")
    print(f"{'TOTAL':<20} {total_memory:<60}")

    return {
        'parameters': parameters_simplified,
        'activations': activations_simplified,
        'gradients': gradients_total,
        'optimizer_state': optimizer_state,
        'total': total_memory
    }


def gpt2_xl_instantiation():
    """
    Problem (b): Instantiate formulas for GPT-2 XL and find maximum batch size.
    """
    print("\n" + "=" * 80)
    print("PROBLEM (b): GPT-2 XL Memory Analysis")
    print("=" * 80)

    # GPT-2 XL hyperparameters
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25  # d_model / num_heads = 64
    d_ff = 4 * d_model

    print(f"GPT-2 XL Hyperparameters:")
    print(f"  vocab_size = {vocab_size:,}")
    print(f"  context_length = {context_length:,}")
    print(f"  num_layers = {num_layers}")
    print(f"  d_model = {d_model:,}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_ff = {d_ff:,}")

    # Calculate parameters (in float32 elements)
    token_emb = vocab_size * d_model
    transformer_params = num_layers * (12 * d_model**2 + 2 * d_model)
    final_norm = d_model
    output_emb = d_model * vocab_size
    total_params = 2 * vocab_size * d_model + transformer_params + final_norm

    print(f"\nParameter Count:")
    print(f"  Token embedding: {token_emb:,}")
    print(f"  Transformer blocks: {transformer_params:,}")
    print(f"  Final RMSNorm: {final_norm:,}")
    print(f"  Output embedding: {output_emb:,}")
    print(f"  Total parameters: {total_params:,}")

    # Memory usage (in bytes, assuming float32 = 4 bytes)
    bytes_per_param = 4

    # Fixed memory (independent of batch_size)
    params_memory = total_params * bytes_per_param
    gradients_memory = total_params * bytes_per_param
    optimizer_memory = 2 * total_params * bytes_per_param
    fixed_memory = params_memory + gradients_memory + optimizer_memory  # 4 × params

    print(f"\nFixed Memory (independent of batch_size):")
    print(f"  Parameters: {params_memory / (1024**3):.2f} GB")
    print(f"  Gradients: {gradients_memory / (1024**3):.2f} GB")
    print(f"  Optimizer state: {optimizer_memory / (1024**3):.2f} GB")
    print(f"  Total fixed: {fixed_memory / (1024**3):.2f} GB")

    # Variable memory (depends on batch_size)
    # Activations per batch element
    block_activations_per_batch = num_layers * (11 * context_length * d_model + 2 * num_heads * context_length**2)
    final_activations_per_batch = context_length * (d_model + vocab_size + 1)
    total_activations_per_batch = block_activations_per_batch + final_activations_per_batch

    variable_memory_per_batch = total_activations_per_batch * bytes_per_param

    print(f"\nVariable Memory (per batch element):")
    print(f"  Block activations: {block_activations_per_batch:,} elements")
    print(f"  Final activations: {final_activations_per_batch:,} elements")
    print(f"  Total per batch: {total_activations_per_batch:,} elements")
    print(f"  Memory per batch: {variable_memory_per_batch / (1024**3):.4f} GB")

    # Express as a × batch_size + b
    a = variable_memory_per_batch / (1024**3)  # GB per batch element
    b = fixed_memory / (1024**3)  # GB fixed

    print(f"\nMemory Formula:")
    print(f"  Total Memory = {a:.4f} × batch_size + {b:.2f} GB")

    # Maximum batch size for 80GB memory
    target_memory_gb = 80
    max_batch_size = int((target_memory_gb - b) / a)

    print(f"\nMaximum Batch Size Analysis:")
    print(f"  Target memory: {target_memory_gb} GB")
    print(f"  Max batch_size = ({target_memory_gb} - {b:.2f}) / {a:.4f} = {max_batch_size}")

    # Verification
    total_memory_at_max = a * max_batch_size + b
    print(f"  Verification: {a:.4f} × {max_batch_size} + {b:.2f} = {total_memory_at_max:.2f} GB")

    return a, b, max_batch_size


def adamw_flops_analysis():
    """
    Problem (c): Calculate FLOPs for one AdamW step.
    """
    print("\n" + "=" * 80)
    print("PROBLEM (c): AdamW FLOPs Analysis")
    print("=" * 80)

    print("AdamW operations per parameter (following Algorithm 1):")
    print("1. First moment update: m ← β₁m + (1 - β₁)g")
    print("   - 1 multiply (β₁m) + 1 multiply-add ((1 - β₁)g) = 3 FLOPs")
    print("2. Second moment update: v ← β₂v + (1 - β₂)g²")
    print("   - 1 multiply (g²) + 1 multiply (β₂v) + 1 multiply-add ((1 - β₂)g²) = 4 FLOPs")
    print("3. Bias correction calculations:")
    print("   - bias_correction1 = 1 - β₁^t: negligible (computed once per step)")
    print("   - bias_correction2 = 1 - β₂^t: negligible (computed once per step)")
    print("4. Parameter update: θ ← θ - αₜ m/(√v + ε)")
    print("   - 1 sqrt (√v) + 1 add (+ε) + 1 divide (m/denom) + 1 multiply (-αₜ) + 1 add = 5 FLOPs")
    print("5. Weight decay: θ ← θ - αλθ")
    print("   - 1 multiply (αλθ) + 1 subtract = 2 FLOPs")

    flops_per_param = 3 + 4 + 5 + 2  # = 14 FLOPs per parameter
    print(f"\nTotal FLOPs per parameter: {flops_per_param}")

    print(f"\nAdamW FLOPs for full model:")
    print(f"Total FLOPs = {flops_per_param} × num_parameters")
    print(f"            = {flops_per_param} × (2 × vocab_size × d_model + num_layers × (12 × d_model² + 2 × d_model) + d_model)")

    return flops_per_param


def training_time_analysis():
    """
    Problem (d): Calculate training time with MFU analysis.
    """
    print("\n" + "=" * 80)
    print("PROBLEM (d): Training Time Analysis with MFU")
    print("=" * 80)

    # Given parameters
    training_steps = 400_000
    batch_size = 1024
    mfu_percent = 50  # 50% MFU
    a100_peak_flops = 19.5e12  # 19.5 teraFLOP/s

    # GPT-2 XL parameters (from part b)
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25

    # Calculate total parameters
    total_params = 2 * vocab_size * d_model + num_layers * (12 * d_model**2 + 2 * d_model) + d_model

    print(f"Training Configuration:")
    print(f"  Training steps: {training_steps:,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Context length: {context_length:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  A100 peak FLOPs: {a100_peak_flops:.1e} FLOP/s")
    print(f"  Model FLOPs Utilization (MFU): {mfu_percent}%")

    # Forward pass FLOPs estimation (following standard approximations)
    # Based on Kaplan et al. 2020 and standard transformer FLOP calculations
    tokens_per_batch = batch_size * context_length

    # More realistic FLOP estimation for transformer forward pass:
    # Based on empirical observations from real transformer training
    # Forward pass computation is dominated by matrix multiplications in attention and FFN
    # Typical estimate: ~2 * N * tokens (more conservative than theoretical 6N)
    forward_flops_per_batch = 2 * total_params * tokens_per_batch

    # Backward pass has 2x FLOPs of forward pass (as stated in problem)
    backward_flops_per_batch = 2 * forward_flops_per_batch

    # AdamW FLOPs (from part c)
    adamw_flops_per_step = 14 * total_params

    # Total FLOPs per training step
    total_flops_per_step = forward_flops_per_batch + backward_flops_per_batch + adamw_flops_per_step

    print(f"\nFLOPs Breakdown per Training Step:")
    print(f"  Forward pass: {forward_flops_per_batch:.2e}")
    print(f"  Backward pass: {backward_flops_per_batch:.2e}")
    print(f"  AdamW optimizer: {adamw_flops_per_step:.2e}")
    print(f"  Total per step: {total_flops_per_step:.2e}")

    # Total FLOPs for entire training
    total_training_flops = total_flops_per_step * training_steps

    print(f"\nTotal Training FLOPs:")
    print(f"  {total_training_flops:.2e} FLOPs")

    # Effective throughput with MFU
    effective_flops_per_second = a100_peak_flops * (mfu_percent / 100)

    # Training time calculation
    training_time_seconds = total_training_flops / effective_flops_per_second
    training_time_hours = training_time_seconds / 3600
    training_time_days = training_time_hours / 24

    print(f"\nTraining Time Calculation:")
    print(f"  Effective FLOP/s: {effective_flops_per_second:.2e} FLOP/s")
    print(f"  Training time: {training_time_seconds:.2e} seconds")
    print(f"  Training time: {training_time_hours:.1f} hours")
    print(f"  Training time: {training_time_days:.1f} days")

    # Tokens processed per second
    total_tokens = training_steps * tokens_per_batch
    tokens_per_second = total_tokens / training_time_seconds

    print(f"\nThroughput Analysis:")
    print(f"  Total tokens: {total_tokens:.2e}")
    print(f"  Tokens per second: {tokens_per_second:.0f}")

    print(f"\nJustification:")
    print(f"The calculation follows empirical transformer training practices:")
    print(f"1. Forward pass FLOPs ≈ 2 × parameters × tokens (matrix multiplication dominated)")
    print(f"2. Backward pass FLOPs = 2 × forward pass FLOPs (gradient computation)")
    print(f"3. AdamW adds {14} FLOPs per parameter (moment updates + parameter updates)")
    print(f"4. MFU accounts for real-world efficiency vs theoretical peak")

    return training_time_days


def main():
    """Run complete AdamW resource accounting analysis."""
    print("AdamW Resource Accounting Analysis")
    print("Following CS336 Assignment Requirements")

    # Problem (a): Memory analysis
    memory_formulas = analyze_adamw_memory()

    # Problem (b): GPT-2 XL instantiation
    a, b, max_batch_size = gpt2_xl_instantiation()

    # Problem (c): FLOPs analysis
    flops_per_param = adamw_flops_analysis()

    # Problem (d): Training time
    training_days = training_time_analysis()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    print(f"\n(a) Memory Formulas:")
    print(f"  Parameters: {memory_formulas['parameters']}")
    print(f"  Activations: {memory_formulas['activations']}")
    print(f"  Gradients: {memory_formulas['gradients']}")
    print(f"  Optimizer state: {memory_formulas['optimizer_state']}")
    print(f"  Total: {memory_formulas['total']}")

    print(f"\n(b) GPT-2 XL Memory:")
    print(f"  Formula: {a:.4f} × batch_size + {b:.2f} GB")
    print(f"  Maximum batch size (80GB): {max_batch_size}")

    print(f"\n(c) AdamW FLOPs:")
    print(f"  FLOPs per parameter: {flops_per_param}")
    print(f"  Total: {flops_per_param} × num_parameters")

    print(f"\n(d) Training Time:")
    print(f"  GPT-2 XL, 400K steps, batch_size=1024, 50% MFU: {training_days:.1f} days")


if __name__ == "__main__":
    main()