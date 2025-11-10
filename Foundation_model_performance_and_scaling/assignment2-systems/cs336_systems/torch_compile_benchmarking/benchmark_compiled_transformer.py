"""
Benchmark torch.compile performance on full Transformer models.

This script compares vanilla PyTorch Transformer with torch.compile optimized version
to measure the impact of JIT compilation on end-to-end model performance.

Usage:
    # Benchmark specific model size
    python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_transformer \
        --model-size small --context-length 512

    # Benchmark all configurations
    python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_transformer \
        --all-configs

    # Save results to CSV
    python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_transformer \
        --all-configs --output results/torch_compile_benchmarking/compiled_transformer_benchmark.csv
"""

import argparse
import timeit
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from cs336_systems.profiling_benchmarking.benchmark import (
    MODEL_CONFIGS,
    ModelConfig,
    create_model,
    generate_random_batch,
)


def benchmark_model_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    pass_type: str = "forward",
    warmup_steps: int = 5,
    measure_steps: int = 10,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    """
    Benchmark a specific pass type for a model.

    Args:
        model: Model to benchmark
        input_ids: Input token IDs
        pass_type: Type of pass - "forward", "forward_backward", or "full_step"
        warmup_steps: Number of warmup iterations
        measure_steps: Number of measurement iterations
        optimizer: Optimizer (required for "full_step")

    Returns:
        Tuple of (mean_time_seconds, std_time_seconds)
    """
    if pass_type == "forward":
        model.eval()
    else:
        model.train()

    # Warmup
    for _ in range(warmup_steps):
        if pass_type == "forward":
            with torch.no_grad():
                _ = model(input_ids)
        elif pass_type == "forward_backward":
            model.zero_grad()
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
        elif pass_type == "full_step":
            if optimizer is None:
                raise ValueError("Optimizer required for full_step")
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(measure_steps):
        start = timeit.default_timer()

        if pass_type == "forward":
            with torch.no_grad():
                _ = model(input_ids)
        elif pass_type == "forward_backward":
            model.zero_grad()
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
        elif pass_type == "full_step":
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_compiled_vs_vanilla_transformer(
    model_size: str,
    context_length: int,
    batch_size: int = 4,
    warmup_steps: int = 5,
    measure_steps: int = 10,
) -> dict:
    """
    Benchmark a Transformer model comparing vanilla vs compiled versions.

    Args:
        model_size: Model size name (small, medium, large, xl, 2.7B)
        context_length: Context length (sequence length)
        batch_size: Batch size
        warmup_steps: Number of warmup iterations
        measure_steps: Number of measurement iterations

    Returns:
        Dictionary with benchmark results
    """
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Get config
        config = MODEL_CONFIGS[model_size]

        # Create vanilla model
        print(f"\nCreating vanilla model ({model_size}, ctx={context_length})...")
        model_vanilla = create_model(config, context_length, device="cuda")
        num_params = sum(p.numel() for p in model_vanilla.parameters())

        # Create compiled model (fresh instance)
        print(f"Creating compiled model...")
        model_compiled = create_model(config, context_length, device="cuda")
        model_compiled = torch.compile(model_compiled)

        # Create optimizers for full step benchmarking
        optimizer_vanilla = torch.optim.AdamW(model_vanilla.parameters(), lr=1e-4)
        optimizer_compiled = torch.optim.AdamW(model_compiled.parameters(), lr=1e-4)

        # Generate input batch
        input_ids = generate_random_batch(
            batch_size, context_length, config.vocab_size, device="cuda"
        )

        results = {
            "model_size": model_size,
            "context_length": context_length,
            "batch_size": batch_size,
            "num_params": num_params,
            "num_params_millions": num_params / 1e6,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
        }

        # Benchmark forward pass
        print("  Benchmarking forward pass (vanilla)...")
        fwd_mean_v, fwd_std_v = benchmark_model_pass(
            model_vanilla, input_ids, "forward", warmup_steps, measure_steps
        )
        results["forward_vanilla_ms"] = fwd_mean_v * 1000
        results["forward_vanilla_std_ms"] = fwd_std_v * 1000

        print("  Benchmarking forward pass (compiled)...")
        fwd_mean_c, fwd_std_c = benchmark_model_pass(
            model_compiled, input_ids, "forward", warmup_steps, measure_steps
        )
        results["forward_compiled_ms"] = fwd_mean_c * 1000
        results["forward_compiled_std_ms"] = fwd_std_c * 1000
        results["forward_speedup"] = fwd_mean_v / fwd_mean_c

        # Benchmark forward+backward
        print("  Benchmarking forward+backward (vanilla)...")
        fwdbwd_mean_v, fwdbwd_std_v = benchmark_model_pass(
            model_vanilla, input_ids, "forward_backward", warmup_steps, measure_steps
        )
        results["fwd_bwd_vanilla_ms"] = fwdbwd_mean_v * 1000
        results["fwd_bwd_vanilla_std_ms"] = fwdbwd_std_v * 1000

        print("  Benchmarking forward+backward (compiled)...")
        fwdbwd_mean_c, fwdbwd_std_c = benchmark_model_pass(
            model_compiled, input_ids, "forward_backward", warmup_steps, measure_steps
        )
        results["fwd_bwd_compiled_ms"] = fwdbwd_mean_c * 1000
        results["fwd_bwd_compiled_std_ms"] = fwdbwd_std_c * 1000
        results["fwd_bwd_speedup"] = fwdbwd_mean_v / fwdbwd_mean_c

        # Benchmark full step (forward+backward+optimizer)
        print("  Benchmarking full step (vanilla)...")
        full_mean_v, full_std_v = benchmark_model_pass(
            model_vanilla, input_ids, "full_step", warmup_steps, measure_steps, optimizer_vanilla
        )
        results["full_step_vanilla_ms"] = full_mean_v * 1000
        results["full_step_vanilla_std_ms"] = full_std_v * 1000

        print("  Benchmarking full step (compiled)...")
        full_mean_c, full_std_c = benchmark_model_pass(
            model_compiled, input_ids, "full_step", warmup_steps, measure_steps, optimizer_compiled
        )
        results["full_step_compiled_ms"] = full_mean_c * 1000
        results["full_step_compiled_std_ms"] = full_std_c * 1000
        results["full_step_speedup"] = full_mean_v / full_mean_c

        results["status"] = "success"

        print(f"  ✓ Complete")
        print(f"    Forward: {fwd_mean_v*1000:.2f}ms → {fwd_mean_c*1000:.2f}ms (speedup: {results['forward_speedup']:.2f}x)")
        print(f"    Fwd+Bwd: {fwdbwd_mean_v*1000:.2f}ms → {fwdbwd_mean_c*1000:.2f}ms (speedup: {results['fwd_bwd_speedup']:.2f}x)")
        print(f"    Full step: {full_mean_v*1000:.2f}ms → {full_mean_c*1000:.2f}ms (speedup: {results['full_step_speedup']:.2f}x)")

        return results

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return {
                "model_size": model_size,
                "context_length": context_length,
                "batch_size": batch_size,
                "status": "OOM",
                "error": "Out of memory",
            }
        else:
            raise


def run_all_configurations(
    warmup_steps: int = 5,
    measure_steps: int = 10,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run benchmarks across all model sizes and context lengths.

    Args:
        warmup_steps: Number of warmup iterations
        measure_steps: Number of measurement iterations
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with all results
    """
    # Test all model sizes from §1.1.2 as required by §1.3.1(b)
    # "same model/context configurations you benchmarked previously"
    model_sizes = ["small", "medium", "large", "xl", "2.7B"]

    # Use same context lengths as §1.1.3
    # Adjust based on what you actually benchmarked in §1.1.3
    context_lengths = [128, 256, 512, 1024]

    print("=" * 80)
    print("Torch.compile Transformer Benchmarking - Section 1.3.1(b)")
    print("=" * 80)
    print(f"Model sizes: {model_sizes}")
    print(f"Context lengths: {context_lengths}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Measure steps: {measure_steps}")
    print()

    results = []

    for model_size in model_sizes:
        for context_length in context_lengths:
            result = benchmark_compiled_vs_vanilla_transformer(
                model_size=model_size,
                context_length=context_length,
                batch_size=4,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
            )
            results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # Print summary tables
    print_comparison_tables(df)

    return df


def print_comparison_tables(df: pd.DataFrame) -> None:
    """Print comparison tables for compiled vs vanilla Transformer."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLES: COMPILED vs VANILLA TRANSFORMER")
    print("=" * 80)
    print()

    # Filter successful results
    df_success = df[df["status"] == "success"].copy()

    if df_success.empty:
        print("No successful benchmarks to display.")
        return

    # Forward pass comparison
    print("Forward Pass Timing (ms)")
    print("-" * 80)
    print("\nVanilla:")
    pivot_vanilla_fwd = df_success.pivot_table(
        index="model_size", columns="context_length", values="forward_vanilla_ms", aggfunc="first"
    )
    print(pivot_vanilla_fwd.to_string())

    print("\nCompiled:")
    pivot_compiled_fwd = df_success.pivot_table(
        index="model_size", columns="context_length", values="forward_compiled_ms", aggfunc="first"
    )
    print(pivot_compiled_fwd.to_string())

    print("\nSpeedup (Vanilla / Compiled):")
    pivot_speedup_fwd = df_success.pivot_table(
        index="model_size", columns="context_length", values="forward_speedup", aggfunc="first"
    )
    print(pivot_speedup_fwd.to_string())

    # Forward+Backward comparison
    print("\n\nForward+Backward Timing (ms)")
    print("-" * 80)
    print("\nVanilla:")
    pivot_vanilla_fwdbwd = df_success.pivot_table(
        index="model_size", columns="context_length", values="fwd_bwd_vanilla_ms", aggfunc="first"
    )
    print(pivot_vanilla_fwdbwd.to_string())

    print("\nCompiled:")
    pivot_compiled_fwdbwd = df_success.pivot_table(
        index="model_size", columns="context_length", values="fwd_bwd_compiled_ms", aggfunc="first"
    )
    print(pivot_compiled_fwdbwd.to_string())

    print("\nSpeedup (Vanilla / Compiled):")
    pivot_speedup_fwdbwd = df_success.pivot_table(
        index="model_size", columns="context_length", values="fwd_bwd_speedup", aggfunc="first"
    )
    print(pivot_speedup_fwdbwd.to_string())

    # Full step comparison
    print("\n\nFull Training Step Timing (ms) [Forward+Backward+Optimizer]")
    print("-" * 80)
    print("\nVanilla:")
    pivot_vanilla_full = df_success.pivot_table(
        index="model_size", columns="context_length", values="full_step_vanilla_ms", aggfunc="first"
    )
    print(pivot_vanilla_full.to_string())

    print("\nCompiled:")
    pivot_compiled_full = df_success.pivot_table(
        index="model_size", columns="context_length", values="full_step_compiled_ms", aggfunc="first"
    )
    print(pivot_compiled_full.to_string())

    print("\nSpeedup (Vanilla / Compiled):")
    pivot_speedup_full = df_success.pivot_table(
        index="model_size", columns="context_length", values="full_step_speedup", aggfunc="first"
    )
    print(pivot_speedup_full.to_string())

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark compiled vs uncompiled Transformer models"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size to benchmark (use --all-configs for all)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Context length (use --all-configs for multiple)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Benchmark all model sizes and context lengths",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=10,
        help="Number of measurement iterations (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/torch_compile_benchmarking/compiled_transformer_benchmark.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return

    # Check torch.compile availability
    if not hasattr(torch, "compile"):
        print("ERROR: torch.compile not available. Requires PyTorch 2.0+")
        return

    if args.all_configs:
        # Run all configurations
        df = run_all_configurations(
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            output_path=args.output,
        )
    else:
        # Single configuration
        if args.model_size is None or args.context_length is None:
            print("ERROR: Must specify --model-size and --context-length, or use --all-configs")
            return

        result = benchmark_compiled_vs_vanilla_transformer(
            model_size=args.model_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )

        # Convert to DataFrame for display
        df = pd.DataFrame([result])
        print_comparison_tables(df)

        # Save if output specified
        if args.output:
            output_file = Path(args.output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ Benchmarking complete")
    print()
    print("Key observations:")
    print("  - torch.compile optimizes entire model computation graph")
    print("  - Speedups vary by model size and operation type")
    print("  - Forward pass may see different speedups than backward")
    print("  - Optimizer step overhead is similar for both versions")
    print()


if __name__ == "__main__":
    main()
