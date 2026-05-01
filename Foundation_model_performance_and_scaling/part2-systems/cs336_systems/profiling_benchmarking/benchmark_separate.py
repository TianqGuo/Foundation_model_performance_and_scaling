"""
Script to benchmark forward and backward passes separately for assignment part (b).

This script measures:
1. Forward pass time
2. Backward pass time
3. Total (forward + backward) time

For assignment deliverable.
"""

import argparse
import pandas as pd
import torch
from pathlib import Path

from cs336_systems.profiling_benchmarking.benchmark import (
    MODEL_CONFIGS,
    create_model,
    generate_random_batch,
    benchmark_forward,
    benchmark_backward,
)


def benchmark_model_separate(
    model_size: str,
    context_length: int = 512,
    batch_size: int = 4,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    device: str = "cuda",
) -> dict:
    """
    Benchmark a model with separate forward and backward pass measurements.

    Args:
        model_size: Model size to benchmark
        context_length: Sequence length
        batch_size: Batch size
        warmup_steps: Number of warmup steps
        measure_steps: Number of measurement steps
        device: Device to use

    Returns:
        Dictionary with benchmark results
    """
    config = MODEL_CONFIGS[model_size]
    config.batch_size = batch_size

    print(f"\nBenchmarking {model_size} model...")

    try:
        # Create model
        model = create_model(config, context_length, device=device)
        num_params = sum(p.numel() for p in model.parameters())

        # Generate batch
        input_ids = generate_random_batch(
            batch_size, context_length, config.vocab_size, device=device
        )

        # Benchmark forward pass
        print("  Measuring forward pass...")
        fwd_mean, fwd_std = benchmark_forward(
            model, input_ids, warmup_steps, measure_steps
        )

        # Benchmark backward pass
        print("  Measuring backward pass...")
        bwd_mean, bwd_std = benchmark_backward(
            model, input_ids, warmup_steps, measure_steps
        )

        # Convert to milliseconds
        fwd_mean_ms = fwd_mean * 1000
        fwd_std_ms = fwd_std * 1000
        bwd_mean_ms = bwd_mean * 1000
        bwd_std_ms = bwd_std * 1000
        total_mean_ms = (fwd_mean + bwd_mean) * 1000

        print(f"  Forward:  {fwd_mean_ms:.2f} ± {fwd_std_ms:.2f} ms")
        print(f"  Backward: {bwd_mean_ms:.2f} ± {bwd_std_ms:.2f} ms")
        print(f"  Total:    {total_mean_ms:.2f} ms")
        print(f"  Ratio (backward/forward): {bwd_mean / fwd_mean:.2f}x")

        # Clean up
        del model
        del input_ids
        torch.cuda.empty_cache()

        return {
            "model_size": model_size,
            "num_params": num_params,
            "context_length": context_length,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "forward_mean_ms": fwd_mean_ms,
            "forward_std_ms": fwd_std_ms,
            "forward_cv_percent": (fwd_std / fwd_mean) * 100,
            "backward_mean_ms": bwd_mean_ms,
            "backward_std_ms": bwd_std_ms,
            "backward_cv_percent": (bwd_std / bwd_mean) * 100,
            "total_mean_ms": total_mean_ms,
            "backward_forward_ratio": bwd_mean / fwd_mean,
            "error": None,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  WARNING: Out of memory")
            error_type = "OOM"
        else:
            print(f"  ERROR: {str(e)[:200]}")
            error_type = "runtime_error"

        # Try to clean up, but don't fail if empty_cache itself fails
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass  # Ignore if empty_cache fails due to OOM

        return {
            "model_size": model_size,
            "num_params": None,
            "context_length": context_length,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "forward_mean_ms": None,
            "forward_std_ms": None,
            "forward_cv_percent": None,
            "backward_mean_ms": None,
            "backward_std_ms": None,
            "backward_cv_percent": None,
            "total_mean_ms": None,
            "backward_forward_ratio": None,
            "error": error_type,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark forward and backward passes separately"
    )
    parser.add_argument(
        "--model-sizes",
        type=str,
        nargs="+",
        default=["small", "medium", "large"],
        help="Model sizes to benchmark",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=10,
        help="Number of measurement steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/profiling_benchmarking_separate.csv",
        help="Output CSV file path (default: results/profiling_benchmarking_separate.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("Benchmarking Forward and Backward Passes Separately")
    print("=" * 60)
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Measure steps: {args.measure_steps}")
    print(f"Device: {args.device}")
    print()

    # Run benchmarks
    results = []
    for model_size in args.model_sizes:
        result = benchmark_model_separate(
            model_size=model_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            device=args.device,
        )
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 60)
    print("Summary Results")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    # Create a simplified table for the writeup
    print("=" * 60)
    print("Table for Writeup (Part b)")
    print("=" * 60)

    # Filter successful results
    successful = df[df["error"].isna()].copy()

    if len(successful) > 0:
        writeup_df = successful[[
            "model_size",
            "num_params",
            "forward_mean_ms",
            "forward_std_ms",
            "backward_mean_ms",
            "backward_std_ms",
            "total_mean_ms",
        ]].copy()

        # Format parameter count
        writeup_df["num_params"] = writeup_df["num_params"].apply(
            lambda x: f"{x/1e6:.1f}M" if pd.notna(x) else "N/A"
        )

        # Rename columns for clarity
        writeup_df.columns = [
            "Model",
            "Parameters",
            "Forward (ms)",
            "Forward Std (ms)",
            "Backward (ms)",
            "Backward Std (ms)",
            "Total (ms)",
        ]

        print(writeup_df.to_markdown(index=False))
        print()

        # Print answer for part (b)
        print("=" * 60)
        print("Answer for Part (b)")
        print("=" * 60)
        for _, row in successful.iterrows():
            fwd = row["forward_mean_ms"]
            fwd_std = row["forward_std_ms"]
            bwd = row["backward_mean_ms"]
            bwd_std = row["backward_std_ms"]
            cv_fwd = row["forward_cv_percent"]
            cv_bwd = row["backward_cv_percent"]

            print(f"\n{row['model_size']} model ({row['num_params']/1e6:.1f}M parameters):")
            print(f"  Forward pass:  {fwd:.2f} ± {fwd_std:.2f} ms (CV: {cv_fwd:.2f}%)")
            print(f"  Backward pass: {bwd:.2f} ± {bwd_std:.2f} ms (CV: {cv_bwd:.2f}%)")
            print(f"  Ratio: Backward is {row['backward_forward_ratio']:.2f}x forward")

        print()
        print("Variability Analysis:")
        avg_cv = successful[["forward_cv_percent", "backward_cv_percent"]].mean().mean()
        if avg_cv < 5:
            print(f"  Low variability (avg CV: {avg_cv:.2f}%) - measurements are stable and reliable.")
        elif avg_cv < 10:
            print(f"  Moderate variability (avg CV: {avg_cv:.2f}%) - acceptable for benchmarking.")
        else:
            print(f"  High variability (avg CV: {avg_cv:.2f}%) - consider more warmup/measure steps.")

    print()

    # Save to CSV
    if args.output:
        # Ensure the output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(args.output, index=False)
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
