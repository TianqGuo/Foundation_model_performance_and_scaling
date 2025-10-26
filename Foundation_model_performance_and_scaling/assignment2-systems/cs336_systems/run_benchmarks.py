"""
Helper script to run benchmark sweeps across multiple model sizes and configurations.

This script automates running benchmarks for all model sizes and can generate
tables for the writeup.
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path


def run_single_benchmark(
    model_size: str,
    context_length: int,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    pass_type: str = "forward_backward",
    device: str = "cuda",
) -> dict:
    """
    Run a single benchmark and parse the results.

    Returns:
        Dictionary with benchmark results
    """
    cmd = [
        "uv", "run", "python", "-m", "cs336_systems.benchmark",
        "--model-size", model_size,
        "--context-length", str(context_length),
        "--batch-size", str(batch_size),
        "--warmup-steps", str(warmup_steps),
        "--measure-steps", str(measure_steps),
        "--pass-type", pass_type,
        "--device", device,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output
    lines = result.stdout.strip().split("\n")

    mean_time = None
    std_time = None
    num_params = None

    for line in lines:
        if "Mean time:" in line:
            # Extract mean time in ms
            parts = line.split()
            mean_time = float(parts[2])  # in ms
        elif "Std dev:" in line:
            # Extract std dev in ms
            parts = line.split()
            std_time = float(parts[2])  # in ms
        elif "Model has" in line:
            # Extract number of parameters
            parts = line.split()
            num_params_str = parts[2].replace(",", "")
            num_params = int(num_params_str)

    return {
        "model_size": model_size,
        "context_length": context_length,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "pass_type": pass_type,
        "num_params": num_params,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "cv_percent": (std_time / mean_time * 100) if mean_time and std_time else None,
    }


def run_benchmark_sweep(
    model_sizes: list[str],
    context_length: int = 512,
    batch_size: int = 4,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    pass_type: str = "forward_backward",
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run benchmarks for multiple model sizes.

    Returns:
        DataFrame with results
    """
    results = []

    for model_size in model_sizes:
        print(f"Running benchmark for {model_size} model...")
        result = run_single_benchmark(
            model_size=model_size,
            context_length=context_length,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            pass_type=pass_type,
            device=device,
        )
        results.append(result)
        print(f"  Mean: {result['mean_time_ms']:.2f} ms, Std: {result['std_time_ms']:.2f} ms")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark sweeps")
    parser.add_argument(
        "--model-sizes",
        type=str,
        nargs="+",
        default=["small", "medium", "large", "xl", "2.7B"],
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
        "--pass-type",
        type=str,
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help="Type of pass to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    # Run sweep
    df = run_benchmark_sweep(
        model_sizes=args.model_sizes,
        context_length=args.context_length,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        pass_type=args.pass_type,
        device=args.device,
    )

    # Display results
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Generate markdown table
    print("Markdown Table:")
    print(df[["model_size", "num_params", "mean_time_ms", "std_time_ms", "cv_percent"]].to_markdown(index=False))
    print()

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
