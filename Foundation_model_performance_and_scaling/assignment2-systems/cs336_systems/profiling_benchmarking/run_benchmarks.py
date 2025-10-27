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
    timeout: int = 600,  # 10 minutes default
) -> dict:
    """
    Run a single benchmark and parse the results.

    Returns:
        Dictionary with benchmark results
    """
    cmd = [
        "uv", "run", "python", "-m", "cs336_systems.profiling_benchmarking.benchmark",
        "--model-size", model_size,
        "--context-length", str(context_length),
        "--batch-size", str(batch_size),
        "--warmup-steps", str(warmup_steps),
        "--measure-steps", str(measure_steps),
        "--pass-type", pass_type,
        "--device", device,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Benchmark timed out after {timeout}s")
        return {
            "model_size": model_size,
            "context_length": context_length,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "pass_type": pass_type,
            "num_params": None,
            "mean_time_ms": None,
            "std_time_ms": None,
            "cv_percent": None,
            "error": "timeout",
        }

    # Check if command failed
    if result.returncode != 0:
        # Check for OOM error
        error_type = "failed"
        if "out of memory" in result.stderr.lower() or "oom" in result.stderr.lower():
            error_type = "OOM"
            print(f"  WARNING: Out of memory - model too large for GPU")
        else:
            print(f"  WARNING: Benchmark failed with return code {result.returncode}")
            print(f"  STDERR: {result.stderr[:200]}")

        return {
            "model_size": model_size,
            "context_length": context_length,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "pass_type": pass_type,
            "num_params": None,
            "mean_time_ms": None,
            "std_time_ms": None,
            "cv_percent": None,
            "error": error_type,
        }

    # Parse output
    lines = result.stdout.strip().split("\n")

    mean_time = None
    std_time = None
    num_params = None

    for line in lines:
        if "Mean time:" in line:
            try:
                # Extract mean time in ms
                parts = line.split()
                mean_time = float(parts[2])  # in ms
            except (IndexError, ValueError) as e:
                print(f"  WARNING: Failed to parse mean time from: {line}")
        elif "Std dev:" in line:
            try:
                # Extract std dev in ms
                parts = line.split()
                std_time = float(parts[2])  # in ms
            except (IndexError, ValueError) as e:
                print(f"  WARNING: Failed to parse std dev from: {line}")
        elif "Model has" in line:
            try:
                # Extract number of parameters
                parts = line.split()
                num_params_str = parts[2].replace(",", "")
                num_params = int(num_params_str)
            except (IndexError, ValueError) as e:
                print(f"  WARNING: Failed to parse num params from: {line}")

    # Check if we got valid results
    if mean_time is None or std_time is None:
        print(f"  WARNING: Failed to parse benchmark results")
        print(f"  Output preview: {result.stdout[:500]}")

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
        "error": None,
    }


def run_benchmark_sweep(
    model_sizes: list[str],
    context_length: int = 512,
    batch_size: int = 4,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    pass_type: str = "forward_backward",
    device: str = "cuda",
    timeout: int = 600,
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
            timeout=timeout,
        )
        results.append(result)

        # Print results if available
        if result['mean_time_ms'] is not None and result['std_time_ms'] is not None:
            print(f"  Mean: {result['mean_time_ms']:.2f} ms, Std: {result['std_time_ms']:.2f} ms")
        else:
            print(f"  Failed: {result.get('error', 'unknown error')}")

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
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each benchmark (default: 600)",
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
        timeout=args.timeout,
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
