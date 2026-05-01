"""
Direct benchmarking script that doesn't use subprocess.

This avoids the subprocess overhead and timeout issues.
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
    benchmark_forward_backward,
)


def run_benchmark_direct(
    model_size: str,
    context_length: int = 512,
    batch_size: int = 4,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    pass_type: str = "forward_backward",
    device: str = "cuda",
) -> dict:
    """
    Run a benchmark directly without subprocess.
    """
    config = MODEL_CONFIGS[model_size]
    config.batch_size = batch_size

    print(f"Benchmarking {model_size} model...")

    try:
        # Create model
        model = create_model(config, context_length, device=device)
        num_params = sum(p.numel() for p in model.parameters())

        # Generate batch
        input_ids = generate_random_batch(
            batch_size, context_length, config.vocab_size, device=device
        )

        # Run benchmark
        if pass_type == "forward":
            mean_time, std_time = benchmark_forward(
                model, input_ids, warmup_steps, measure_steps
            )
        else:
            mean_time, std_time = benchmark_forward_backward(
                model, input_ids, warmup_steps, measure_steps
            )

        # Convert to milliseconds
        mean_time_ms = mean_time * 1000
        std_time_ms = std_time * 1000

        print(f"  Mean: {mean_time_ms:.2f} ms, Std: {std_time_ms:.2f} ms")

        # Clean up
        del model
        del input_ids
        torch.cuda.empty_cache()

        return {
            "model_size": model_size,
            "context_length": context_length,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "pass_type": pass_type,
            "num_params": num_params,
            "mean_time_ms": mean_time_ms,
            "std_time_ms": std_time_ms,
            "cv_percent": (std_time_ms / mean_time_ms * 100),
            "error": None,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  WARNING: Out of memory - model too large for GPU")
            error_type = "OOM"
        else:
            print(f"  ERROR: {str(e)[:200]}")
            error_type = "runtime_error"

        torch.cuda.empty_cache()

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


def main():
    parser = argparse.ArgumentParser(description="Run benchmark sweeps directly")
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
        "--pass-type",
        type=str,
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help="Type of pass to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/profiling_benchmarking_direct.csv",
        help="Output CSV file path (default: results/profiling_benchmarking_direct.csv)",
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

    print(f"Running benchmarks with:")
    print(f"  Context length: {args.context_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Measure steps: {args.measure_steps}")
    print(f"  Pass type: {args.pass_type}")
    print()

    # Run benchmarks
    results = []
    for model_size in args.model_sizes:
        result = run_benchmark_direct(
            model_size=model_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            pass_type=args.pass_type,
            device=args.device,
        )
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Generate markdown table
    print("Markdown Table:")
    print(
        df[["model_size", "num_params", "mean_time_ms", "std_time_ms", "cv_percent"]].to_markdown(
            index=False
        )
    )
    print()

    # Save to CSV if requested
    if args.output:
        # Ensure the output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
