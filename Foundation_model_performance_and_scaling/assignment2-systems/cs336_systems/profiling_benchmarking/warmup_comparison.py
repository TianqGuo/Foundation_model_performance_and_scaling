"""
Script to compare benchmark results with different warmup settings.

This helps answer assignment part (c) about the effect of warmup steps.
"""

import pandas as pd
import torch
from cs336_systems.profiling_benchmarking.benchmark import (
    MODEL_CONFIGS,
    create_model,
    generate_random_batch,
    benchmark_forward_backward,
)


def benchmark_with_warmup(model_size: str, warmup_steps: int, measure_steps: int = 10):
    """Run benchmark with specific warmup setting."""
    config = MODEL_CONFIGS[model_size]
    context_length = 512
    batch_size = 4

    print(f"Warmup={warmup_steps}: ", end="", flush=True)

    # Create model
    model = create_model(config, context_length, device="cuda")
    input_ids = generate_random_batch(batch_size, context_length, config.vocab_size, device="cuda")

    # Run benchmark
    mean_time, std_time = benchmark_forward_backward(
        model, input_ids, warmup_steps=warmup_steps, measure_steps=measure_steps
    )

    # Cleanup
    del model, input_ids
    torch.cuda.empty_cache()

    mean_ms = mean_time * 1000
    std_ms = std_time * 1000

    print(f"Mean={mean_ms:.2f}ms, Std={std_ms:.2f}ms, CV={std_ms/mean_ms*100:.2f}%")

    return {
        "warmup_steps": warmup_steps,
        "mean_time_ms": mean_ms,
        "std_time_ms": std_ms,
        "cv_percent": std_ms / mean_ms * 100,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare different warmup settings")
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl", "2.7B"],
        help="Model size to test",
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
        default=None,
        help="Output CSV file",
    )

    args = parser.parse_args()

    print(f"Testing warmup effect on {args.model_size} model")
    print(f"Measurement steps: {args.measure_steps}")
    print()

    # Test different warmup settings
    warmup_settings = [0, 1, 2, 5, 10]
    results = []

    for warmup in warmup_settings:
        result = benchmark_with_warmup(args.model_size, warmup, args.measure_steps)
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("Warmup Comparison Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Calculate differences from baseline (warmup=5)
    baseline_idx = df[df["warmup_steps"] == 5].index[0]
    baseline_mean = df.loc[baseline_idx, "mean_time_ms"]

    print("Difference from warmup=5 baseline:")
    for idx, row in df.iterrows():
        diff_ms = row["mean_time_ms"] - baseline_mean
        diff_pct = (diff_ms / baseline_mean) * 100
        print(f"  Warmup={row['warmup_steps']}: {diff_ms:+.2f}ms ({diff_pct:+.2f}%)")

    print()

    # Markdown table
    print("Markdown Table:")
    print(df.to_markdown(index=False))

    # Save if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
