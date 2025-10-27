"""
Helper script to benchmark large models with reduced settings to avoid OOM.

For XL and 2.7B models, we need to reduce:
- Context length
- Batch size
Or use gradient checkpointing / mixed precision
"""

import argparse
import subprocess
import pandas as pd


def benchmark_with_reduced_settings(model_size: str) -> dict:
    """
    Benchmark a model with settings optimized to avoid OOM.
    """
    # Recommended settings for different model sizes on consumer GPUs
    settings = {
        "small": {"ctx": 512, "bs": 4},
        "medium": {"ctx": 512, "bs": 4},
        "large": {"ctx": 512, "bs": 4},
        "xl": {"ctx": 256, "bs": 2},      # Reduced for 16GB VRAM
        "2.7B": {"ctx": 128, "bs": 1},    # Reduced for 16GB VRAM
    }

    config = settings.get(model_size, {"ctx": 256, "bs": 2})

    cmd = [
        "uv", "run", "python", "-m", "cs336_systems.profiling_benchmarking.benchmark",
        "--model-size", model_size,
        "--context-length", str(config["ctx"]),
        "--batch-size", str(config["bs"]),
        "--warmup-steps", "5",
        "--measure-steps", "10",
        "--pass-type", "forward_backward",
    ]

    print(f"Benchmarking {model_size} with ctx={config['ctx']}, bs={config['bs']}...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Failed: {result.stderr[:200]}")
        return None

    # Parse results
    lines = result.stdout.split("\n")
    mean_time = None
    std_time = None

    for line in lines:
        if "Mean time:" in line:
            parts = line.split()
            mean_time = float(parts[2])
        elif "Std dev:" in line:
            parts = line.split()
            std_time = float(parts[2])

    return {
        "model_size": model_size,
        "context_length": config["ctx"],
        "batch_size": config["bs"],
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark large models with OOM-safe settings"
    )
    parser.add_argument(
        "--model-sizes",
        type=str,
        nargs="+",
        default=["xl", "2.7B"],
        help="Model sizes to benchmark",
    )

    args = parser.parse_args()

    results = []
    for model_size in args.model_sizes:
        result = benchmark_with_reduced_settings(model_size)
        if result:
            results.append(result)
            print(f"  Success: {result['mean_time_ms']:.2f} ms")
        else:
            print(f"  Failed to benchmark {model_size}")

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("Large Model Benchmark Results (Reduced Settings)")
        print("=" * 80)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
