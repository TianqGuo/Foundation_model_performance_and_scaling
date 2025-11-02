"""
Benchmark Transformer models with mixed precision (BF16) support.

This script extends the standard benchmarking to compare FP32 vs BF16 performance
across different model sizes.

Usage:
    # Benchmark with FP32 (default)
    python -m cs336_systems.mixed_precision.benchmark_mixed_precision \
        --model-size small --context-length 512

    # Benchmark with BF16 mixed precision
    python -m cs336_systems.mixed_precision.benchmark_mixed_precision \
        --model-size small --context-length 512 --use-mixed-precision

    # Benchmark all model sizes
    python -m cs336_systems.mixed_precision.benchmark_mixed_precision \
        --all-models --context-length 512 --output results/mixed_precision/mixed_precision_benchmark.csv
"""

import argparse
import timeit
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from cs336_systems.profiling_benchmarking.benchmark import (
    MODEL_CONFIGS,
    create_model,
    generate_random_batch,
)


def benchmark_with_mixed_precision(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    use_mixed_precision: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, float, float, float]:
    """
    Benchmark forward and backward passes with optional mixed precision.

    Args:
        model: Model to benchmark
        input_ids: Input token IDs
        warmup_steps: Number of warmup steps before timing
        measure_steps: Number of steps to measure
        use_mixed_precision: Whether to use mixed precision (autocast)
        dtype: Data type for mixed precision (torch.float16 or torch.bfloat16)

    Returns:
        Tuple of (fwd_mean, fwd_std, bwd_mean, bwd_std) in seconds
    """
    model.train()

    # Setup autocast context (or nullcontext if not using mixed precision)
    if use_mixed_precision:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    # Warmup
    for _ in range(warmup_steps):
        model.zero_grad()
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.sum()
        loss.backward()
        torch.cuda.synchronize()

    # Measure forward pass
    fwd_times = []
    for _ in range(measure_steps):
        model.zero_grad()
        start = timeit.default_timer()
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.sum()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        fwd_times.append(end - start)

    # Measure backward pass
    bwd_times = []
    for _ in range(measure_steps):
        model.zero_grad()
        # Forward pass (not timed)
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.sum()
        torch.cuda.synchronize()

        # Time only backward
        start = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        bwd_times.append(end - start)

    return (
        np.mean(fwd_times),
        np.std(fwd_times),
        np.mean(bwd_times),
        np.std(bwd_times),
    )


def benchmark_all_models(
    context_length: int = 512,
    batch_size: int = 4,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    dtype: torch.dtype = torch.bfloat16,
    output_path: Optional[str] = None,
):
    """
    Benchmark all model sizes with and without mixed precision.

    Args:
        context_length: Sequence length
        batch_size: Batch size
        warmup_steps: Number of warmup steps
        measure_steps: Number of measurement steps
        dtype: Data type for mixed precision
        output_path: Path to save results CSV (optional)
    """
    # Check GPU memory to decide which models to run
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_memory_gb:.1f} GB")
        if gpu_memory_gb < 20:
            print("WARNING: Limited GPU memory, skipping xl/2.7B models")
            model_sizes = ["small", "medium", "large"]
        else:
            model_sizes = list(MODEL_CONFIGS.keys())
    else:
        print("ERROR: CUDA not available")
        return

    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(f"\nBenchmarking all models (FP32 vs {dtype_name.upper()} mixed precision)")
    print(f"Context length: {context_length}, Batch size: {batch_size}")
    print(f"Warmup steps: {warmup_steps}, Measure steps: {measure_steps}")
    print()

    results = []

    for model_size in model_sizes:
        config = MODEL_CONFIGS[model_size]
        print("=" * 80)
        print(f"Model: {model_size}")
        print("=" * 80)

        # Create model
        try:
            model = create_model(config, context_length, device="cuda")
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

            # Generate input
            input_ids = generate_random_batch(
                batch_size, context_length, config.vocab_size, device="cuda"
            )

            # Benchmark FP32
            print(f"\nBenchmarking FP32...")
            try:
                fwd_fp32, fwd_std_fp32, bwd_fp32, bwd_std_fp32 = benchmark_with_mixed_precision(
                    model, input_ids, warmup_steps, measure_steps, use_mixed_precision=False
                )
                print(f"  Forward:  {fwd_fp32 * 1000:.2f} ± {fwd_std_fp32 * 1000:.2f} ms")
                print(f"  Backward: {bwd_fp32 * 1000:.2f} ± {bwd_std_fp32 * 1000:.2f} ms")
                print(f"  Total:    {(fwd_fp32 + bwd_fp32) * 1000:.2f} ms")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ✗ OOM with FP32")
                    fwd_fp32 = fwd_std_fp32 = bwd_fp32 = bwd_std_fp32 = float('nan')
                    torch.cuda.empty_cache()
                else:
                    raise

            # Benchmark mixed precision
            print(f"\nBenchmarking {dtype_name.upper()} mixed precision...")
            try:
                fwd_mp, fwd_std_mp, bwd_mp, bwd_std_mp = benchmark_with_mixed_precision(
                    model, input_ids, warmup_steps, measure_steps,
                    use_mixed_precision=True, dtype=dtype
                )
                print(f"  Forward:  {fwd_mp * 1000:.2f} ± {fwd_std_mp * 1000:.2f} ms")
                print(f"  Backward: {bwd_mp * 1000:.2f} ± {bwd_std_mp * 1000:.2f} ms")
                print(f"  Total:    {(fwd_mp + bwd_mp) * 1000:.2f} ms")

                # Calculate speedup
                if not np.isnan(fwd_fp32):
                    fwd_speedup = fwd_fp32 / fwd_mp
                    bwd_speedup = bwd_fp32 / bwd_mp
                    total_speedup = (fwd_fp32 + bwd_fp32) / (fwd_mp + bwd_mp)
                    print(f"\nSpeedup ({dtype_name.upper()} vs FP32):")
                    print(f"  Forward:  {fwd_speedup:.2f}x")
                    print(f"  Backward: {bwd_speedup:.2f}x")
                    print(f"  Total:    {total_speedup:.2f}x")
                else:
                    fwd_speedup = bwd_speedup = total_speedup = float('nan')

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ✗ OOM with {dtype_name.upper()}")
                    fwd_mp = fwd_std_mp = bwd_mp = bwd_std_mp = float('nan')
                    fwd_speedup = bwd_speedup = total_speedup = float('nan')
                    torch.cuda.empty_cache()
                else:
                    raise

            # Store results
            results.append({
                "model_size": model_size,
                "num_params_M": num_params / 1e6,
                "context_length": context_length,
                "batch_size": batch_size,
                # FP32
                "fp32_forward_ms": fwd_fp32 * 1000,
                "fp32_forward_std_ms": fwd_std_fp32 * 1000,
                "fp32_backward_ms": bwd_fp32 * 1000,
                "fp32_backward_std_ms": bwd_std_fp32 * 1000,
                "fp32_total_ms": (fwd_fp32 + bwd_fp32) * 1000,
                # Mixed precision
                f"{dtype_name}_forward_ms": fwd_mp * 1000,
                f"{dtype_name}_forward_std_ms": fwd_std_mp * 1000,
                f"{dtype_name}_backward_ms": bwd_mp * 1000,
                f"{dtype_name}_backward_std_ms": bwd_std_mp * 1000,
                f"{dtype_name}_total_ms": (fwd_mp + bwd_mp) * 1000,
                # Speedup
                "forward_speedup": fwd_speedup,
                "backward_speedup": bwd_speedup,
                "total_speedup": total_speedup,
            })

            # Clean up
            del model
            torch.cuda.empty_cache()
            print()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ OOM during model creation")
                torch.cuda.empty_cache()
            else:
                raise

    # Save results
    df = pd.DataFrame(results)
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        # Print as table
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print(df.to_string(index=False))
        print()

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer models with mixed precision support"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Single model size to benchmark",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark all model sizes",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length (sequence length)",
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
        "--use-mixed-precision",
        action="store_true",
        help="Use mixed precision (BF16)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type for mixed precision",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mixed_precision/mixed_precision_benchmark.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return

    # Check BF16 support
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("WARNING: BF16 not supported on this GPU, falling back to FP16")
        dtype = torch.float16

    if args.all_models:
        # Benchmark all models
        benchmark_all_models(
            context_length=args.context_length,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            dtype=dtype,
            output_path=args.output,
        )
    else:
        # Benchmark single model
        if not args.model_size:
            print("ERROR: Must specify --model-size or --all-models")
            return

        config = MODEL_CONFIGS[args.model_size]
        print(f"Benchmarking {config.name} model")
        print(f"  Mixed precision: {args.use_mixed_precision}")
        print(f"  Data type: {args.dtype if args.use_mixed_precision else 'fp32'}")
        print()

        # Create model
        model = create_model(config, args.context_length, device="cuda")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
        print()

        # Generate input
        input_ids = generate_random_batch(
            args.batch_size,
            args.context_length,
            config.vocab_size,
            device="cuda",
        )

        # Benchmark
        print("Running benchmark...")
        fwd_mean, fwd_std, bwd_mean, bwd_std = benchmark_with_mixed_precision(
            model,
            input_ids,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            use_mixed_precision=args.use_mixed_precision,
            dtype=dtype,
        )

        print()
        print("Results:")
        print(f"  Forward pass:")
        print(f"    Mean: {fwd_mean * 1000:.2f} ± {fwd_std * 1000:.2f} ms")
        print(f"  Backward pass:")
        print(f"    Mean: {bwd_mean * 1000:.2f} ± {bwd_std * 1000:.2f} ms")
        print(f"  Total:")
        print(f"    Mean: {(fwd_mean + bwd_mean) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
