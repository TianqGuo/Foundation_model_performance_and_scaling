#!/usr/bin/env python3
"""
Batch Size Sweep Experiment

Problem (batch_size_experiment): Investigate how batch size impacts
training efficiency and validation quality for the TinyStories transformer.

This script mirrors the learning-rate sweep infrastructure but varies the
`batch_size` hyperparameter while keeping the total number of tokens processed
approximately constant. Use it to gather the learning curves and discussion
points required by the assignment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


# Ensure package imports work when executed as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.basics.run_experiment import run_experiment


DEFAULT_BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]
DEFAULT_TOTAL_TOKENS = 327_680_000
LOW_RESOURCE_TOTAL_TOKENS = 40_000_000


def calculate_training_steps(
    total_tokens: int,
    batch_size: int,
    context_length: int,
) -> tuple[int, int]:
    """Return (steps, processed_tokens) for the desired training budget."""
    denominator = batch_size * context_length
    if denominator <= 0:
        raise ValueError("batch_size and context_length must be positive")

    steps = max(1, total_tokens // denominator)
    actual_tokens = steps * denominator

    if steps == 0:
        raise ValueError(
            "Total tokens too small for the requested batch size/context length"
        )

    return steps, actual_tokens


def make_config(
    *,
    batch_size: int,
    learning_rate: float,
    total_tokens: int,
    context_length: int,
    description: str,
    device: str | None,
    quick_test: bool,
) -> dict:
    """Build the experiment configuration dictionary."""

    max_iters, processed_tokens = calculate_training_steps(
        total_tokens=total_tokens,
        batch_size=batch_size,
        context_length=context_length,
    )

    if quick_test:
        # Roughly 100 iterations to smoke-test plumbing.
        max_iters = min(max_iters, 100)
        processed_tokens = batch_size * max_iters * context_length

    warmup = max(100, max_iters // 100)  # 1% warmup by default

    config = {
        "description": description,
        "dataset": "TinyStories",
        "vocab_size": 10_000,
        "context_length": context_length,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10_000.0,
        "batch_size": batch_size,
        "max_iterations": max_iters,
        "learning_rate": learning_rate,
        "min_learning_rate": learning_rate * 0.1,
        "warmup_iters": warmup,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "train_data_path": "cs336_basics/artifacts/datasets/tinystories_train_tokens.npy",
        "val_data_path": "cs336_basics/artifacts/datasets/tinystories_tokens.npy",
        "device": device,
        "dtype": "float32",
        "seed": 42,
        "log_interval": 50,
        "eval_interval": max(200, warmup),
        "checkpoint_interval": max(1000, max_iters // 10),
        # Metadata for downstream analysis
        "processed_tokens": processed_tokens,
    }

    return config


def run_batch_size_sweep(
    *,
    batch_sizes: Iterable[int],
    learning_rate: float,
    total_tokens: int,
    context_length: int,
    output_dir: Path,
    device: str | None,
    quick_test: bool,
) -> None:
    """Run the batch size experiments sequentially and record outcomes."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BATCH SIZE SWEEP EXPERIMENT")
    print("=" * 80)
    print(f"Batch sizes: {list(batch_sizes)}")
    print(f"Target tokens: {total_tokens:,}")
    print(f"Context length: {context_length}")
    print(f"Learning rate: {learning_rate}")
    if device:
        print(f"Device override: {device}")
    if quick_test:
        print("Mode: quick smoke-test (<=100 iterations per trial)")
    print(f"Output directory: {output_dir}\n")

    summary: list[dict] = []

    for batch in batch_sizes:
        exp_name = f"batch_{batch}"
        exp_dir = output_dir / exp_name
        description = (
            f"Batch size sweep — batch_size={batch}, lr={learning_rate}, "
            f"tokens≈{total_tokens:,}"
        )

        print("\n" + "-" * 80)
        print(f"Starting experiment: {exp_name}")
        print("-" * 80)

        config = make_config(
            batch_size=batch,
            learning_rate=learning_rate,
            total_tokens=total_tokens,
            context_length=context_length,
            description=description,
            device=device,
            quick_test=quick_test,
        )

        exp_dir.mkdir(parents=True, exist_ok=True)
        template_path = exp_dir / "config_template.json"
        with open(template_path, "w") as f:
            json.dump(config, f, indent=2)

        try:
            run_experiment(
                experiment_name=exp_name,
                config_dict=config,
                output_dir=exp_dir,
            )

            result = {
                "batch_size": batch,
                "status": "completed",
                "learning_rate": learning_rate,
                "max_iterations": config["max_iterations"],
                "processed_tokens": config["processed_tokens"],
            }

        except RuntimeError as err:
            error_msg = str(err)
            if "out of memory" in error_msg.lower():
                print("⚠️  CUDA OOM encountered; consider reducing batch size or using gradient accumulation.")

            result = {
                "batch_size": batch,
                "status": "failed",
                "error": error_msg,
            }

        except Exception as err:  # noqa: BLE001 - want to capture any failure
            result = {
                "batch_size": batch,
                "status": "failed",
                "error": str(err),
            }

        summary.append(result)

    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("BATCH SIZE SWEEP COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_path}")
    print("Outcomes:")
    for item in summary:
        status = "✓" if item["status"] == "completed" else "✗"
        print(f"  {status} batch={item['batch_size']}: {item['status']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep over batch sizes for the TinyStories transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default GPU sweep
  uv run python -m cs336_basics.assignment_experiments.batch_size_experiment.batch_size_sweep

  # CPU/Apple Silicon budget run
  uv run python -m cs336_basics.assignment_experiments.batch_size_experiment.batch_size_sweep --low-resource

  # Custom configuration
  uv run python -m cs336_basics.assignment_experiments.batch_size_experiment.batch_size_sweep \
      --batch-sizes 16 64 256 --learning-rate 2e-4 --context-length 512
        """,
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes to evaluate (default: [1, 8, 32, 64, 128, 256, 512])",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Base learning rate to reuse across runs",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        help="Override the target total tokens (default: 327,680,000 or 40,000,000 in low-resource mode)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Sequence length for training batches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cs336_basics/basics/runs/batch_size_sweep"),
        help="Directory to store experiment artifacts",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device override (default: auto-detect)",
    )
    parser.add_argument(
        "--low-resource",
        action="store_true",
        help="Use reduced token budget for CPU/MPS experiments",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Limit each run to <=100 iterations for debugging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_sizes:
        batch_sizes = [b for b in args.batch_sizes if b > 0]
    else:
        batch_sizes = DEFAULT_BATCH_SIZES

    if not batch_sizes:
        raise ValueError("At least one positive batch size is required")

    if args.low_resource:
        total_tokens = args.total_tokens or LOW_RESOURCE_TOTAL_TOKENS
    else:
        total_tokens = args.total_tokens or DEFAULT_TOTAL_TOKENS

    run_batch_size_sweep(
        batch_sizes=batch_sizes,
        learning_rate=args.learning_rate,
        total_tokens=total_tokens,
        context_length=args.context_length,
        output_dir=args.output_dir,
        device=args.device,
        quick_test=args.quick_test,
    )


if __name__ == "__main__":
    main()
