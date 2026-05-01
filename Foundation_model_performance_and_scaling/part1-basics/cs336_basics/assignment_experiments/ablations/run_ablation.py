#!/usr/bin/env python3
"""
Ablation Experiment Runner

This script runs ablation experiments to understand the impact of various
Transformer architectural components:

1. layer_norm_ablation: Remove RMSNorm
2. pre_norm_ablation: Post-norm vs. Pre-norm
3. no_pos_emb: No position embeddings (NoPE)
4. swiglu_ablation: SwiGLU vs. SiLU FFN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure package imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.basics.run_experiment import run_experiment


ABLATION_CONFIGS = {
    "baseline": {
        "description": "Baseline model with all standard components",
        "use_layer_norm": True,
        "post_norm": False,
        "use_swiglu": True,
        "use_rope": True,
        "d_ff_multiplier": 8/3,  # SwiGLU uses ~(8/3) * d_model
    },
    "no_layer_norm": {
        "description": "Ablation: Remove all RMSNorm layers",
        "use_layer_norm": False,
        "post_norm": False,
        "use_swiglu": True,
        "use_rope": True,
        "d_ff_multiplier": 8/3,
    },
    "post_norm": {
        "description": "Ablation: Post-norm instead of pre-norm",
        "use_layer_norm": True,
        "post_norm": True,
        "use_swiglu": True,
        "use_rope": True,
        "d_ff_multiplier": 8/3,
    },
    "no_position_emb": {
        "description": "Ablation: No position embeddings (NoPE)",
        "use_layer_norm": True,
        "post_norm": False,
        "use_swiglu": True,
        "use_rope": False,
        "d_ff_multiplier": 8/3,
    },
    "silu_ffn": {
        "description": "Ablation: SiLU FFN instead of SwiGLU",
        "use_layer_norm": True,
        "post_norm": False,
        "use_swiglu": False,
        "use_rope": True,
        "d_ff_multiplier": 4,  # SiLU uses 4 * d_model to match parameters
    },
}


def calculate_d_ff(d_model: int, multiplier: float) -> int:
    """Calculate d_ff ensuring it's a multiple of 64 for GPU efficiency."""
    d_ff = int(multiplier * d_model)
    # Round to nearest multiple of 64
    d_ff = ((d_ff + 31) // 64) * 64
    return d_ff


def make_ablation_config(
    *,
    ablation_type: str,
    learning_rate: float,
    total_tokens: int,
    batch_size: int,
    context_length: int,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    device: str | None = None,
) -> dict:
    """Build configuration for ablation experiment."""
    if ablation_type not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    ablation_spec = ABLATION_CONFIGS[ablation_type]

    # Calculate training steps
    denominator = batch_size * context_length
    max_iters = max(1, total_tokens // denominator)
    processed_tokens = max_iters * denominator

    # Calculate d_ff based on ablation type
    d_ff = calculate_d_ff(d_model, ablation_spec["d_ff_multiplier"])

    warmup = max(100, max_iters // 100)  # 1% warmup

    config = {
        "experiment_name": ablation_type,
        "description": ablation_spec["description"],
        "dataset": "TinyStories",
        "vocab_size": 10_000,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
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
        "processed_tokens": processed_tokens,
        # Ablation-specific flags
        "use_layer_norm": ablation_spec["use_layer_norm"],
        "post_norm": ablation_spec["post_norm"],
        "use_swiglu": ablation_spec["use_swiglu"],
        "use_rope": ablation_spec["use_rope"],
        "use_ablation_model": True,  # Flag to use FlexibleTransformerLM
    }

    return config


def run_ablation_experiment(
    *,
    ablation_type: str,
    learning_rate: float,
    total_tokens: int,
    batch_size: int,
    context_length: int,
    output_dir: Path,
    device: str | None = None,
) -> None:
    """Run a single ablation experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"ABLATION EXPERIMENT: {ablation_type}")
    print("=" * 80)
    print(f"Description: {ABLATION_CONFIGS[ablation_type]['description']}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Target tokens: {total_tokens:,}")
    if device:
        print(f"Device override: {device}")
    print(f"Output directory: {output_dir}\n")

    config = make_ablation_config(
        ablation_type=ablation_type,
        learning_rate=learning_rate,
        total_tokens=total_tokens,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    # Run experiment
    try:
        run_experiment(
            experiment_name=ablation_type,
            config_dict=config,
            output_dir=output_dir,
        )
        print(f"\n✓ Experiment '{ablation_type}' completed successfully!")
    except Exception as e:
        print(f"\n✗ Experiment '{ablation_type}' failed:")
        print(f"   {type(e).__name__}: {e}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for Transformer components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available ablations:
  baseline         - Standard pre-norm Transformer with SwiGLU and RoPE
  no_layer_norm    - Remove all RMSNorm layers
  post_norm        - Use post-norm instead of pre-norm
  no_position_emb  - Remove RoPE position embeddings (NoPE)
  silu_ffn         - Use SiLU FFN instead of SwiGLU

Examples:
  # Run baseline experiment
  uv run python -m cs336_basics.assignment_experiments.ablations.run_ablation \\
      --ablation baseline

  # Run no layer norm ablation with lower learning rate
  uv run python -m cs336_basics.assignment_experiments.ablations.run_ablation \\
      --ablation no_layer_norm --learning-rate 1e-4

  # Run all ablations
  for abl in baseline no_layer_norm post_norm no_position_emb silu_ffn; do
      uv run python -m cs336_basics.assignment_experiments.ablations.run_ablation \\
          --ablation $abl
  done
        """,
    )

    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Type of ablation to run",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4, use lower for no_layer_norm)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32 for fast iteration)",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=327_680_000,
        help="Total tokens to process (default: 327.68M)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory (default: runs/ablations/{ablation_type})",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device override (default: auto-detect)",
    )
    parser.add_argument(
        "--low-resource",
        action="store_true",
        help="Use reduced token budget (40M tokens)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set total tokens based on resource mode
    if args.low_resource:
        total_tokens = 40_000_000
        print("Low-resource mode: training with 40M tokens")
    else:
        total_tokens = args.total_tokens

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(f"cs336_basics/basics/runs/ablations/{args.ablation}")

    # Run experiment
    run_ablation_experiment(
        ablation_type=args.ablation,
        learning_rate=args.learning_rate,
        total_tokens=total_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
