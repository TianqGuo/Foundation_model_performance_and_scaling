#!/usr/bin/env python3
"""
Text Generation Script

Problem (generate): Generate text using trained Transformer models.

This script loads a trained checkpoint and generates text samples to demonstrate
the model's language generation capabilities. It allows experimenting with different
decoding parameters (temperature, top-p, top-k) to control generation quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import torch
from pathlib import Path

# Ensure package imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizer.bpe_tokenizer import Tokenizer
from cs336_basics.transformer_training.model.transformer_lm import TransformerLM
from cs336_basics.transformer_decode.decoding import generate_text


def load_model_from_checkpoint(checkpoint_path: Path, device: str | None = None) -> tuple[TransformerLM, dict]:
    """Load model and config from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config (could be in different formats depending on checkpoint)
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Try to load config from same directory
        config_path = checkpoint_path.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

    # Initialize model
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
    )

    # Load model weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Handle torch.compile() prefix (_orig_mod.)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Vocabulary size: {config['vocab_size']:,}")
    print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    return model, config


def load_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str] | None = None) -> Tokenizer:
    """Load BPE tokenizer from vocabulary and merges files."""
    print(f"Loading tokenizer...")
    print(f"  Vocab: {vocab_path}")
    print(f"  Merges: {merges_path}")

    tokenizer = Tokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=special_tokens
    )

    print(f"Tokenizer loaded successfully!")
    return tokenizer


def generate_samples(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token: str = "<|endoftext|>",
    device: str | None = None,
    seed: int | None = None,
) -> list[str]:
    """Generate text samples from multiple prompts."""
    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None

    if device is None:
        device = next(model.parameters()).device

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt!r}")
        print(f"-" * 80)

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token=eos_token,
            device=device,
            generator=generator,
        )

        print(generated)
        print(f"{'='*80}\n")

        results.append(generated)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from trained Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from best batch_8 model with default settings
  python -m cs336_basics.assignment_experiments.generate_text

  # Use custom checkpoint
  python -m cs336_basics.assignment_experiments.generate_text \\
      --checkpoint cs336_basics/basics/runs/lr_sweep/lr_3e_04/checkpoints/best_model.pt

  # Adjust temperature for more creative output
  python -m cs336_basics.assignment_experiments.generate_text --temperature 1.2

  # Use top-p (nucleus) sampling
  python -m cs336_basics.assignment_experiments.generate_text --top-p 0.9

  # Use top-k sampling
  python -m cs336_basics.assignment_experiments.generate_text --top-k 50

  # Custom prompts
  python -m cs336_basics.assignment_experiments.generate_text \\
      --prompts "Once upon a time" "In a faraway land"
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("cs336_basics/basics/runs/batch_size_sweep/batch_8/checkpoints/best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("cs336_basics/artifacts/vocabularies/tinystories_vocab.json"),
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--merges",
        type=Path,
        default=Path("cs336_basics/artifacts/vocabularies/tinystories_merges.pkl"),
        help="Path to merges file",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["Once upon a time, there"],
        help="Text prompts to generate from",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling: only sample from top k tokens",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p (nucleus) sampling: sample from tokens with cumulative prob >= p",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default="<|endoftext|>",
        help="End of sequence token",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device override (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: save generated text to file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("="*80)
    print("TEXT GENERATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperature: {args.temperature}")
    if args.top_k:
        print(f"Top-k: {args.top_k}")
    if args.top_p:
        print(f"Top-p: {args.top_p}")
    print(f"Max new tokens: {args.max_tokens}")
    print(f"Random seed: {args.seed}")
    print("="*80)

    # Load tokenizer
    tokenizer = load_tokenizer(
        vocab_path=args.vocab,
        merges_path=args.merges,
        special_tokens=[args.eos_token]
    )

    # Load model
    model, config = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Generate samples
    results = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=args.prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token=args.eos_token,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
    )

    # Save to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"# Text Generation Results\n\n")
            f.write(f"**Model:** {args.checkpoint}\n\n")
            f.write(f"**Parameters:**\n")
            f.write(f"- Temperature: {args.temperature}\n")
            if args.top_k:
                f.write(f"- Top-k: {args.top_k}\n")
            if args.top_p:
                f.write(f"- Top-p: {args.top_p}\n")
            f.write(f"- Max tokens: {args.max_tokens}\n")
            f.write(f"- Seed: {args.seed}\n\n")

            for i, (prompt, result) in enumerate(zip(args.prompts, results)):
                f.write(f"## Sample {i+1}\n\n")
                f.write(f"**Prompt:** {prompt}\n\n")
                f.write(f"**Generated:**\n\n")
                f.write(f"{result}\n\n")
                f.write("-" * 80 + "\n\n")

        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
