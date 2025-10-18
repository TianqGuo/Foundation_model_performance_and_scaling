#!/usr/bin/env python3
"""
Generate text samples from trained OpenWebText model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
from cs336_basics.transformer_training.model.transformer_lm import TransformerLM
from cs336_basics.tokenizer import BPETokenizer


def load_owt_tokenizer():
    """Load OpenWebText BPE tokenizer."""
    vocab_path = "cs336_basics/artifacts/vocabularies/owt_vocab.json"
    merges_path = "cs336_basics/artifacts/vocabularies/owt_merges.txt"

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    with open(merges_path, 'r') as f:
        merges_lines = f.read().strip().split('\n')

    merges = [tuple(line.split()) for line in merges_lines if line.strip()]

    return BPETokenizer(vocab, merges)


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Handle torch.compile prefix
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Initialize model
    model = TransformerLM(
        vocab_size=32000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def generate_samples(model, tokenizer, prompts, device='cuda'):
    """Generate text samples from prompts."""
    results = []

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)

        # Tokenize prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

        # Generate with different strategies
        strategies = [
            ("Greedy", {}),
            ("Temperature=0.8", {"temperature": 0.8}),
            ("Top-k=50", {"top_k": 50}),
            ("Top-p=0.9", {"top_p": 0.9}),
        ]

        for strategy_name, kwargs in strategies:
            generated = model.generate(
                input_ids,
                max_new_tokens=100,
                **kwargs
            )

            text = tokenizer.decode(generated[0].tolist())
            print(f"\n[{strategy_name}]")
            print(text)

            results.append({
                "prompt": prompt,
                "strategy": strategy_name,
                "generated_text": text
            })

        print("=" * 80)

    return results


def main():
    """Main generation script."""
    print("=" * 80)
    print("OPENWEBTEXT TEXT GENERATION")
    print("=" * 80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_owt_tokenizer()
    print(f"Vocabulary size: {len(tokenizer.vocab):,}")

    # Load model
    checkpoint_path = "cs336_basics/basics/runs/openwebtext/checkpoints/best_model.pt"
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device=device)
    print(f"Model loaded successfully")

    # Define prompts (similar style to OpenWebText)
    prompts = [
        "The history of artificial intelligence began",
        "In a recent scientific study, researchers found that",
        "The impact of climate change on",
        "According to experts in the field,",
        "Baseball Prospectus director of technology",
    ]

    # Generate samples
    print("\n" + "=" * 80)
    print("GENERATING TEXT SAMPLES")
    print("=" * 80)

    results = generate_samples(model, tokenizer, prompts, device=device)

    # Save results
    output_path = Path("cs336_basics/assignment_experiments/openwebtext_experiment/generation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
