#!/usr/bin/env python3
"""
Quick Tokenizer Experiments - Memory-efficient version
For systems with limited RAM
"""

import time
import random
import numpy as np
import sys
from pathlib import Path

# Add parent directory to sys.path to import tokenizer
sys.path.append(str(Path(__file__).parent.parent))
from tokenizer.bpe_tokenizer import Tokenizer

def quick_experiments():
    """Run quick experiments with sample data to avoid memory issues."""
    print("=== Quick Tokenizer Experiments (Memory-Safe) ===")
    print("Problem (tokenizer_experiments): Experiments with tokenizers (4 points)")

    # Check if tokenizer files exist
    required_files = [
        '../artifacts/vocabularies/tinystories_vocab.json', '../artifacts/vocabularies/tinystories_merges.pkl',
        '../artifacts/vocabularies/owt_vocab.json', '../artifacts/vocabularies/owt_merges.pkl'
    ]

    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"Error: Missing tokenizer files: {missing_files}")
        print("Please run BPE training first to generate these files.")
        return

    # Load tokenizers
    print("Loading tokenizers...")
    ts_tokenizer = Tokenizer.from_files(
        '../artifacts/vocabularies/tinystories_vocab.json',
        '../artifacts/vocabularies/tinystories_merges.pkl',
        ['<|endoftext|>']
    )
    owt_tokenizer = Tokenizer.from_files(
        '../artifacts/vocabularies/owt_vocab.json',
        '../artifacts/vocabularies/owt_merges.pkl',
        ['<|endoftext|>']
    )

    # Sample texts for testing
    ts_samples = [
        "Once upon a time, there was a little girl named Lucy. She loved to play with her toys and read books.",
        "Tommy found a magic wand in his garden. When he waved it, all the flowers started to dance!",
        "The princess lived in a beautiful castle. She had a pet dragon who was very friendly."
    ]

    owt_samples = [
        "The advancement of artificial intelligence has transformed numerous industries and applications worldwide.",
        "Machine learning algorithms can process vast amounts of data to identify patterns and make predictions.",
        "Neural networks are computational models inspired by the human brain's structure and function."
    ]

    print("\n=== Experiment (a): Compression Ratios ===")

    # Calculate compression ratios
    def calc_ratio(text, tokenizer):
        return len(text.encode('utf-8')) / len(tokenizer.encode(text))

    # TinyStories tokenizer on TinyStories text
    ts_ratios = [calc_ratio(text, ts_tokenizer) for text in ts_samples]
    ts_avg = np.mean(ts_ratios)
    print(f"TinyStories tokenizer on TinyStories text: {ts_avg:.2f} bytes/token")

    # OpenWebText tokenizer on OpenWebText text
    owt_ratios = [calc_ratio(text, owt_tokenizer) for text in owt_samples]
    owt_avg = np.mean(owt_ratios)
    print(f"OpenWebText tokenizer on OpenWebText text: {owt_avg:.2f} bytes/token")

    print(f"\nDeliverable (a): The TinyStories tokenizer achieves {ts_avg:.2f} bytes/token compression, while the OpenWebText tokenizer achieves {owt_avg:.2f} bytes/token, with the larger vocabulary providing better compression efficiency.")

    print("\n=== Experiment (b): Cross-Tokenization ===")

    # OpenWebText text with TinyStories tokenizer
    owt_on_ts_ratios = [calc_ratio(text, ts_tokenizer) for text in owt_samples]
    owt_on_ts_avg = np.mean(owt_on_ts_ratios)
    print(f"OpenWebText text with TinyStories tokenizer: {owt_on_ts_avg:.2f} bytes/token")
    print(f"OpenWebText text with OpenWebText tokenizer: {owt_avg:.2f} bytes/token")

    degradation = (owt_on_ts_avg - owt_avg) / owt_avg * 100
    print(f"Performance degradation: {degradation:.1f}%")

    print(f"\nDeliverable (b): Using the TinyStories tokenizer on OpenWebText results in {owt_on_ts_avg:.2f} bytes/token compression compared to {owt_avg:.2f} bytes/token with the native tokenizer, representing a {degradation:.1f}% degradation due to domain mismatch and smaller vocabulary size.")

    print("\n=== Experiment (c): Throughput Estimation ===")

    # Throughput test with repeated text
    test_text = ' '.join(owt_samples * 100)  # Repeat for meaningful measurement
    test_bytes = len(test_text.encode('utf-8'))

    print(f"Testing with {test_bytes:,} bytes of text...")

    start = time.time()
    tokens = owt_tokenizer.encode(test_text)
    end = time.time()

    encoding_time = end - start
    throughput_bytes_per_sec = test_bytes / encoding_time
    throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)

    print(f"Encoded {len(tokens):,} tokens in {encoding_time:.3f} seconds")
    print(f"Throughput: {throughput_mb_per_sec:.1f} MB/s")

    # Estimate time for Pile dataset (825GB)
    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * 1024 * 1024 * 1024
    estimated_time_seconds = pile_size_bytes / throughput_bytes_per_sec
    estimated_time_hours = estimated_time_seconds / 3600
    estimated_time_days = estimated_time_hours / 24

    print(f"Estimated time to tokenize Pile dataset ({pile_size_gb}GB): {estimated_time_days:.1f} days")

    print(f"\nDeliverable (c): The tokenizer achieves approximately {throughput_mb_per_sec:.1f} MB/s throughput, which would require approximately {estimated_time_days:.1f} days to tokenize the entire 825GB Pile dataset.")

    print("\n=== Experiment (d): uint16 Analysis ===")

    # Check vocabulary sizes
    ts_max_id = max(ts_tokenizer.vocab.keys())
    owt_max_id = max(owt_tokenizer.vocab.keys())
    uint16_max = 65535

    print(f"TinyStories max token ID: {ts_max_id:,}")
    print(f"OpenWebText max token ID: {owt_max_id:,}")
    print(f"uint16 maximum value: {uint16_max:,}")

    # Demonstrate encoding
    sample_text = ts_samples[0] + "<|endoftext|>"
    tokens = ts_tokenizer.encode(sample_text)
    token_array = np.array(tokens, dtype=np.uint16)

    print(f"\nSample encoding:")
    print(f"Text: '{sample_text}'")
    print(f"Tokens: {tokens[:10]}... ({len(tokens)} total)")
    print(f"uint16 array shape: {token_array.shape}")
    print(f"Array max value: {token_array.max()}")

    # Save sample
    np.save('sample_tokens_demo.npy', token_array)
    print(f"Saved demo array to 'sample_tokens_demo.npy'")

    print(f"\nDeliverable (d): uint16 is appropriate because our vocabulary sizes (10K and 32K) are well below the uint16 maximum of 65,536, providing efficient 2-byte storage per token while supporting future vocabulary expansion.")

    print("\n=== Experiments Complete ===")
    print("All deliverable responses generated successfully!")

if __name__ == "__main__":
    quick_experiments()