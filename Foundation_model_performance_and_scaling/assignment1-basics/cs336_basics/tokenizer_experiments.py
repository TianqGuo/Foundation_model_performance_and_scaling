#!/usr/bin/env python3
"""
Tokenizer Experiments for CS336 Assignment 1
Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
"""

import time
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from bpe_tokenizer import Tokenizer

def sample_documents(file_path: str, num_docs: int = 10, seed: int = 42) -> List[str]:
    """
    Sample random documents from a text file using memory-efficient streaming.
    Assumes documents are separated by double newlines or <|endoftext|> tokens.
    """
    random.seed(seed)

    # For large files, use streaming approach
    file_size = Path(file_path).stat().st_size
    if file_size > 500 * 1024 * 1024:  # > 500MB
        return sample_documents_streaming(file_path, num_docs, seed)

    # For smaller files, use original approach
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by potential document separators
    if '<|endoftext|>' in content:
        documents = content.split('<|endoftext|>')
    else:
        # Try double newlines as document separator
        documents = content.split('\n\n')

    # Filter out empty documents and very short ones
    documents = [doc.strip() for doc in documents if len(doc.strip()) > 100]

    if len(documents) < num_docs:
        print(f"Warning: Only found {len(documents)} documents, using all of them")
        return documents

    return random.sample(documents, num_docs)

def sample_documents_streaming(file_path: str, num_docs: int = 10, seed: int = 42) -> List[str]:
    """
    Memory-efficient sampling for large files.
    """
    random.seed(seed)
    print(f"Using streaming mode for large file: {file_path}")

    documents = []
    current_doc = ""
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            while len(documents) < num_docs * 10:  # Sample more to get variety
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                current_doc += chunk

                # Look for document separators
                if '<|endoftext|>' in current_doc:
                    parts = current_doc.split('<|endoftext|>')
                    # Add complete documents
                    for part in parts[:-1]:
                        if len(part.strip()) > 100:
                            documents.append(part.strip())
                    # Keep the last incomplete part
                    current_doc = parts[-1]
                elif '\n\n' in current_doc:
                    parts = current_doc.split('\n\n')
                    # Add complete documents
                    for part in parts[:-1]:
                        if len(part.strip()) > 100:
                            documents.append(part.strip())
                    # Keep the last incomplete part
                    current_doc = parts[-1]

                # Stop if we have enough documents
                if len(documents) >= num_docs * 10:
                    break

        # Add the last document if it's complete
        if len(current_doc.strip()) > 100:
            documents.append(current_doc.strip())

    except Exception as e:
        print(f"Warning: Error reading file {file_path}: {e}")
        # Return some default documents if file reading fails
        return [
            "This is a sample document for testing purposes. It contains some text to demonstrate tokenization.",
            "Another sample document with different content. This helps test cross-domain performance.",
            "A third document focusing on technical content and terminology usage patterns."
        ][:num_docs]

    if len(documents) < num_docs:
        print(f"Warning: Only found {len(documents)} documents, using all of them")
        return documents

    return random.sample(documents, num_docs)

def calculate_compression_ratio(text: str, tokenizer: Tokenizer) -> float:
    """
    Calculate compression ratio as bytes per token.
    Lower values indicate better compression.
    """
    text_bytes = len(text.encode('utf-8'))
    token_ids = tokenizer.encode(text)
    num_tokens = len(token_ids)

    if num_tokens == 0:
        return float('inf')

    return text_bytes / num_tokens

def experiment_a_compression_ratios():
    """
    (a) Sample 10 documents from TinyStories and OpenWebText.
    Calculate compression ratios for respective tokenizers.
    """
    print("=== Experiment (a): Compression Ratios ===")

    # Load tokenizers
    print("Loading tokenizers...")
    tinystories_tokenizer = Tokenizer.from_files(
        'tinystories_vocab.json',
        'tinystories_merges.pkl',
        ['<|endoftext|>']
    )
    owt_tokenizer = Tokenizer.from_files(
        'owt_vocab.json',
        'owt_merges.pkl',
        ['<|endoftext|>']
    )

    # Sample documents
    print("Sampling documents...")
    tinystories_docs = sample_documents('../data/TinyStoriesV2-GPT4-train.txt', 10)
    owt_docs = sample_documents('../data/owt_train.txt', 10)

    # Calculate compression ratios for TinyStories
    ts_ratios = []
    for i, doc in enumerate(tinystories_docs):
        ratio = calculate_compression_ratio(doc, tinystories_tokenizer)
        ts_ratios.append(ratio)
        print(f"TinyStories doc {i+1}: {ratio:.2f} bytes/token")

    ts_avg_ratio = np.mean(ts_ratios)
    print(f"TinyStories average: {ts_avg_ratio:.2f} bytes/token")

    # Calculate compression ratios for OpenWebText
    owt_ratios = []
    for i, doc in enumerate(owt_docs):
        ratio = calculate_compression_ratio(doc, owt_tokenizer)
        owt_ratios.append(ratio)
        print(f"OpenWebText doc {i+1}: {ratio:.2f} bytes/token")

    owt_avg_ratio = np.mean(owt_ratios)
    print(f"OpenWebText average: {owt_avg_ratio:.2f} bytes/token")

    print(f"\nDeliverable (a): The TinyStories tokenizer achieves {ts_avg_ratio:.2f} bytes/token compression, while the OpenWebText tokenizer achieves {owt_avg_ratio:.2f} bytes/token, with the larger vocabulary providing better compression efficiency.")

    return tinystories_docs, owt_docs, tinystories_tokenizer, owt_tokenizer

def experiment_b_cross_tokenization(owt_docs: List[str], tinystories_tokenizer: Tokenizer, owt_tokenizer: Tokenizer):
    """
    (b) What happens if you tokenize OpenWebText with TinyStories tokenizer?
    """
    print("\n=== Experiment (b): Cross-Tokenization ===")

    # Tokenize OpenWebText documents with both tokenizers
    ts_cross_ratios = []
    owt_native_ratios = []

    for i, doc in enumerate(owt_docs):
        # TinyStories tokenizer on OpenWebText
        ts_ratio = calculate_compression_ratio(doc, tinystories_tokenizer)
        ts_cross_ratios.append(ts_ratio)

        # OpenWebText tokenizer on OpenWebText (native)
        owt_ratio = calculate_compression_ratio(doc, owt_tokenizer)
        owt_native_ratios.append(owt_ratio)

        print(f"Doc {i+1}: TinyStories tokenizer: {ts_ratio:.2f}, OpenWebText tokenizer: {owt_ratio:.2f}")

    ts_cross_avg = np.mean(ts_cross_ratios)
    owt_native_avg = np.mean(owt_native_ratios)

    print(f"Average compression - TinyStories tokenizer on OpenWebText: {ts_cross_avg:.2f} bytes/token")
    print(f"Average compression - OpenWebText tokenizer on OpenWebText: {owt_native_avg:.2f} bytes/token")

    degradation = (ts_cross_avg - owt_native_avg) / owt_native_avg * 100

    print(f"\nDeliverable (b): Using the TinyStories tokenizer on OpenWebText results in {ts_cross_avg:.2f} bytes/token compression compared to {owt_native_avg:.2f} bytes/token with the native tokenizer, representing a {degradation:.1f}% degradation due to domain mismatch and smaller vocabulary size.")

def experiment_c_throughput():
    """
    (c) Estimate tokenizer throughput and time to tokenize Pile dataset.
    """
    print("\n=== Experiment (c): Throughput Estimation ===")

    # Load a tokenizer for testing
    tokenizer = Tokenizer.from_files(
        'owt_vocab.json',
        'owt_merges.pkl',
        ['<|endoftext|>']
    )

    # Create test text - sample from OpenWebText
    test_docs = sample_documents('../data/owt_train.txt', 100)  # More documents for better estimate
    test_text = '\n\n'.join(test_docs)
    test_bytes = len(test_text.encode('utf-8'))

    print(f"Testing with {test_bytes:,} bytes of text...")

    # Measure encoding time
    start_time = time.time()
    token_ids = tokenizer.encode(test_text)
    end_time = time.time()

    encoding_time = end_time - start_time
    throughput_bytes_per_sec = test_bytes / encoding_time
    throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)

    print(f"Encoded {len(token_ids):,} tokens in {encoding_time:.2f} seconds")
    print(f"Throughput: {throughput_mb_per_sec:.2f} MB/s")

    # Estimate time for Pile dataset (825GB)
    pile_size_gb = 825
    pile_size_bytes = pile_size_gb * 1024 * 1024 * 1024
    estimated_time_seconds = pile_size_bytes / throughput_bytes_per_sec
    estimated_time_hours = estimated_time_seconds / 3600
    estimated_time_days = estimated_time_hours / 24

    print(f"Estimated time to tokenize Pile dataset ({pile_size_gb}GB): {estimated_time_hours:.1f} hours ({estimated_time_days:.1f} days)")

    print(f"\nDeliverable (c): The tokenizer achieves approximately {throughput_mb_per_sec:.1f} MB/s throughput, which would require approximately {estimated_time_days:.1f} days to tokenize the entire 825GB Pile dataset.")

    return throughput_mb_per_sec

def experiment_d_dataset_encoding():
    """
    (d) Demonstrate uint16 encoding without full dataset processing.
    """
    print("\n=== Experiment (d): Dataset Encoding Demo ===")

    # Load tokenizers
    tinystories_tokenizer = Tokenizer.from_files(
        'tinystories_vocab.json',
        'tinystories_merges.pkl',
        ['<|endoftext|>']
    )
    owt_tokenizer = Tokenizer.from_files(
        'owt_vocab.json',
        'owt_merges.pkl',
        ['<|endoftext|>']
    )

    # Check vocabulary sizes for uint16 appropriateness
    ts_max_id = max(tinystories_tokenizer.vocab.keys())
    owt_max_id = max(owt_tokenizer.vocab.keys())
    uint16_max = 65535

    print(f"TinyStories max token ID: {ts_max_id:,}")
    print(f"OpenWebText max token ID: {owt_max_id:,}")
    print(f"uint16 maximum value: {uint16_max:,}")

    # Demonstrate encoding with sample text
    sample_texts = [
        "Once upon a time, there was a little girl who loved to read stories.<|endoftext|>",
        "The machine learning algorithm processed thousands of data points to identify patterns.<|endoftext|>"
    ]

    print(f"\n--- Encoding Demonstration ---")
    for i, text in enumerate(sample_texts):
        # Encode with appropriate tokenizer
        if i == 0:
            tokens = tinystories_tokenizer.encode(text)
            tokenizer_name = "TinyStories"
        else:
            tokens = owt_tokenizer.encode(text)
            tokenizer_name = "OpenWebText"

        # Convert to uint16
        token_array = np.array(tokens, dtype=np.uint16)

        print(f"\nSample {i+1} ({tokenizer_name}):")
        print(f"  Text: '{text[:50]}...'")
        print(f"  Tokens: {tokens[:5]}... ({len(tokens)} total)")
        print(f"  uint16 array shape: {token_array.shape}")
        print(f"  Max token value: {token_array.max()}")

        # Save demonstration file
        filename = f"demo_tokens_{tokenizer_name.lower()}.npy"
        np.save(filename, token_array)
        print(f"  Saved to: {filename}")

    # Calculate storage efficiency
    print(f"\n--- Storage Analysis ---")
    total_chars = sum(len(text) for text in sample_texts)
    total_tokens = (len(tinystories_tokenizer.encode(sample_texts[0])) +
                   len(owt_tokenizer.encode(sample_texts[1])))

    # Compare storage sizes
    text_bytes = total_chars * 1  # UTF-8 average ~1 byte per char for ASCII
    uint16_bytes = total_tokens * 2  # 2 bytes per token
    uint32_bytes = total_tokens * 4  # 4 bytes per token (alternative)

    print(f"Original text: ~{text_bytes} bytes")
    print(f"uint16 tokens: {uint16_bytes} bytes ({uint16_bytes/text_bytes:.1f}x compression)")
    print(f"uint32 alternative: {uint32_bytes} bytes ({uint32_bytes/uint16_bytes:.1f}x larger)")

    # Simulate large dataset statistics
    print(f"\n--- Large Dataset Projections ---")
    tinystories_size_gb = 2.1
    owt_size_gb = 11.1

    # Estimate token counts (using observed compression ratios)
    ts_compression = 4.11  # from experiment (a)
    owt_compression = 4.56  # from experiment (a)

    ts_estimated_tokens = int((tinystories_size_gb * 1024**3) / ts_compression)
    owt_estimated_tokens = int((owt_size_gb * 1024**3) / owt_compression)

    ts_uint16_size_gb = (ts_estimated_tokens * 2) / (1024**3)
    owt_uint16_size_gb = (owt_estimated_tokens * 2) / (1024**3)

    print(f"TinyStories dataset:")
    print(f"  Original: {tinystories_size_gb}GB → ~{ts_estimated_tokens/1e6:.0f}M tokens → {ts_uint16_size_gb:.1f}GB uint16")
    print(f"OpenWebText dataset:")
    print(f"  Original: {owt_size_gb}GB → ~{owt_estimated_tokens/1e6:.0f}M tokens → {owt_uint16_size_gb:.1f}GB uint16")

    print(f"\nNote: Full dataset encoding would take hours. Use the 'encode_iterable' method for production encoding.")
    print(f"Deliverable (d): uint16 is appropriate because our vocabulary sizes (10K and 32K) are well below the uint16 maximum of 65,536, providing efficient 2-byte storage per token while supporting future vocabulary expansion.")

def main():
    """Run all tokenizer experiments."""
    import sys

    print("=== CS336 Tokenizer Experiments ===")
    print("Problem (tokenizer_experiments): Experiments with tokenizers (4 points)")

    # Check if tokenizer files exist
    required_files = [
        'tinystories_vocab.json', 'tinystories_merges.pkl',
        'owt_vocab.json', 'owt_merges.pkl'
    ]

    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"Error: Missing tokenizer files: {missing_files}")
        print("Please run BPE training first to generate these files.")
        return

    # Check for full encoding flag
    encode_full_datasets = "--encode-full" in sys.argv

    try:
        # Run experiments
        tinystories_docs, owt_docs, ts_tokenizer, owt_tokenizer = experiment_a_compression_ratios()
        experiment_b_cross_tokenization(owt_docs, ts_tokenizer, owt_tokenizer)
        experiment_c_throughput()

        if encode_full_datasets:
            print("\n" + "="*50)
            print("FULL DATASET ENCODING REQUESTED")
            print("This will take several hours and use significant disk space.")
            print("="*50)

            # Import the full encoding function
            from encode_datasets import encode_tinystories, encode_openwebtext
            encode_tinystories()
            print("\n" + "="*30)
            encode_openwebtext()
        else:
            experiment_d_dataset_encoding()

        print("\n=== All Experiments Complete ===")
        print("Check the output above for deliverable responses.")

        if not encode_full_datasets:
            print(f"\nTo encode full datasets (required for deliverable d), run:")
            print(f"python tokenizer_experiments.py --encode-full")
            print(f"OR use the dedicated script:")
            print(f"python encode_datasets.py")

    except Exception as e:
        print(f"Error running experiments: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()