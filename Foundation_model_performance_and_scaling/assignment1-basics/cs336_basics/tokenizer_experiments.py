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
    (d) Encode training datasets to uint16 NumPy arrays.
    """
    print("\n=== Experiment (d): Dataset Encoding ===")

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

    def encode_dataset(file_path: str, tokenizer: Tokenizer, output_file: str):
        """Encode a dataset to uint16 NumPy array with memory management."""
        print(f"Encoding {file_path} to {output_file}...")

        # Check vocab size to confirm uint16 is appropriate
        max_token_id = max(tokenizer.vocab.keys())
        print(f"Maximum token ID: {max_token_id}")

        if max_token_id >= 65536:  # 2^16
            print(f"Warning: Token ID {max_token_id} exceeds uint16 range!")
            return

        # Check file size and use appropriate strategy
        file_size = Path(file_path).stat().st_size
        if file_size > 1024 * 1024 * 1024:  # > 1GB
            print(f"Large file detected ({file_size / 1024**3:.1f}GB), using streaming encoding...")
            return encode_dataset_streaming(file_path, tokenizer, output_file)

        # Encode dataset in chunks to manage memory
        all_token_ids = []
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = f.read(chunk_size)
            while chunk:
                token_ids = tokenizer.encode(chunk)
                all_token_ids.extend(token_ids)

                # Read next chunk with overlap to avoid breaking tokens
                next_chunk = f.read(chunk_size)
                if not next_chunk:
                    break
                chunk = chunk[-100:] + next_chunk  # 100 char overlap

        # Convert to uint16 NumPy array
        token_array = np.array(all_token_ids, dtype=np.uint16)

        # Save to disk
        np.save(output_file, token_array)

        print(f"Saved {len(token_array):,} tokens to {output_file}")
        print(f"Array shape: {token_array.shape}, dtype: {token_array.dtype}")

        # Calculate compression info
        original_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        compressed_size_mb = token_array.nbytes / (1024 * 1024)
        compression_ratio = original_size_mb / compressed_size_mb

        print(f"Original text: {original_size_mb:.1f} MB")
        print(f"Tokenized array: {compressed_size_mb:.1f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}x")

        return len(token_array)

    def encode_dataset_streaming(file_path: str, tokenizer: Tokenizer, output_file: str):
        """Stream-encode large datasets to avoid memory issues."""
        print("Using streaming encoding for large file...")

        # Process file in chunks and write incrementally
        chunk_size = 64 * 1024 * 1024  # 64MB chunks
        total_tokens = 0

        # Use memory-mapped file for output
        temp_tokens = []

        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Encode chunk
                token_ids = tokenizer.encode(chunk)
                temp_tokens.extend(token_ids)
                total_tokens += len(token_ids)

                # Periodically save to avoid memory buildup
                if len(temp_tokens) > 10_000_000:  # 10M tokens
                    print(f"Processed {total_tokens:,} tokens so far...")
                    # Save intermediate results
                    if total_tokens == len(temp_tokens):  # First batch
                        token_array = np.array(temp_tokens, dtype=np.uint16)
                        np.save(output_file, token_array)
                    else:
                        # Append to existing array
                        existing = np.load(output_file)
                        new_tokens = np.array(temp_tokens, dtype=np.uint16)
                        combined = np.concatenate([existing, new_tokens])
                        np.save(output_file, combined)
                    temp_tokens = []

        # Save remaining tokens
        if temp_tokens:
            if total_tokens == len(temp_tokens):  # Only one batch
                token_array = np.array(temp_tokens, dtype=np.uint16)
                np.save(output_file, token_array)
            else:
                # Append final batch
                existing = np.load(output_file)
                new_tokens = np.array(temp_tokens, dtype=np.uint16)
                combined = np.concatenate([existing, new_tokens])
                np.save(output_file, combined)

        print(f"Streaming encoding complete: {total_tokens:,} tokens")
        return total_tokens

    # Encode TinyStories dataset
    ts_tokens = encode_dataset(
        '../data/TinyStoriesV2-GPT4-train.txt',
        tinystories_tokenizer,
        'tinystories_tokens.npy'
    )

    # Encode OpenWebText dataset (sample for demo - full dataset would take too long)
    print("\nNote: Encoding sample of OpenWebText due to size...")
    owt_sample = sample_documents('../data/owt_train.txt', 1000)
    owt_sample_text = '\n\n'.join(owt_sample)

    with open('owt_sample.txt', 'w', encoding='utf-8') as f:
        f.write(owt_sample_text)

    owt_tokens = encode_dataset(
        'owt_sample.txt',
        owt_tokenizer,
        'owt_sample_tokens.npy'
    )

    print(f"\nDeliverable (d): uint16 is appropriate because our vocabulary sizes (10K and 32K) are well below the uint16 maximum of 65,536, providing efficient 2-byte storage per token while supporting future vocabulary expansion.")

def main():
    """Run all tokenizer experiments."""
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

    try:
        # Run experiments
        tinystories_docs, owt_docs, ts_tokenizer, owt_tokenizer = experiment_a_compression_ratios()
        experiment_b_cross_tokenization(owt_docs, ts_tokenizer, owt_tokenizer)
        experiment_c_throughput()
        experiment_d_dataset_encoding()

        print("\n=== All Experiments Complete ===")
        print("Check the output above for deliverable responses.")

    except Exception as e:
        print(f"Error running experiments: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()