#!/usr/bin/env python3
"""
Dataset Encoding Script for CS336 Assignment 1
Encode TinyStories and OpenWebText datasets to uint16 NumPy arrays
"""

import time
import numpy as np
from pathlib import Path
from bpe_tokenizer import Tokenizer

def encode_dataset_streaming(file_path: str, tokenizer: Tokenizer, output_file: str):
    """
    Encode a large dataset to uint16 NumPy array using streaming to manage memory.
    This is the production version for actual dataset encoding.
    """
    print(f"Encoding {file_path} to {output_file}...")

    # Check vocab size to confirm uint16 is appropriate
    max_token_id = max(tokenizer.vocab.keys())
    print(f"Maximum token ID: {max_token_id}")

    if max_token_id >= 65536:  # 2^16
        print(f"Warning: Token ID {max_token_id} exceeds uint16 range!")
        return None

    file_size_gb = Path(file_path).stat().st_size / (1024**3)
    print(f"File size: {file_size_gb:.1f}GB")

    # Process file in chunks to manage memory
    chunk_size = 16 * 1024 * 1024  # 16MB chunks for better performance
    total_tokens = 0
    batch_tokens = []
    batch_size_limit = 50_000_000  # 50M tokens per batch to manage memory
    batch_num = 0

    print("Starting streaming encoding...")
    start_time = time.time()

    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Encode chunk
            try:
                token_ids = tokenizer.encode(chunk)
                batch_tokens.extend(token_ids)
                total_tokens += len(token_ids)

                # Progress update
                if total_tokens % 10_000_000 == 0:  # Every 10M tokens
                    elapsed = time.time() - start_time
                    rate = total_tokens / elapsed / 1000  # tokens per ms
                    print(f"  Processed {total_tokens:,} tokens ({rate:.1f}k tokens/sec)")

                # Save batch when limit reached
                if len(batch_tokens) >= batch_size_limit:
                    batch_array = np.array(batch_tokens, dtype=np.uint16)

                    if batch_num == 0:
                        # First batch - create new file
                        np.save(output_file, batch_array)
                        print(f"  Saved batch {batch_num}: {len(batch_tokens):,} tokens")
                    else:
                        # Subsequent batches - append
                        existing = np.load(output_file)
                        combined = np.concatenate([existing, batch_array])
                        np.save(output_file, combined)
                        print(f"  Saved batch {batch_num}: {len(batch_tokens):,} tokens (total: {len(combined):,})")

                    batch_tokens = []
                    batch_num += 1

            except Exception as e:
                print(f"Warning: Error encoding chunk: {e}")
                continue

    # Save remaining tokens
    if batch_tokens:
        batch_array = np.array(batch_tokens, dtype=np.uint16)

        if batch_num == 0:
            # Only one batch
            np.save(output_file, batch_array)
        else:
            # Append final batch
            existing = np.load(output_file)
            combined = np.concatenate([existing, batch_array])
            np.save(output_file, combined)
        print(f"  Saved final batch: {len(batch_tokens):,} tokens")

    end_time = time.time()
    encoding_time = end_time - start_time

    # Load final array for stats
    final_array = np.load(output_file)

    print(f"\nEncoding complete:")
    print(f"  Total tokens: {len(final_array):,}")
    print(f"  Encoding time: {encoding_time/60:.1f} minutes")
    print(f"  Throughput: {len(final_array)/encoding_time:.0f} tokens/sec")
    print(f"  Array shape: {final_array.shape}")
    print(f"  Array dtype: {final_array.dtype}")
    print(f"  File size: {final_array.nbytes / (1024**3):.2f}GB")

    # Calculate compression ratio
    original_size_gb = Path(file_path).stat().st_size / (1024**3)
    compressed_size_gb = final_array.nbytes / (1024**3)
    compression_ratio = original_size_gb / compressed_size_gb

    print(f"  Compression: {original_size_gb:.1f}GB → {compressed_size_gb:.2f}GB ({compression_ratio:.1f}x)")

    return len(final_array)

def encode_tinystories():
    """Encode TinyStories dataset."""
    print("=== Encoding TinyStories Dataset ===")

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        'tinystories_vocab.json',
        'tinystories_merges.pkl',
        ['<|endoftext|>']
    )

    # Encode training dataset
    tokens = encode_dataset_streaming(
        '../data/TinyStoriesV2-GPT4-train.txt',
        tokenizer,
        'tinystories_train_tokens.npy'
    )

    return tokens

def encode_openwebtext():
    """Encode OpenWebText dataset."""
    print("=== Encoding OpenWebText Dataset ===")

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        'owt_vocab.json',
        'owt_merges.pkl',
        ['<|endoftext|>']
    )

    # Encode training dataset
    tokens = encode_dataset_streaming(
        '../data/owt_train.txt',
        tokenizer,
        'owt_train_tokens.npy'
    )

    return tokens

def main():
    """Main function to encode both datasets."""
    print("=== Dataset Encoding for CS336 Assignment 1 ===")
    print("Encoding training datasets to uint16 NumPy arrays...")

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

    start_time = time.time()

    # Encode TinyStories
    ts_tokens = encode_tinystories()

    print("\n" + "="*50)

    # Encode OpenWebText
    owt_tokens = encode_openwebtext()

    total_time = time.time() - start_time

    print(f"\n=== Encoding Summary ===")
    print(f"TinyStories tokens: {ts_tokens:,}" if ts_tokens else "TinyStories: Failed")
    print(f"OpenWebText tokens: {owt_tokens:,}" if owt_tokens else "OpenWebText: Failed")
    print(f"Total encoding time: {total_time/60:.1f} minutes")

    print(f"\nFiles generated:")
    print(f"  - tinystories_train_tokens.npy")
    print(f"  - owt_train_tokens.npy")

    print(f"\nDeliverable (d): uint16 is appropriate because our vocabulary sizes (10K and 32K) are well below the uint16 maximum of 65,536, providing efficient 2-byte storage per token while supporting future vocabulary expansion.")

if __name__ == "__main__":
    main()