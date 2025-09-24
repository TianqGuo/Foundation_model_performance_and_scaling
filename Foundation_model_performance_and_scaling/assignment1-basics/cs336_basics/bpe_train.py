import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple

# Regex pattern from GPT-2 tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to training text file
        vocab_size: Maximum vocabulary size (including initial 256 bytes + special tokens)
        special_tokens: List of special tokens to preserve

    Returns:
        vocab: dict[int, bytes] - mapping from token ID to token bytes
        merges: list[tuple[bytes, bytes]] - list of merge operations
    """

    # Read input text
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize vocabulary with 256 byte values
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # Add special tokens to vocabulary
    special_token_ids = {}
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        vocab[next_id] = token_bytes
        special_token_ids[token] = next_id
        next_id += 1

    # Split text on special tokens to avoid merging across boundaries
    if special_tokens:
        # Create regex pattern to split on special tokens
        special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        text_chunks = re.split(f'({special_pattern})', text)
        # Keep only non-special-token chunks for pre-tokenization
        text_chunks = [chunk for chunk in text_chunks if chunk not in special_tokens]
        text = ''.join(text_chunks)

    # Pre-tokenize using regex pattern
    pre_tokens = []
    for match in re.finditer(PAT, text):
        token_str = match.group()
        token_bytes = token_str.encode('utf-8')
        pre_tokens.append(list(token_bytes))  # Convert to list of byte values

    # Calculate number of merges needed
    max_merges = vocab_size - len(vocab)  # Account for initial vocab + special tokens
    merges: list[tuple[bytes, bytes]] = []

    # BPE training loop
    for _ in range(max_merges):
        # Count adjacent byte pairs within each pre-token
        pair_counts = defaultdict(int)

        for pre_token in pre_tokens:
            if len(pre_token) < 2:
                continue
            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                pair_counts[pair] += 1

        if not pair_counts:
            break

        # Find most frequent pair (with lexicographic tie-breaking)
        max_count = max(pair_counts.values())
        most_frequent_pairs = [(pair, count) for pair, count in pair_counts.items() if count == max_count]

        # Break ties lexicographically by choosing the maximum pair
        best_pair = max(most_frequent_pairs, key=lambda x: x[0])[0]

        # Create new token for the merged pair
        token1_bytes = vocab[best_pair[0]] if best_pair[0] in vocab else bytes([best_pair[0]])
        token2_bytes = vocab[best_pair[1]] if best_pair[1] in vocab else bytes([best_pair[1]])

        merged_bytes = token1_bytes + token2_bytes
        vocab[next_id] = merged_bytes

        # Record the merge (as bytes, not int IDs)
        merges.append((token1_bytes, token2_bytes))

        # Replace all occurrences of the pair in pre-tokens
        new_pre_tokens = []
        for pre_token in pre_tokens:
            new_token = merge_pair_in_token(pre_token, best_pair, next_id)
            new_pre_tokens.append(new_token)
        pre_tokens = new_pre_tokens

        next_id += 1

    return vocab, merges


def merge_pair_in_token(token: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Merge a specific pair in a token with a new ID."""
    result = []
    i = 0
    while i < len(token):
        if i < len(token) - 1 and token[i] == pair[0] and token[i + 1] == pair[1]:
            # Found the pair to merge
            result.append(new_id)
            i += 2  # Skip both elements of the pair
        else:
            result.append(token[i])
            i += 1
    return result


# Example usage
if __name__ == "__main__":
    # This would need actual file and parameters
    vocab, merges = train_bpe("../data/dummy_test_file.txt", 1000, ["<|endoftext|>"])
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")