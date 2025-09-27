import regex as re
from collections import defaultdict
import heapq
from typing import Dict, List, Tuple

# Regex pattern from GPT-2 tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class ReverseLexOrderPair:
    """
    Encapsulates (bytes, bytes) so that in a min-heap, the "largest in normal lex order"
    is treated as the smallest. Ensures that tie frequencies pop in reverse lex order.
    """

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        # Invert normal order: self < other if self is > other (so larger lex sorts first).
        return self.pair > other.pair

    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair


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
        special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        text_chunks = [
            chunk
            for chunk in re.split(f'({special_pattern})', text)
            if chunk and chunk not in special_tokens
        ]
    else:
        text_chunks = [text]

    # Pre-tokenize using regex pattern - store as dict of frequencies
    freqs: dict[tuple[bytes], int] = {}
    pat_compiled = re.compile(PAT)
    for chunk in text_chunks:
        for match in pat_compiled.finditer(chunk):
            token_str = match.group()
            token_bytes = token_str.encode('utf-8')
            # Convert to tuple of single-byte objects for compatibility with reference
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            freqs[token_tuple] = freqs.get(token_tuple, 0) + 1

    # Calculate number of merges needed
    max_merges = vocab_size - len(vocab)  # Account for initial vocab + special tokens
    merges: list[tuple[bytes, bytes]] = []

    # Build initial pair frequencies using reference approach
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)

    # Build a max-heap by pushing negative frequencies (for min-heap behavior)
    pair_heap = []
    for p, f in pair_freqs.items():
        if f > 0:
            heapq.heappush(pair_heap, (-f, ReverseLexOrderPair(p), p))

    # BPE training loop
    for i in range(max_merges):
        if not pair_heap:
            break

        # Note that only the pairs_to_keys and pair_freqs were updated in the merge_tokens function process.
        # And only new pair were added to pair_heap. The old/modified ones in the pair_heap were not touched.
        # That's why we need the following loop to check the stale pairs in the pair_heep.

        # Pop until we find the top pair that still matches pair_freqs
        while pair_heap:
            neg_freq, _, top_pair = heapq.heappop(pair_heap)
            freq = -neg_freq
            if pair_freqs.get(top_pair, 0) == freq:
                pair = top_pair
                break
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[top_pair], ReverseLexOrderPair(top_pair), top_pair))
        else:
            # If pair_heap is empty after the loop, we are done
            break

        if pair_freqs.get(pair, 0) <= 0:
            break

        # Add this new merge token to vocab and record the merge
        vocab[next_id] = pair[0] + pair[1]
        merges.append(pair)

        # Merge in freqs, then update the heap for pairs changed by this merge
        changed_pairs = merge_tokens(freqs, pair_freqs, pairs_to_keys, pair)
        for cp in changed_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[cp], ReverseLexOrderPair(cp), cp))

        next_id += 1

    return vocab, merges


def get_pair_freqs(
    freqs: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """
    Builds a pair-frequency table and reverse mapping (pair -> set of keys).
    """
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)

    for symbols, freq in freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freqs[pair] += freq
            pairs_to_keys[pair].add(symbols)

    return pair_freqs, pairs_to_keys


def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    """Replaces every occurrence of pair=(x,y) in old_repr with the merged symbol x+y."""
    new_symbols = []
    i = 0
    while i < len(old_repr):
        if i < len(old_repr) - 1 and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbols.append(old_repr[i] + old_repr[i + 1])  # merges, e.g. b'A' + b'B' => b'AB'
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1
    return tuple(new_symbols)


def merge_tokens(
    freqs: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
    pair: tuple[bytes, bytes],
) -> set[tuple[bytes, bytes]]:
    """Merges 'pair' into freqs and updates pair_freqs & pairs_to_keys for all affected old/new keys."""
    changed_pairs = set()
    keys_to_modify = pairs_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = freqs.pop(old_key)
        new_key = build_new_repr(old_key, pair)

        # Decrement frequencies in pair_freqs for old_key's adjacencies
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            pair_freqs[left, right] -= old_freq
            changed_pairs.add((left, right))
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left, right].discard(old_key)

        # Increment frequencies for new_key's adjacencies
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            pair_freqs[left, right] += old_freq
            changed_pairs.add((left, right))
            pairs_to_keys[left, right].add(new_key)

        # Put new_key back with updated freq
        freqs[new_key] = freqs.get(new_key, 0) + old_freq

    pairs_to_keys[pair] = set()
    return changed_pairs


# Example usage
if __name__ == "__main__":
    # This would need actual file and parameters
    vocab, merges = train_bpe("../data/dummy_test_file.txt", 1000, ["<|endoftext|>"])
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")