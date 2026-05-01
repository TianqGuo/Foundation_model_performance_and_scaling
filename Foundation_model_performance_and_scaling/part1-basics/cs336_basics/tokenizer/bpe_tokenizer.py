import regex as re
import json
import pickle
from typing import Dict, List, Tuple, Iterator, Iterable, Optional


# Same regex pattern from GPT-2 tokenizer used in training
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    BPE Tokenizer for encoding text to token IDs and decoding token IDs to text.

    This tokenizer applies the same BPE merges that were learned during training
    to encode text into token sequences, and can decode token sequences back to text.
    """

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and special tokens.

        Args:
            vocab: Mapping from token ID to token bytes
            merges: List of BPE merge operations in order of creation
            special_tokens: Optional list of special token strings
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()

        # Create reverse vocabulary for encoding (bytes -> token_id)
        self.byte_to_id = {v: k for k, v in self.vocab.items()}

        # Add special tokens to vocabulary if provided
        self.special_tokens = special_tokens or []
        self._add_special_tokens()

        # Create merge ranking for efficient merge application
        self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}

        # Compile regex pattern for pre-tokenization
        self.pat = re.compile(PAT)

        # Create special token patterns for splitting
        if self.special_tokens:
            special_pattern = '|'.join(re.escape(token) for token in self.special_tokens)
            self.special_pat = re.compile(f'({special_pattern})')
        else:
            self.special_pat = None

    def _add_special_tokens(self):
        """Add special tokens to vocabulary if they're not already present."""
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0

        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.byte_to_id:
                self.vocab[next_id] = token_bytes
                self.byte_to_id[token_bytes] = next_id
                next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None) -> 'Tokenizer':
        """
        Class method that constructs a Tokenizer from serialized files.

        Args:
            vocab_filepath: Path to JSON file containing vocabulary
            merges_filepath: Path to pickle file containing merges
            special_tokens: Optional list of special token strings

        Returns:
            Tokenizer instance
        """
        # Load vocabulary from JSON file
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        # Convert back to proper format (int keys, bytes values)
        vocab = {}
        for k, v in vocab_data.items():
            token_id = int(k)
            token_bytes = v.encode('utf-8')
            vocab[token_id] = token_bytes

        # Load merges from pickle file
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _get_pairs(self, tokens: List[bytes]) -> set:
        """Get all adjacent pairs in a token sequence."""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def _apply_bpe(self, token_bytes: List[bytes]) -> List[bytes]:
        """
        Apply BPE merges to a sequence of bytes, following the merge order from training.

        Args:
            token_bytes: List of individual byte tokens

        Returns:
            List of merged byte tokens
        """
        if len(token_bytes) <= 1:
            return token_bytes

        # Keep applying merges until no more are possible
        while True:
            pairs = self._get_pairs(token_bytes)
            if not pairs:
                break

            # Find the highest priority merge (earliest in training)
            bigram = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float('inf')))

            # If no valid merge found, we're done
            if bigram not in self.merge_ranks:
                break

            # Apply the merge
            token_bytes = self._merge_tokens(token_bytes, bigram)

        return token_bytes

    def _merge_tokens(self, tokens: List[bytes], pair: Tuple[bytes, bytes]) -> List[bytes]:
        """
        Merge all occurrences of a specific pair in the token sequence.

        Args:
            tokens: Current token sequence
            pair: Pair to merge (first, second)

        Returns:
            New token sequence with pair merged
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                # Merge the pair
                merged = tokens[i] + tokens[i + 1]
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text using the same method as training.
        Handles special tokens by splitting on them first, prioritizing longer special tokens.

        Args:
            text: Input text string

        Returns:
            List of pre-token strings
        """
        # First split on special tokens if any
        if self.special_pat:
            # Handle overlapping special tokens by prioritizing longer ones
            result = []
            i = 0
            while i < len(text):
                # Check for special tokens starting at position i, prioritizing longer ones
                found_special = None
                for special_token in sorted(self.special_tokens, key=len, reverse=True):
                    if text[i:].startswith(special_token):
                        found_special = special_token
                        break

                if found_special:
                    # Found a special token
                    result.append(found_special)
                    i += len(found_special)
                else:
                    # Find the next special token or end of string
                    next_special_pos = len(text)
                    for special_token in self.special_tokens:
                        pos = text.find(special_token, i)
                        if pos != -1 and pos < next_special_pos:
                            next_special_pos = pos

                    # Extract text up to next special token
                    regular_text = text[i:next_special_pos]
                    if regular_text:
                        # Apply regular pre-tokenization
                        for match in self.pat.finditer(regular_text):
                            result.append(match.group())
                    i = next_special_pos

            return result
        else:
            # No special tokens, just apply regular pre-tokenization
            return [match.group() for match in self.pat.finditer(text)]

    def encode(self, text: str) -> List[int]:
        """
        Encode input text into a sequence of token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        if not text:
            return []

        # Step 1: Pre-tokenize the text
        pre_tokens = self._pre_tokenize(text)

        # Step 2: Process each pre-token independently
        all_ids = []
        for pre_token in pre_tokens:
            # Handle special tokens directly
            if pre_token in self.special_tokens:
                token_bytes = pre_token.encode('utf-8')
                if token_bytes in self.byte_to_id:
                    all_ids.append(self.byte_to_id[token_bytes])
                continue

            # Convert pre-token to bytes
            token_bytes = [bytes([b]) for b in pre_token.encode('utf-8')]

            # Apply BPE merges
            merged_tokens = self._apply_bpe(token_bytes)

            # Convert to token IDs
            for token in merged_tokens:
                if token in self.byte_to_id:
                    all_ids.append(self.byte_to_id[token])
                else:
                    # This shouldn't happen if tokenizer is properly trained
                    # Fall back to individual bytes
                    for byte_val in token:
                        all_ids.append(byte_val)

        return all_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        This enables memory-efficient tokenization of large files.

        Args:
            iterable: Iterable of text strings (e.g., file lines)

        Yields:
            Token IDs one at a time
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        if not ids:
            return ""

        # Look up each ID in vocabulary and concatenate bytes
        byte_sequence = b''
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            else:
                # Invalid token ID - skip or handle gracefully
                # You could also raise an exception here
                continue

        # Decode bytes to string, replacing invalid sequences
        try:
            return byte_sequence.decode('utf-8', errors='replace')
        except Exception:
            # Fallback for any other decoding issues
            return byte_sequence.decode('utf-8', errors='replace')


# Example usage and testing
if __name__ == "__main__":
    # Example demonstrating the encoding process from the requirements
    print("=== BPE Tokenizer Example ===")

    # Example vocabulary and merges from the requirements
    example_vocab = {
        0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
        6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'
    }

    example_merges = [
        (b't', b'h'),      # Creates 'th'
        (b' ', b'c'),      # Creates ' c'
        (b' ', b'a'),      # Creates ' a'
        (b'th', b'e'),     # Creates 'the'
        (b' a', b't'),     # Creates ' at'
    ]

    # Create tokenizer
    tokenizer = Tokenizer(example_vocab, example_merges)

    # Test encoding
    input_text = "the cat ate"
    encoded = tokenizer.encode(input_text)
    print(f"Input: '{input_text}'")
    print(f"Encoded: {encoded}")

    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    print(f"Round-trip successful: {input_text == decoded}")

    # Test with special tokens
    print("\n=== Special Tokens Example ===")
    special_tokenizer = Tokenizer(example_vocab, example_merges, ["<|endoftext|>"])

    special_text = "the cat<|endoftext|>ate"
    special_encoded = special_tokenizer.encode(special_text)
    special_decoded = special_tokenizer.decode(special_encoded)

    print(f"Input: '{special_text}'")
    print(f"Encoded: {special_encoded}")
    print(f"Decoded: '{special_decoded}'")