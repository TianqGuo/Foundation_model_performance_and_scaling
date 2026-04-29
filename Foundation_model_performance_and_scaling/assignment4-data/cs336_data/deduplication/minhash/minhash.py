"""
MinHash + LSH document deduplication (Section 3.2).

Pipeline:
  1. Normalize text (lowercase, remove punctuation/accents, NFD).
  2. Compute word n-gram sets.
  3. Build MinHash signatures using k hash functions via linear hashing.
  4. LSH: bucket documents by band; collect candidate duplicate pairs.
  5. Verify candidates with true Jaccard similarity.
  6. Cluster verified pairs with union-find; keep one per cluster.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

# Mersenne prime 2^31-1 — keeps a*h+b within int64 range (max ≈ 2^62)
_P = np.int64((1 << 31) - 1)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    # NFD + strip combining marks (accents)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# N-grams
# ---------------------------------------------------------------------------

def _word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------

def _hash_params(num_hashes: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a = rng.randint(1, int(_P), size=num_hashes, dtype=np.int64)
    b = rng.randint(0, int(_P), size=num_hashes, dtype=np.int64)
    return a, b


def _base_hash(ngram: str) -> np.int64:
    raw = int.from_bytes(hashlib.md5(ngram.encode()).digest()[:4], "little")
    return np.int64(raw % int(_P))


def _signature(ngrams: set[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sig = np.full(len(a), int(_P), dtype=np.int64)
    for ng in ngrams:
        h = _base_hash(ng)
        hashes = (a * h + b) % _P
        np.minimum(sig, hashes, out=sig)
    return sig


# ---------------------------------------------------------------------------
# LSH banding
# ---------------------------------------------------------------------------

def _band_keys(sig: np.ndarray, num_bands: int) -> list[bytes]:
    r = len(sig) // num_bands
    return [sig[b * r : (b + 1) * r].tobytes() for b in range(num_bands)]


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self) -> None:
        self._p: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._p:
            self._p[x] = x
        if self._p[x] != x:
            self._p[x] = self.find(self._p[x])
        return self._p[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._p[px] = py


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
    seed: int = 42,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Sort for deterministic keep selection
    paths = sorted(Path(p) for p in input_files)
    a, b = _hash_params(num_hashes, seed)

    # Read documents; compute normalized n-gram sets and signatures
    texts: list[str] = []
    doc_ngrams: list[set[str]] = []
    sigs: list[np.ndarray] = []

    for path in paths:
        text = path.read_text(encoding="utf-8")
        texts.append(text)
        norm = _normalize(text)
        ng = _word_ngrams(norm, ngrams)
        doc_ngrams.append(ng)
        sigs.append(_signature(ng, a, b))

    # LSH: group docs into buckets per band
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band_idx, key in enumerate(_band_keys(sig, num_bands)):
            buckets[(band_idx, key)].append(i)

    # Collect candidate pairs (share at least one bucket)
    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u, v = members[i], members[j]
                candidates.add((min(u, v), max(u, v)))

    # Verify with true Jaccard; cluster confirmed duplicates
    uf = _UnionFind()
    for i, j in candidates:
        if _jaccard(doc_ngrams[i], doc_ngrams[j]) >= jaccard_threshold:
            uf.union(i, j)

    # Keep the lowest-index member of each cluster
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(paths)):
        clusters[uf.find(i)].append(i)

    to_keep: set[int] = {min(members) for members in clusters.values()}

    # Write kept documents to output directory
    for i, path in enumerate(paths):
        if i in to_keep:
            (output_directory / path.name).write_text(texts[i], encoding="utf-8")
