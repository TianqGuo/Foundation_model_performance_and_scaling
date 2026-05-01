#!/bin/bash
# ==============================================================================
# Section 3.2 – MinHash + LSH Document Deduplication (Problem: minhash_deduplication)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/deduplication/minhash
#   ./part_3_2.sh
#
# WHAT IT DOES:
#   Runs both minhash deduplication tests:
#     1. Exact-duplicate detection (doc1 == doc2 → keep one)
#     2. Fuzzy-duplicate detection (rails MIT ≈ react MIT → keep one)
#
#   Pipeline per test:
#     - Normalize text (lowercase, remove punctuation/accents, NFD)
#     - Compute word n-gram sets
#     - Build MinHash signatures with linear hash family (a*h+b mod P)
#     - LSH banding: bucket docs sharing a band as candidate pairs
#     - Verify candidates with true Jaccard similarity
#     - Union-Find clustering; keep lowest-index member per cluster
#
# OUTPUT:
#   Deduplicated files written to pytest's tmp_path (see test output).
#
# NOTES:
#   - num_hashes must be evenly divisible by num_bands.
#   - Text normalization follows Penedo et al. (2023): lowercase, remove
#     punctuation, normalize whitespace, remove accents, NFD unicode.
#   - Keep selection is deterministic: lowest-index (alphabetically first
#     filename) member of each duplicate cluster is retained.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

echo "=== Section 3.2: MinHash + LSH Deduplication ==="
cd "${ROOT}"
.venv/bin/pytest -k "test_minhash_deduplication" -v

echo ""
echo "=== Done ==="
