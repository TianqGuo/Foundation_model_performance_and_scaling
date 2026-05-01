#!/bin/bash
# ==============================================================================
# Section 4 – Tokenize Data (Problem: tokenize_data)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/leaderboard/tokenize_data
#   ./part_4_tokenize.sh
#
# WHAT IT DOES:
#   Reads all filtered .txt files in data/filtered/ (one document per line),
#   tokenizes each document with the GPT-2 tokenizer, appends <|endoftext|>
#   after each document, and writes all token IDs as a np.uint16 binary file.
#
# OUTPUT:
#   data/tokenized/train.bin   — np.uint16 array of all token IDs
#
# NOTES:
#   - Requires filtered data from part_4_filter.sh to exist in data/filtered/.
#   - Uses multiprocessing (one worker per CPU core) for fast tokenization.
#   - Output format is compatible with cs336-basics training script:
#       np.fromfile("train.bin", dtype=np.uint16)
#   - Token count is printed at the end — needed for the written answer.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

INPUT_DIR="${ROOT}/data/filtered"
OUTPUT="${ROOT}/data/tokenized/train.bin"

echo "=== Part 4: Tokenize Data ==="
echo "  Input:  ${INPUT_DIR}"
echo "  Output: ${OUTPUT}"
echo ""

cd "${ROOT}"
.venv/bin/python -m cs336_data.leaderboard.tokenize_data.tokenize \
    --input-dir "${INPUT_DIR}" \
    --output    "${OUTPUT}"

echo ""
echo "=== Done. Tokenized data at ${OUTPUT} ==="
