#!/bin/bash
# ==============================================================================
# Section 2.7 – Quality Classifier (Problem: quality_classifier)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/filtering_cc/quality_classifier
#   ./part_2_7.sh
#
# WHAT IT DOES:
#   1. Downloads the Wikipedia extracted-URL list (if not present).
#   2. Trains a fastText binary classifier (wiki vs cc).
#   3. Runs the quality classifier test.
#
# OUTPUT:
#   cs336_data/assets/quality_classifier.bin
#
# NOTES:
#   - The Wikipedia URL file is ~500 MB compressed. Download only runs once.
#   - Training scrapes ~8000 Wikipedia-linked URLs (30 parallel workers).
#     Expect ~5–15 minutes depending on network speed.
#   - On the Together cluster: wiki URLs are at /data/wiki/ and are symlinked
#     automatically; no download needed.
#   - Adjust --n-wiki, --n-cc, --n-try-wiki to trade speed vs. accuracy.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

WIKI_URL="https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz"
WIKI_DIR="${ROOT}/data/wiki"
WIKI_PATH="${WIKI_DIR}/enwiki-20240420-extracted_urls.txt.gz"
WARC_PATH="${ROOT}/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
MODEL_PATH="${ROOT}/cs336_data/assets/quality_classifier.bin"

# ── Step 1: Acquire Wikipedia URL file ───────────────────────────────────────
echo "=== Step 1: Acquire Wikipedia URL file ==="
if [ -e "${MODEL_PATH}" ]; then
    echo "✓ Trained model already exists, skipping training."
    echo "  Delete ${MODEL_PATH} to retrain."
    SKIP_TRAIN=1
elif [ -e "${WIKI_PATH}" ]; then
    echo "✓ Wiki URL file already exists, skipping download."
    SKIP_TRAIN=0
elif [ -f "/data/wiki/enwiki-20240420-extracted_urls.txt.gz" ]; then
    echo "→ Found wiki file on cluster, creating symlink..."
    mkdir -p "${WIKI_DIR}"
    ln -sf "/data/wiki/enwiki-20240420-extracted_urls.txt.gz" "${WIKI_PATH}"
    SKIP_TRAIN=0
else
    echo "↓ Downloading Wikipedia URL file (~500 MB)..."
    mkdir -p "${WIKI_DIR}"
    wget --show-progress -q "${WIKI_URL}" -O "${WIKI_PATH}"
    echo "✓ Download complete."
    SKIP_TRAIN=0
fi
echo ""

# ── Step 2: Train classifier ──────────────────────────────────────────────────
if [ "${SKIP_TRAIN}" = "0" ]; then
    echo "=== Step 2: Train quality classifier ==="
    cd ../../..
    .venv/bin/python -m cs336_data.filtering_cc.quality_classifier.train \
        --wiki-urls "${WIKI_PATH}" \
        --cc-warc   "${WARC_PATH}" \
        --output    "${MODEL_PATH}" \
        --n-wiki    1000 \
        --n-cc      1000 \
        --n-try-wiki 4000 \
        --workers   30
    cd cs336_data/filtering_cc/quality_classifier
    echo ""
else
    echo "=== Step 2: Skipped (model exists) ==="
    echo ""
fi

# ── Step 3: Run test ──────────────────────────────────────────────────────────
echo "=== Step 3: Run quality classifier test ==="
cd ../../..
.venv/bin/pytest -k test_classify_quality -v

echo ""
echo "=== Done ==="