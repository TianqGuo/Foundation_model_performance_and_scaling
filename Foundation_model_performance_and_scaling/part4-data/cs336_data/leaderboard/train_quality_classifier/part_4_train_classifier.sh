#!/bin/bash
# ==============================================================================
# Section 4 – Train Quality Classifier
# ==============================================================================
#
# USAGE:
#   cd cs336_data/leaderboard/train_quality_classifier
#   ./part_4_train_classifier.sh
#
# WHAT IT DOES:
#   Downloads the Wikipedia URL list (~500 MB) if not present, then trains
#   the fastText quality classifier using the first WET file in data/CC/.
#
# OUTPUT:
#   cs336_data/assets/quality_classifier.bin
#
# NOTES:
#   - Requires at least 1 WET file in data/CC/ (run part_4_download.sh first)
#   - Training takes ~5-10 minutes
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

WIKI_URL="https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz"
WIKI_PATH="${ROOT}/data/wiki/enwiki-20240420-extracted_urls.txt.gz"
MODEL_PATH="${ROOT}/cs336_data/assets/quality_classifier.bin"
CC_WET=$(ls "${ROOT}/data/CC/"*.warc.wet.gz 2>/dev/null | head -1)

echo "=== Train Quality Classifier ==="

if [ -f "${MODEL_PATH}" ]; then
    echo "Model already exists at ${MODEL_PATH}, skipping."
    exit 0
fi

if [ -z "${CC_WET}" ]; then
    echo "ERROR: No WET files found in ${ROOT}/data/CC/"
    echo "       Run part_4_download.sh first."
    exit 1
fi

mkdir -p "${ROOT}/data/wiki" "${ROOT}/cs336_data/assets"

if [ ! -f "${WIKI_PATH}" ]; then
    echo "Downloading Wikipedia URL list (~500 MB) ..."
    wget -q --show-progress "${WIKI_URL}" -O "${WIKI_PATH}"
fi

echo "Training classifier using ${CC_WET} ..."
cd "${ROOT}"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m cs336_data.filtering_cc.quality_classifier.train \
    --wiki-urls "${WIKI_PATH}" \
    --cc-warc   "${CC_WET}" \
    --output    "${MODEL_PATH}" \
    --n-wiki 1000 --n-cc 1000 --n-try-wiki 4000 --workers 16

echo ""
echo "=== Done. Model saved to ${MODEL_PATH} ==="
