#!/bin/bash
# ==============================================================================
# Section 4 – Download Paloma Validation Set
# ==============================================================================
#
# USAGE:
#   cd cs336_data/leaderboard/download_paloma
#   ./part_4_download_paloma.sh
#
# WHAT IT DOES:
#   Downloads the Paloma C4-100-domains validation split from HuggingFace
#   and tokenizes it with the GPT-2 tokenizer, reproducing the file at
#   /data/paloma/tokenized_paloma_c4_100_domains_validation.bin on the cluster.
#
# OUTPUT:
#   data/paloma/tokenized_paloma_c4_100_domains_validation.bin
#
# NOTES:
#   - Only needed on cloud VMs; Together cluster has this file pre-loaded.
#   - Requires the 'datasets' package (included in cs336_data dependencies).
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

OUTPUT="${ROOT}/data/paloma/tokenized_paloma_c4_100_domains_validation.bin"

echo "=== Download Paloma Validation Set ==="
echo "  Output: ${OUTPUT}"
echo ""

if [ -f "${OUTPUT}" ]; then
    echo "Already exists, skipping."
    exit 0
fi

# On Together cluster, symlink instead of downloading
if [ -f "/data/paloma/tokenized_paloma_c4_100_domains_validation.bin" ]; then
    echo "Found cluster copy, creating symlink ..."
    mkdir -p "${ROOT}/data/paloma"
    ln -s "/data/paloma/tokenized_paloma_c4_100_domains_validation.bin" "${OUTPUT}"
    echo "Done."
    exit 0
fi

cd "${ROOT}"
.venv/bin/python -m cs336_data.leaderboard.download_paloma.download_paloma \
    --output "${OUTPUT}"

echo ""
echo "=== Done. Validation data at ${OUTPUT} ==="
