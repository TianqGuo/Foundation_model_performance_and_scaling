#!/bin/bash
# ==============================================================================
# Cloud VM Setup — assignment4-data
# ==============================================================================
#
# Run this ONCE after cloning the repo on a new cloud VM (e.g., vast.ai A100).
# It installs uv, syncs both Python environments, downloads model assets,
# and creates the required data directories.
#
# USAGE:
#   cd assignment4-data
#   bash setup_vm.sh
#
# AFTER SETUP, run the pipeline in order:
#   1. Download WET files:
#        cs336_data/leaderboard/download_wet/part_4_download.sh 600
#   2. Filter each WET file:
#        cs336_data/leaderboard/filter_data/part_4_filter.sh
#   3. Tokenize filtered data:
#        cs336_data/leaderboard/tokenize_data/part_4_tokenize.sh
#   4. Train the model:
#        export WANDB_ENTITY=<your-wandb-username>
#        cs336_data/leaderboard/train_model/part_4_train.sh
#
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  assignment4-data VM setup"
echo "  Working directory: ${SCRIPT_DIR}"
echo "============================================================"
echo ""

# ── 1. Install uv ─────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="${HOME}/.local/bin:${PATH}"
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# ── 2. Install cs336_data dependencies ────────────────────────────────────────
echo ""
echo "[2/5] Syncing cs336_data (assignment4-data) …"
uv sync

# ── 3. Install cs336-basics dependencies ──────────────────────────────────────
echo ""
echo "[3/5] Syncing cs336-basics …"
cd "${SCRIPT_DIR}/cs336-basics"
uv sync
cd "${SCRIPT_DIR}"

# ── 4. Download model assets (fastText lid, NSFW, hate-speech models) ─────────
echo ""
echo "[4/5] Downloading classifier assets …"
mkdir -p cs336_data/assets
bash get_assets.sh

# ── 5. Create data directories ────────────────────────────────────────────────
echo ""
echo "[5/5] Creating data directories …"
mkdir -p data/CC data/filtered data/tokenized results/leaderboard

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Download WET files (e.g., 600 for a full run):"
echo "       cs336_data/leaderboard/download_wet/part_4_download.sh 600"
echo ""
echo "  2. Filter all downloaded WET files:"
echo "       cs336_data/leaderboard/filter_data/part_4_filter.sh"
echo ""
echo "  3. Tokenize the filtered data:"
echo "       cs336_data/leaderboard/tokenize_data/part_4_tokenize.sh"
echo ""
echo "  4. Train (set your wandb entity first):"
echo "       export WANDB_ENTITY=<your-wandb-username>"
echo "       cs336_data/leaderboard/train_model/part_4_train.sh"
echo "============================================================"
