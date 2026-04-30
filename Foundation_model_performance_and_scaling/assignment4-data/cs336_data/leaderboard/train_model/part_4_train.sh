#!/bin/bash
# ==============================================================================
# Section 4 – Train Model (Problem: train_model)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/leaderboard/train_model
#   WANDB_ENTITY=<your-entity> ./part_4_train.sh
#
# WHAT IT DOES:
#   1. Patches cs336-basics/configs/experiment/your_data.yaml with the correct
#      train_bin path (data/tokenized/train.bin) and wandb_entity.
#   2. Launches multi-GPU training via torchrun (auto-detects GPU count).
#
# PREREQUISITES:
#   - Tokenized data at data/tokenized/train.bin  (run part_4_tokenize.sh first)
#   - WANDB_ENTITY env var set (or edit your_data.yaml manually)
#   - cs336-basics .venv synced  (setup_vm.sh handles this)
#
# ENVIRONMENT VARIABLES:
#   WANDB_ENTITY   Your wandb username / team (required, no default)
#   GPUS           Number of GPUs to use (default: auto-detect)
#   TRAIN_STEPS    Training steps override (default: 100000)
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
BASICS_DIR="${ROOT}/cs336-basics"
TRAIN_BIN="${ROOT}/data/tokenized/train.bin"
VALID_BIN="${ROOT}/data/paloma/tokenized_paloma_c4_100_domains_validation.bin"
CONFIG="${BASICS_DIR}/configs/experiment/your_data.yaml"

# ── Validate prerequisites ───────────────────────────────────────────────────

if [ ! -f "${TRAIN_BIN}" ]; then
    echo "ERROR: Tokenized data not found at ${TRAIN_BIN}"
    echo "       Run cs336_data/leaderboard/tokenize_data/part_4_tokenize.sh first."
    exit 1
fi

if [ ! -f "${VALID_BIN}" ]; then
    echo "Paloma validation file not found. Downloading now ..."
    bash "${ROOT}/cs336_data/leaderboard/download_paloma/part_4_download_paloma.sh"
fi

if [ -z "${WANDB_ENTITY}" ]; then
    echo "WARNING: WANDB_ENTITY is not set — wandb entity will be inferred from login credentials."
fi

# ── Detect GPU count ─────────────────────────────────────────────────────────

if [ -n "${GPUS}" ]; then
    NPROC="${GPUS}"
elif command -v nvidia-smi &>/dev/null; then
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NPROC=$(( NPROC > 0 ? NPROC : 1 ))
else
    NPROC=1
fi

echo "=== Part 4: Train Model ==="
echo "  Train bin:    ${TRAIN_BIN}"
echo "  wandb entity: ${WANDB_ENTITY}"
echo "  GPUs:         ${NPROC}"
echo "  Steps:        100000 (default from train_config.py)"
echo ""

# ── Patch your_data.yaml ─────────────────────────────────────────────────────
# Use Python + ruamel.yaml so we don't corrupt the YAML with sed
"${BASICS_DIR}/.venv/bin/python" - <<PYEOF
from pathlib import Path
import re

config_path = Path("${CONFIG}")
text = config_path.read_text()

# Replace train_bin path
text = re.sub(
    r"(train_bin:\s*).*",
    r"\g<1>${TRAIN_BIN}",
    text,
)
# Replace valid_bin path
text = re.sub(
    r"(valid_bin:\s*).*",
    r"\g<1>${VALID_BIN}",
    text,
)
# Replace wandb_entity only if WANDB_ENTITY is explicitly set
if "${WANDB_ENTITY}":
    text = re.sub(
        r"(wandb_entity:\s*).*",
        r"\g<1>${WANDB_ENTITY}",
        text,
    )
config_path.write_text(text)
print(f"Patched {config_path}")
PYEOF

# ── Launch training ───────────────────────────────────────────────────────────

cd "${BASICS_DIR}"

if [ "${NPROC}" -gt 1 ]; then
    echo "Launching multi-GPU training (${NPROC} GPUs) …"
    .venv/bin/torchrun \
        --standalone \
        --nproc_per_node="${NPROC}" \
        scripts/train.py \
        --config-name=experiment/your_data
else
    echo "Launching single-GPU training …"
    .venv/bin/python scripts/train.py \
        --config-name=experiment/your_data
fi

echo ""
echo "=== Training complete ==="
