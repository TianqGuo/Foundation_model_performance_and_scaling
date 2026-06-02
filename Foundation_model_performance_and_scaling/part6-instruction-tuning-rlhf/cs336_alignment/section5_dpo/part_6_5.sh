#!/usr/bin/env bash
# =============================================================================
# USAGE:   bash cs336_alignment/section5_dpo/part_6_5.sh [--smoke-test] [--no-wandb]
#
# WHAT IT DOES:
#   DPO fine-tunes the SFT checkpoint on Anthropic HH preference pairs.
#
#   Setup:
#     - π_ref (frozen) on cuda:1, π_θ (trained) on cuda:0
#     - 1 epoch, β=0.1, lr=1e-6, effective batch 64 via gradient accumulation
#     - RMSprop optimizer
#     - Validates implicit reward accuracy every 50 steps
#     - Saves best checkpoint (highest val accuracy) to assets/dpo_hh/best/
#
# OUTPUT:
#   results/section5/train_metrics_dpo_hh.jsonl   — per-step metrics
#   results/section5/final_val_dpo_hh.json        — final/best val accuracy
#   assets/dpo_hh/best/                           — best checkpoint
#   assets/dpo_hh/final/                          — final checkpoint
#
# NOTES:
#   Requires 2× 80 GB GPUs (one per model copy).
#   SFT checkpoint must be available at assets/sft_ultrachat/ or on HuggingFace Hub.
#   --smoke-test runs 20 optimizer steps for quick validation.
#   Data is downloaded automatically if not present.
# =============================================================================
set -e
ulimit -n 65536 2>/dev/null || true
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SMOKE_TEST=0
SKIP_JUDGES=0
NO_WANDB=""
BALANCE_SOURCES=""
for arg in "$@"; do
    case "$arg" in
        --smoke-test)        SMOKE_TEST=1 ;;
        --skip-judges)       SKIP_JUDGES=1 ;;
        --no-wandb)          NO_WANDB="--no-wandb" ;;
        --balance-sources)   BALANCE_SOURCES="--balance-sources" ;;
    esac
done

# ---------------------------------------------------------------------------
# resolve_model: cluster → local → HuggingFace download
# ---------------------------------------------------------------------------
resolve_data() {
    local cluster="$1" local_path="$2" download_cmd="$3" glob="${4:-}"
    _path_ok() {
        [ -e "$1" ] || return 1
        [ -z "${glob}" ] && return 0
        ls $1/${glob} >/dev/null 2>&1
    }
    if _path_ok "${cluster}"; then echo "${cluster}"
    elif _path_ok "${local_path}"; then echo "${local_path}"
    else
        echo "INFO: Downloading data -> ${local_path}" >&2
        eval "${download_cmd}"
        echo "${local_path}"
    fi
}

resolve_model() {
    local cluster="$1" local_path="$2" hf_repo="$3"
    _model_complete() { [ -f "${1}/config.json" ] && [ -f "${1}/tokenizer_config.json" ]; }
    if _model_complete "${cluster}"; then
        echo "${cluster}"
    elif _model_complete "${local_path}"; then
        echo "${local_path}"
    else
        echo "INFO: Downloading ${hf_repo} -> ${local_path}" >&2
        mkdir -p "${local_path}"
        local attempt=0
        while true; do
            attempt=$(( attempt + 1 ))
            if uv run huggingface-cli download "${hf_repo}" \
                    --local-dir "${local_path}" --max-workers 2 >&2; then
                break
            fi
            [ "${attempt}" -ge 3 ] && { echo "ERROR: Download failed." >&2; exit 1; }
            sleep 5
        done
        echo "${local_path}"
    fi
}

# ---------------------------------------------------------------------------
# SFT checkpoint
# ---------------------------------------------------------------------------
echo "==> Resolving SFT checkpoint ..."
SFT_MODEL=$(resolve_model \
    "/data/a5-alignment/sft_ultrachat" \
    "${ROOT}/assets/sft_ultrachat" \
    "sclion/llama-3.1-8b-sft-ultrachat")
echo "    ${SFT_MODEL}"

# ---------------------------------------------------------------------------
# Anthropic HH data
# ---------------------------------------------------------------------------
DATA_DIR="${ROOT}/data/hh"
mkdir -p "${DATA_DIR}"

echo "==> Checking Anthropic HH data ..."
CLUSTER_HH="/data/a5-alignment/hh"
HH_FILES="harmless-base.jsonl.gz helpful-base.jsonl.gz helpful-online.jsonl.gz helpful-rejection-sampled.jsonl.gz"

if [ -d "${CLUSTER_HH}" ]; then
    echo "    Using cluster data at ${CLUSTER_HH}"
    DATA_DIR="${CLUSTER_HH}"
else
    ALL_PRESENT=1
    for FILE in ${HH_FILES}; do
        [ ! -f "${DATA_DIR}/${FILE}" ] && ALL_PRESENT=0 && break
    done

    if [ "${ALL_PRESENT}" -eq 0 ]; then
        echo "    Downloading from HuggingFace ..."
        uv run python -c "
from huggingface_hub import hf_hub_download
import os, shutil, tempfile
# Actual HF paths are in subdirs: harmless-base/train.jsonl.gz, etc.
file_map = [
    ('harmless-base/train.jsonl.gz',          'harmless-base.jsonl.gz'),
    ('helpful-base/train.jsonl.gz',           'helpful-base.jsonl.gz'),
    ('helpful-online/train.jsonl.gz',         'helpful-online.jsonl.gz'),
    ('helpful-rejection-sampled/train.jsonl.gz', 'helpful-rejection-sampled.jsonl.gz'),
]
with tempfile.TemporaryDirectory() as tmp:
    for hf_path, local_name in file_map:
        dest = os.path.join('${DATA_DIR}', local_name)
        if not os.path.exists(dest):
            print(f'Downloading {hf_path} ...')
            path = hf_hub_download(repo_id='Anthropic/hh-rlhf', filename=hf_path,
                                   repo_type='dataset', local_dir=tmp)
            shutil.copy(path, dest)
            print(f'  -> {dest}')
"
    else
        echo "    Data already present at ${DATA_DIR}"
    fi
fi

echo "    Data dir: ${DATA_DIR}"

# ---------------------------------------------------------------------------
# Extra flags
# ---------------------------------------------------------------------------
EXTRA_FLAGS=""
if [ "${SMOKE_TEST}" -eq 1 ]; then
    echo "==> Smoke-test mode: 20 optimizer steps"
    EXTRA_FLAGS="--max-steps 20 --val-interval 10"
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
RESULTS_DIR="${ROOT}/results/section5"
CHECKPOINT_DIR="${ROOT}/assets/dpo_hh"
mkdir -p "${RESULTS_DIR}"

echo "==> Starting DPO training ..."
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

uv run python "${ROOT}/cs336_alignment/section5_dpo/train_dpo.py" \
    --model              "${SFT_MODEL}" \
    --data-dir           "${DATA_DIR}" \
    --output             "${RESULTS_DIR}" \
    --checkpoint-dir     "${CHECKPOINT_DIR}" \
    --n-epochs           1 \
    --beta               0.1 \
    --lr                 1e-6 \
    --gradient-accumulation-steps 64 \
    --n-val              200 \
    --val-interval       50 \
    --policy-device      cuda:0 \
    --ref-device         cuda:1 \
    --run-name           dpo_hh \
    --wandb-project      cs336-part6-dpo \
    ${NO_WANDB} \
    ${BALANCE_SOURCES} \
    ${EXTRA_FLAGS}

echo "==> DPO training done."
echo "    Best checkpoint: ${CHECKPOINT_DIR}/best"

# ---------------------------------------------------------------------------
# §5.4 — Evaluate DPO model (delegated to standalone eval script)
# ---------------------------------------------------------------------------
EVAL_FLAGS=""
[ "${SKIP_JUDGES}" -eq 1 ] && EVAL_FLAGS="${EVAL_FLAGS} --skip-judges"
[ "${SMOKE_TEST}"  -eq 1 ] && EVAL_FLAGS="${EVAL_FLAGS} --smoke-test"

echo ""
bash "${ROOT}/cs336_alignment/section5_dpo/part_6_5_eval.sh" ${EVAL_FLAGS}

echo ""
echo "==> §5 complete. Results in ${RESULTS_DIR}"