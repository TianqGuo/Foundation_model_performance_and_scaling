#!/usr/bin/env bash
# USAGE:   bash get_assets.sh [--annotator]
#
# WHAT IT DOES:
#   Downloads the models needed for Part 6 into assets/.
#
#   By default downloads only the 8B base model (needed for all evaluations).
#   Pass --annotator to also download the 70B Instruct model (needed for
#   AlpacaEval winrate and SimpleSafetyTests safety annotation).
#
# MODELS:
#   assets/Llama-3.1-8B              meta-llama/Llama-3.1-8B
#   assets/Llama-3.3-70B-Instruct    meta-llama/Llama-3.3-70B-Instruct  (--annotator)
#
# NOTES:
#   Both models are gated on HuggingFace — you must:
#     1. Accept the Llama licence at https://huggingface.co/meta-llama/Llama-3.1-8B
#     2. Accept the Llama licence at https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
#     3. Run:  uv run huggingface-cli login
#   before running this script.
#
#   On the cluster the models are pre-cached at /data/a5-alignment/models/ —
#   this script is only needed on local machines without cluster access.

set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"

DOWNLOAD_ANNOTATOR=0
for arg in "$@"; do
    case $arg in
        --annotator) DOWNLOAD_ANNOTATOR=1 ;;
    esac
done

mkdir -p "${ROOT}/assets"

# ── Llama 3.1 8B base (required) ─────────────────────────────────────────────
CLUSTER_MODEL="/data/a5-alignment/models/Llama-3.1-8B"
LOCAL_MODEL="${ROOT}/assets/Llama-3.1-8B"

if [ -d "${CLUSTER_MODEL}" ]; then
    echo "Llama 3.1 8B found on cluster: ${CLUSTER_MODEL}"
elif [ -d "${LOCAL_MODEL}" ]; then
    echo "Llama 3.1 8B already downloaded: ${LOCAL_MODEL}"
else
    echo "Downloading Llama 3.1 8B from HuggingFace -> ${LOCAL_MODEL}"
    echo "(This requires a HuggingFace login and accepted Llama licence)"
    uv run huggingface-cli download meta-llama/Llama-3.1-8B \
        --local-dir "${LOCAL_MODEL}"
    echo "Done: ${LOCAL_MODEL}"
fi

# ── Llama 3.3 70B Instruct (annotator, optional) ─────────────────────────────
if [ "${DOWNLOAD_ANNOTATOR}" -eq 1 ]; then
    CLUSTER_ANN="/data/a5-alignment/models/Llama-3.3-70B-Instruct"
    LOCAL_ANN="${ROOT}/assets/Llama-3.3-70B-Instruct"

    if [ -d "${CLUSTER_ANN}" ]; then
        echo "Llama 3.3 70B Instruct found on cluster: ${CLUSTER_ANN}"
    elif [ -d "${LOCAL_ANN}" ]; then
        echo "Llama 3.3 70B Instruct already downloaded: ${LOCAL_ANN}"
    else
        echo "Downloading Llama 3.3 70B Instruct from HuggingFace -> ${LOCAL_ANN}"
        echo "(This requires a HuggingFace login and accepted Llama licence)"
        uv run huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
            --local-dir "${LOCAL_ANN}"
        echo "Done: ${LOCAL_ANN}"
    fi
fi

echo ""
echo "========================================"
echo "  Assets ready. Run the evaluation with:"
echo "  bash cs336_alignment/section2_zero_shot/part_6_2.sh --smoke-test --plot"
echo "========================================"
