#!/usr/bin/env bash
# CLOUD BUILD ONLY: bash infrastructure/7c_kuberay/build_image.sh --cloud BASE_DIGEST LOCAL_TAG
# Builds source/dependencies once. Does not load models, push images or deploy jobs.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Cloud: build_image.sh --cloud registry/base@sha256:DIGEST local-image:tag'
  echo 'Base must provide uv, git and CUDA 13 development toolkit. Publish/pin final registry digest separately.'
  exit 0
fi
[[ "${1:-}" == --cloud && "$#" == 3 ]] || { echo 'Expected --cloud BASE_DIGEST LOCAL_TAG' >&2; exit 2; }
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
base_image="$2"
local_tag="$3"
python - "$part7_root" "$base_image" <<'CHECK'
from importlib import import_module
import sys
sys.path.insert(0, sys.argv[1])
import_module('infrastructure.7c_kuberay.render').pinned_image(sys.argv[2])
CHECK
docker build --pull --platform linux/amd64 --build-arg "BASE_IMAGE=$base_image" \
  -f "$part7_root/infrastructure/7c_kuberay/Dockerfile" -t "$local_tag" "$part7_root"
echo 'Local image configuration ID (not a registry manifest digest):'
docker image inspect --format '{{.Id}}' "$local_tag"
echo 'No image push performed. GPU/import/preflight validation remains required.'
