#!/usr/bin/env bash
# Usage: part7C.sh build --cloud BASE_REFERENCE LOCAL_TAG
# Cloud only: resolve base tag to digest, retain build evidence, build without model loading.
# No registry publication, job submission or changes to Parts 1–6.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Cloud: build_image.sh --cloud registry/base:tag local-image:tag'
  echo 'Digest references also accepted. Base requires Ubuntu apt and CUDA 13 development toolkit; uv/git/tar are added.'
  exit 0
fi
[[ "${1:-}" == --cloud && "$#" == 3 ]] || { echo 'Expected --cloud BASE_REFERENCE LOCAL_TAG' >&2; exit 2; }
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
base_reference="$2"
local_tag="$3"
# Reject option-like references before handing any user value to Docker.
[[ "$base_reference" != -* && "$local_tag" != -* && -n "$base_reference" && -n "$local_tag" ]] || exit 2
docker_command=(docker)
if ! docker info >/dev/null 2>&1; then
  sudo -n docker info >/dev/null
  docker_command=(sudo -n docker)
fi
mkdir -p "$part7_root/results/7c"
if [[ -n "${PART7_BUILD_OUTPUT:-}" ]]; then
  build_output="$PART7_BUILD_OUTPUT"
  mkdir "$build_output"
else
  build_output="$(mktemp -d "$part7_root/results/7c/image_build_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
fi
if [[ -n "${DOCKER_CONFIG:-}" ]]; then
  docker_command+=(--config "$DOCKER_CONFIG")
fi
uv_reference="ghcr.io/astral-sh/uv:0.12.19"
"${docker_command[@]}" pull --platform linux/amd64 "$uv_reference" 2>&1 | tee "$build_output/uv_pull.log"
uv_image="$("${docker_command[@]}" image inspect --format '{{index .RepoDigests 0}}' "$uv_reference")"
# Tag resolution happens once, then FROM uses the immutable registry reference.
"${docker_command[@]}" pull --platform linux/amd64 "$base_reference" 2>&1 | tee "$build_output/base_pull.log"
base_image="$("${docker_command[@]}" image inspect --format '{{index .RepoDigests 0}}' "$base_reference")"
python3 - "$part7_root" "$base_image" "$uv_image" <<'CHECK'
from importlib import import_module
import sys
sys.path.insert(0, sys.argv[1])
for reference in sys.argv[2:]:
    import_module('infrastructure.7c_kuberay.render').pinned_image(reference)
CHECK
"${docker_command[@]}" build --pull --platform linux/amd64 --build-arg "BASE_IMAGE=$base_image" --build-arg "UV_IMAGE=$uv_image" \
  -f "$part7_root/infrastructure/7c_kuberay/Dockerfile" -t "$local_tag" "$part7_root" \
  2>&1 | tee "$build_output/build.log"
image_id="$("${docker_command[@]}" image inspect --format '{{.Id}}' "$local_tag")"
python3 - "$build_output" "$base_reference" "$base_image" "$local_tag" "$image_id" "$part7_root" "$uv_image" <<'RECEIPT'
import hashlib
import json
from pathlib import Path
import sys
output, reference, digest, tag, image_id, root, uv_image = sys.argv[1:]
root = Path(root)
identities = {}
for relative in ('infrastructure/7c_kuberay/Dockerfile', 'infrastructure/7a_verl/environment/requirements.lock'):
    identities[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
receipt = dict(base_reference=reference, base_digest=digest, local_tag=tag,
               local_configuration_id=image_id, uv_digest=uv_image, source_hashes=identities,
               published=False, gpu_verified=False)
(Path(output) / 'image_build.json').write_text(json.dumps(receipt, indent=2) + '\n')
RECEIPT
printf 'Build evidence: %s\n' "$build_output"
echo 'Build complete; no image push performed. Registry digest and GPU preflight remain pending.'
