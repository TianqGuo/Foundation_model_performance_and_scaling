#!/usr/bin/env bash
# Usage: bash infrastructure/7d_serve/build_image.sh --cloud [LOCAL_TAG]
# Cloud-only candidate build + model-free import gate. Never pushes an image.
set -euo pipefail
if [[ "${1:-}" != --cloud ]]; then
  echo 'Explicit --cloud required; run on the cloud VM.' >&2; exit 2
fi
shift
case "$(uname -r)" in
  *[Mm]icrosoft*|*[Ww][Ss][Ll]*) echo 'Run on the cloud GPU VM, not WSL.' >&2; exit 2 ;;
esac
if [[ $# -gt 1 || -e /.dockerenv || "$(ps -p 1 -o comm= | xargs)" != systemd ]]; then
  echo 'Expected a full cloud VM and at most one local image tag.' >&2; exit 2
fi
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="${1:-part7-serving:7d-candidate}"
if [[ "$tag" == -* || "$tag" == *[[:space:]]* ]]; then
  echo 'Invalid local image tag.' >&2; exit 2
fi
output="${PART7_BUILD_OUTPUT:-$part7_root/results/7d/image_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "$(dirname -- "$output")"
mkdir "$output"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
if [[ -n "${PART7_DOCKER_CONFIG:-}" ]]; then docker_cmd+=(--config "$PART7_DOCKER_CONFIG"); fi
"${docker_cmd[@]}" build --progress=plain \
  -f "$part7_root/infrastructure/7d_serve/Dockerfile" \
  --tag "$tag" "$part7_root" 2>&1 | tee "$output/build.log"
# No model download or engine construction; failure prevents a passing receipt.
"${docker_cmd[@]}" run --rm --network none --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint python "$tag" /opt/part7/infrastructure/7d_serve/runtime_check.py --cloud 2>&1 \
  | tee "$output/runtime_check.log"
"${docker_cmd[@]}" image inspect "$tag" --format '{{.Id}}' > "$output/local_image_id.txt"
echo "Candidate import gate passed; cloud serving/E2E remains pending. Evidence: $output"
