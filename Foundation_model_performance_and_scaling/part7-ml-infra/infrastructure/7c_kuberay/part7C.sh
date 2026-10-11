#!/usr/bin/env bash
# Usage: part7C.sh render [render.py options] | build --cloud BASE_DIGEST LOCAL_TAG
# CPU render is read-only with respect to Kubernetes; no apply/provisioning here yet.
set -euo pipefail
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:---help}" in
  render)
    shift
    exec python "$part7_root/infrastructure/7c_kuberay/render.py" "$@"
    ;;
  phase)
    shift
    exec python "$part7_root/infrastructure/7c_kuberay/run_7c.py" "$@"
    ;;
  build)
    shift
    exec bash "$part7_root/infrastructure/7c_kuberay/build_image.sh" "$@"
    ;;
  --help)
    echo 'part7C.sh render --image REGISTRY/IMAGE@sha256:DIGEST --node-name NODE [--namespace part7] [--pvc part7-workspace] [--secret part7-huggingface] [--output PATH]'
    echo 'part7C.sh build --cloud BASE_IMAGE@sha256:DIGEST LOCAL_TAG'
    echo 'Render/build/phase preparation: no cluster apply, operator installation or provisioning is automated yet.'
    ;;
  *) echo 'Unknown action; use --help.' >&2; exit 2 ;;
esac
