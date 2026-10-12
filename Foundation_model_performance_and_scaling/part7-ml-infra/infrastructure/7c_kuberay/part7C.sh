#!/usr/bin/env bash
# Usage: part7C.sh render|phase|job [CLI options] | build --cloud BASE_DIGEST LOCAL_TAG
# render/job --check are CPU-only; job --cloud submits to an existing cluster.
set -euo pipefail
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:---help}" in
  render)
    shift
    exec python "$part7_root/infrastructure/7c_kuberay/render.py" "$@"
    ;;
  job)
    shift
    exec python "$part7_root/infrastructure/7c_kuberay/job_control.py" "$@"
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
    echo 'part7C.sh job --check --manifest PATH | --cloud --manifest PATH --context CONTEXT --output PATH [--cleanup-on-success]'
    echo 'Operator installation and cluster provisioning remain external prerequisites.'
    ;;
  *) echo 'Unknown action; use --help.' >&2; exit 2 ;;
esac
