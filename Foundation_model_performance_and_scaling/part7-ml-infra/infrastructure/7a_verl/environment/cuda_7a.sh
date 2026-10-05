#!/usr/bin/env bash
# Sourced by the cloud setup/runner. Selects the image toolkit; downloads nothing.
select_7a_cuda() {
  local toolkit_path nvcc_binary
  toolkit_path="${CUDA_HOME:-}"
  if [[ -z "$toolkit_path" ]]; then
    nvcc_binary="$(command -v nvcc || true)"
    if [[ -z "$nvcc_binary" ]]; then
      echo 'The image must provide a CUDA 13 development toolkit (nvcc).' >&2
      return 1
    fi
    toolkit_path="$(dirname -- "$(dirname -- "$(readlink -f -- "$nvcc_binary")")")"
  fi
  if [[ ! -x "$toolkit_path/bin/nvcc" ]] || [[ "$("$toolkit_path/bin/nvcc" --version)" != *"release 13."* ]]; then
    echo 'This pinned cu130 stack requires the image CUDA 13 toolkit; check CUDA_HOME.' >&2
    return 1
  fi
  export CUDA_HOME="$toolkit_path"
  export PATH="$CUDA_HOME/bin:$PATH"
  # Runtime libraries come from the pinned wheels; keep system links unchanged.
}
