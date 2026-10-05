#!/usr/bin/env bash
# Sourced by cloud setup/runner. Installs toolkit only; never installs a driver.
# Official installer options: NVIDIA CUDA 12.8 Linux installation guide.
select_7a_cuda() {
  local project_root="$1" mode="$2" toolkit_path candidate installer original_link
  toolkit_path=""
  for candidate in "${CUDA_HOME:-/nonexistent}" "$project_root/.cuda-7a/12.8" /usr/local/cuda-12.8 /usr/local/cuda; do
    if [[ "$candidate" == "$project_root/.cuda-7a/12.8" ]] && [[ ! -f "$candidate/.setup_complete" ]]; then
      continue
    fi
    if [[ -x "$candidate/bin/nvcc" ]] && [[ "$("$candidate/bin/nvcc" --version)" == *"release 12.8"* ]]; then
      toolkit_path="$candidate"
      break
    fi
  done
  if [[ -z "$toolkit_path" ]]; then
    if [[ "$mode" != install ]]; then
      echo 'CUDA 12.8 toolkit is missing; rerun setup on cloud.' >&2
      return 1
    fi
    [[ "$(uname -m)" == x86_64 ]] || { echo 'Automatic toolkit installation requires Linux x86_64.' >&2; return 1; }
    command -v curl >/dev/null || return 1
    command -v g++ >/dev/null || { echo 'The image must provide g++ for CUDA extension builds.' >&2; return 1; }
    toolkit_path="$project_root/.cuda-7a/12.8"
    mkdir -p "$project_root/.cuda-7a" "$project_root/results/7a/environment"
    installer="$project_root/.cuda-7a/cuda_12.8.0_570.86.10_linux.run"
    echo "Installing CUDA 12.8 toolkit alongside the image toolkit at $toolkit_path"
    curl --fail --location --retry 3 --continue-at - \
      https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run \
      --output "$installer" || return 1
    sha256sum "$installer" > "$project_root/results/7a/environment/cuda_installer.sha256"
    original_link="$(readlink /usr/local/cuda || true)"
    # Keep the image's default CUDA link even if NVIDIA's installer changes it.
    # The subshell trap also runs when installation fails or is interrupted.
    (
      restore_cuda_link() {
        if [[ -n "$original_link" ]] && [[ "$(readlink /usr/local/cuda || true)" != "$original_link" ]]; then
          ln -sfnT "$original_link" /usr/local/cuda
        fi
      }
      trap restore_cuda_link EXIT
      sh "$installer" --silent --toolkit --toolkitpath="$toolkit_path" \
        --defaultroot="$toolkit_path" --no-man-page
    ) || return 1
    [[ -x "$toolkit_path/bin/nvcc" ]] || return 1
    [[ "$("$toolkit_path/bin/nvcc" --version)" == *"release 12.8"* ]] || return 1
    touch "$toolkit_path/.setup_complete"
  fi
  export CUDA_HOME="$toolkit_path"
  export PATH="$CUDA_HOME/bin:$PATH"
  # Torch wheels supply runtime libraries. Do not globally redirect CUDA 13 libraries.
  [[ "$(nvcc --version)" == *"release 12.8"* ]] || return 1
}
