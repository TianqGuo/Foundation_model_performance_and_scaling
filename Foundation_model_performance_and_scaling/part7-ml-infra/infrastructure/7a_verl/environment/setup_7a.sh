#!/usr/bin/env bash
# Cloud only: bash infrastructure/7a_verl/environment/setup_7a.sh --cloud
# Requires uv, git, the image CUDA 13 toolkit, and two cloud GPUs. Uses GPU wheels.
set -euo pipefail
if [[ "${1:-}" != "--cloud" ]]; then
  echo 'Run only on the approved cloud instance with --cloud.' >&2
  exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -e "$root/.venv-7a" ]]; then
  echo 'Use a fresh .venv-7a directory; existing environments are preserved.' >&2
  exit 2
fi
command -v uv >/dev/null
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 2 ]]
mkdir -p "$root/results/7a/environment"
source "$root/infrastructure/7a_verl/environment/cuda_7a.sh"
select_7a_cuda
cuda_toolkit_version="$(nvcc --version)"
printf '%s\n' "$cuda_toolkit_version" | tee "$root/results/7a/environment/cuda_version.txt"
if [[ "$cuda_toolkit_version" != *"release 13."* ]]; then
  echo 'This setup requires the image CUDA 13 development toolkit.' >&2
  exit 2
fi
uv python install 3.12.14
uv venv --python 3.12.14 "$root/.venv-7a"
python_path="$root/.venv-7a/bin/python"
mkdir -p "$root/.venv-7a/src"
git clone https://github.com/verl-project/verl.git "$root/.venv-7a/src/verl"
git -C "$root/.venv-7a/src/verl" checkout --detach bec9ef74768dd201881cd4e54cd0385e87caae27
# Official GPU wheels: no separate toolkit download or CUDA compilation.
uv pip sync --python "$python_path" "$root/infrastructure/7a_verl/environment/requirements.lock" \
  --index-url https://pypi.org/simple --only-binary torch,vllm,flash-attn,torchvision,torchaudio
uv pip check --python "$python_path"
uv pip freeze --python "$python_path" > "$root/results/7a/environment/installed.txt"
"$python_path" -c 'import torch; assert torch.version.cuda == "13.0"; assert torch._C._GLIBCXX_USE_CXX11_ABI; import vllm, verl, flash_attn, ray; assert torch.cuda.device_count() == 2; print(torch.__version__, torch.version.cuda, vllm.__version__, ray.__version__)' \
  > "$root/results/7a/environment/import_check.txt"
cd "$root"
sha256sum infrastructure/7a_verl/environment/requirements.txt infrastructure/7a_verl/environment/requirements-cu130.in \
  infrastructure/7a_verl/environment/requirements.lock infrastructure/7a_verl/environment/setup_7a.sh \
  infrastructure/7a_verl/environment/cuda_7a.sh \
  > .venv-7a/setup_inputs.sha256
