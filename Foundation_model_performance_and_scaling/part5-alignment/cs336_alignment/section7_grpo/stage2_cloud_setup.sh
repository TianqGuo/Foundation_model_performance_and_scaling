#!/usr/bin/env bash
# CLOUD ONLY: install the isolated, locked Stage 2 environment (Python 3.12).
# USAGE: bash cs336_alignment/section7_grpo/stage2_cloud_setup.sh
# Requires a CUDA 12.9-capable driver and two visible GPUs. Leaves .venv intact.
# No external FlashAttention wheel/site-packages patch is used; trainer uses SDPA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
command -v uv >/dev/null
command -v nvidia-smi >/dev/null
python3 - <<'PY'
import re
import subprocess
info = subprocess.check_output(['nvidia-smi'], text=True)
print(info)
version = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', info)
if not version or tuple(map(int, version.groups())) < (12, 9):
    raise SystemExit('Stage 2 requires a driver advertising CUDA >= 12.9. Choose a newer cloud image/host.')
PY
UV_PROJECT_ENVIRONMENT="${ROOT}/.venv-server" uv sync \
    --project "${ROOT}/cs336_alignment/section7_grpo/server_environment" \
    --frozen --python 3.12
"${ROOT}/.venv-server/bin/python" - <<'PY'
from importlib.metadata import version
import torch
for name in ('torch', 'vllm', 'transformers', 'wandb'):
    print(name, version(name))
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit('Stage 2 requires two visible cloud GPUs.')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB')
PY
printf '%s\n' 'Setup complete. Run: bash cs336_alignment/section7_grpo/stage2_cloud_smoke.sh'
