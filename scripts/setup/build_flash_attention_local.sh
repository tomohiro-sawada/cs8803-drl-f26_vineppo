#!/bin/bash
# No Slurm/GPU required. Use on login nodes only if site policy permits long builds.
# Do not run other installers or inference jobs in this environment during the build.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
source "$repo_root/bash.sh"
mkdir -p artifacts
log="artifacts/flash-attn-local-$(timestamp).log"
exec > >(tee "$log") 2>&1
printf 'Started: %s; commit: %s; log: %s\n' "$(timestamp)" "$(git rev-parse --short HEAD)" "$log"

module load cuda/13.0.1
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=1 NVCC_THREADS=1 PYTHONUNBUFFERED=1
export FLASH_ATTN_CUDA_ARCHS='80;90'
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-inference-glibc$(getconf GNU_LIBC_VERSION | cut -d ' ' -f 2)"
python="$UV_PROJECT_ENVIRONMENT/bin/python"
test -x "$python"
exec 9>"$UV_PROJECT_ENVIRONMENT/.flash-attn-build.lock"
flock --nonblock 9
nvcc --version
"$python" - <<'PY'
import subprocess
import torch
import ninja, packaging, psutil, setuptools, wheel

assert torch.version.cuda == "13.0", torch.version.cuda
assert "release 13.0," in subprocess.check_output(["nvcc", "--version"], text=True)
print("Building against Torch", torch.__version__)
PY

# Same locked inference stack as install.py, without its GPU-driver-dependent vLLM check.
uv sync --locked --python 3.13 --no-group sdpa --group inference \
    --no-cache --reinstall-package flash-attn
"$python" - <<'PY'
import torch
import flash_attn
import flash_attn_2_cuda
from importlib.metadata import version

print("PASS: FlashAttention native import", flash_attn.__version__)
print("Installed vLLM:", version("vllm"), "(not imported or GPU-validated)")
PY
printf 'Completed: %s; commit: %s\n' "$(timestamp)" "$(git rev-parse --short HEAD)"
printf 'GPU execution and vLLM generation still require validation on a GPU node.\n'
