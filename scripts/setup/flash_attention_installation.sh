#!/bin/bash
#SBATCH --job-name=flash-attn-build
#SBATCH --partition=ice-gpu
#SBATCH --account=math
#SBATCH --qos=coc-ice
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-h100|gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/flash-attn-%j.log

# Submit from the repository root with sbatch, not bash. No tmux is needed on PACE.
# Do not run other installers or inference jobs using this environment concurrently.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
repo_root=${SLURM_SUBMIT_DIR:?Submit from the repository root}
cd "$repo_root"
test -f scripts/setup/install.py
source "$repo_root/bash.sh"
printf 'Started: %s; commit: %s; job: %s\n' "$(timestamp)" "$(git rev-parse --short HEAD)" "$SLURM_JOB_ID"

module load cuda/13.0.1
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4 NVCC_THREADS=1 PYTHONUNBUFFERED=1
# Ampere/Ada and Hopper; this is not a Blackwell build.
export FLASH_ATTN_CUDA_ARCHS='80;90'
nvcc --version
nvidia-smi

python=".venv-inference-glibc$(getconf GNU_LIBC_VERSION | cut -d ' ' -f 2)/bin/python"
test -x "$python"
# Prevent two copies of this script from modifying the same environment.
exec 9>"$(dirname "$python")/../.flash-attn-build.lock"
flock --nonblock 9
"$python" - <<'PY'
import subprocess
import torch
import ninja, packaging, psutil, setuptools, wheel

assert torch.version.cuda == "13.0", torch.version.cuda
assert "release 13.0," in subprocess.check_output(["nvcc", "--version"], text=True)
assert torch.cuda.is_available(), "A working GPU/driver is required for validation"
assert torch.cuda.get_device_capability()[0] in (8, 9)
print("Torch:", torch.__version__, "GPU:", torch.cuda.get_device_name())
PY

.venv/bin/python scripts/setup/install.py --inference --build-flash-attn

"$python" - <<'PY'
import torch
import torch.nn.functional as F
import flash_attn
import flash_attn_2_cuda
import vllm
import vllm._C_stable_libtorch
from vllm import LLM, SamplingParams
from torch.nn.attention import SDPBackend, sdpa_kernel

print("Versions:", torch.__version__, flash_attn.__version__, vllm.__version__)
torch.manual_seed(0)
for dtype in (torch.float16, torch.bfloat16):
    q, k, v = [torch.randn(2, 128, 4, 64, device="cuda", dtype=dtype,
                          requires_grad=True) for _ in range(3)]
    actual = flash_attn.flash_attn_func(q, k, v, causal=True)
    refs = [x.detach().float().requires_grad_() for x in (q, k, v)]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            *(x.transpose(1, 2) for x in refs), is_causal=True
        ).transpose(1, 2)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.float())
    for x, ref in zip((q, k, v), refs):
        torch.testing.assert_close(x.grad.float(), ref.grad, atol=tolerance, rtol=tolerance)
    torch.cuda.synchronize()
    print(f"PASS: FlashAttention {dtype} forward/backward against FP32 SDPA")
print("PASS: vLLM native/API imports (generation not tested)")
PY
printf 'Completed: %s; commit: %s\n' "$(timestamp)" "$(git rev-parse --short HEAD)"
