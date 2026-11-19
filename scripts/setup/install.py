"""Install the default SDPA environment or an isolated optional inference stack."""

import argparse
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Install vLLM and FlashAttention instead of the default SDPA group",
    )
    parser.add_argument(
        "--build-flash-attn",
        action="store_true",
        help="Force FlashAttention source compilation (requires matching CUDA toolkit)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.build_flash_attn and not args.inference:
        parser.error("--build-flash-attn requires --inference")

    env = os.environ.copy()
    command = ["uv", "sync", "--locked"]
    if args.inference:
        libc, version = platform.libc_ver()
        if (
            platform.system() != "Linux"
            or platform.machine() != "x86_64"
            or libc != "glibc"
        ):
            parser.error("The inference stack requires Linux x86_64 with glibc")
        env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / f".venv-inference-glibc{version}")
        command += ["--python", "3.13", "--no-group", "sdpa", "--group", "inference"]
        env.setdefault("MAX_JOBS", "4")
        if args.build_flash_attn:
            env["FLASH_ATTENTION_FORCE_BUILD"] = "TRUE"
            command += ["--no-cache", "--reinstall-package", "flash-attn"]
    else:
        env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv")
    if args.dry_run:
        command.append("--dry-run")
    print(f"Environment: {env['UV_PROJECT_ENVIRONMENT']}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.inference and not args.dry_run:
        python = str(Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin/python")
        subprocess.run(
            [
                python,
                "-c",
                "import torch, vllm, vllm._C_stable_libtorch, flash_attn, flash_attn_2_cuda; from vllm import LLM, SamplingParams; print(torch.__version__, vllm.__version__, flash_attn.__version__)",
            ],
            cwd=ROOT,
            check=True,
        )
        print(
            "vLLM API and FlashAttention native imports passed; GPU execution still needs validation."
        )


if __name__ == "__main__":
    main()
