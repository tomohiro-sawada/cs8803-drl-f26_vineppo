"""One persistent vLLM replica on the single GPU Slurm assigned to this step.

Run with the inference environment. Writes --worker-dir/worker-<rank>.json (URL,
GPU UUID, versions) before serving, so the learner can verify that every process
owns a distinct device. exec replaces this process with `vllm serve`, so the
Slurm step owns the server and launch.sh stops it with the learner. The port is
derived from the job ID and rank: a collision makes the server exit, which
fails the job loudly.
"""

import argparse
import json
import os
import socket
import sys
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core.vllm_inference import BASE_MODEL, gpu_identity, slurm_attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--max-lora-rank", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()
    assert os.environ["VLLM_API_KEY"], "Workers require an API key"
    rank = args.rank
    host = socket.gethostname()
    port = 20000 + (int(os.environ["SLURM_JOB_ID"]) * 16 + rank) % 20000
    identity = {
        **slurm_attempt(),
        "rank": rank,
        "url": f"http://{host}:{port}",
        "host": host,
        "pid": os.getpid(),
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": gpu_identity(),
        "software": {
            name: version(name) for name in ("vllm", "torch", "transformers", "peft")
        },
        "engine": vars(args) | {"dtype": "bfloat16", "generation_config": "vllm"},
    }
    path = args.worker_dir / f"worker-{rank}.json"
    assert not path.exists(), f"Duplicate worker rank: {path}"
    staging = path.with_suffix(".staging")
    staging.write_text(json.dumps(identity, indent=2, default=str) + "\n")
    staging.rename(path)
    print(json.dumps(identity, default=str), flush=True)
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    os.execv(
        str(Path(sys.executable).with_name("vllm")),
        [
            "vllm",
            "serve",
            str(args.model_path.resolve(strict=True)),
            "--served-model-name",
            BASE_MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            str(args.max_sequence_length),
            # Ignore the checkpoint's sampling defaults; requests are explicit.
            "--generation-config",
            "vllm",
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--enable-lora",
            "--max-lora-rank",
            str(args.max_lora_rank),
            # Old and new actor coexist only while a new version is acknowledged.
            "--max-loras",
            "2",
            "--max-cpu-loras",
            "4",
            "--seed",
            "0",
        ],
    )


if __name__ == "__main__":
    main()
