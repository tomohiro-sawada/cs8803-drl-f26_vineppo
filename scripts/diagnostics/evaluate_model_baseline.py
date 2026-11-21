"""Evaluate the pinned unadapted model on the test split via the PACE launcher."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from transformers import AutoTokenizer

from core.prompt_protocol import prompt_protocol
from experiments.training import evaluate
from experiments.training.resume import atomic_record
from experiments.training.run import git, parse_args, write_json


def main():
    args = parse_args()
    assert args.evaluation == "disabled" and args.resume is None
    profile = prompt_protocol(args.model_protocol, args.task)
    assert args.max_sequence_length >= profile.context_size
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    args.output.mkdir(parents=True)
    write_json(
        args.output / "metadata.json",
        {
            "cli": vars(args),
            "model": {
                "path": str(args.model_path),
                "config_sha256": evaluate.file_sha256(args.model_path / "config.json"),
            },
            "dataset": {"path": str(args.dataset_path)},
            "prompt": profile.template,
            "tokenizer_sha256": hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode()
            ).hexdigest(),
            "git": {
                "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")),
            },
        },
    )
    options = [
        "--split",
        "test",
        "--baseline",
        "--training-output",
        str(args.output),
        "--output",
        str(args.output / "test"),
        "--rounds",
        "16",
        "--seed",
        str(args.eval_seed),
        "--batch-size",
        str(args.eval_batch_size),
    ]
    for key in (
        "backend",
        "num_workers",
        "worker_dir",
        "expected_gpu",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_name",
    ):
        value = getattr(args, key)
        if value is not None:
            options.extend((f"--{key.replace('_', '-')}", str(value)))
    result = evaluate.main(options)
    assert result["complete"] and result["rounds_completed"] == 16
    atomic_record(
        args.output / "COMPLETE", {"result": str(args.output / "test/result.json")}
    )


if __name__ == "__main__":
    main()
