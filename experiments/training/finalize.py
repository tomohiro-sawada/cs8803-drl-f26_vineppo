"""Complete a resumable run's frozen test without recreating its trainer."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from experiments.training import evaluate
from experiments.training.run import parse_args, trainer_config


def frozen_plan(args):
    path = args.resume_selection
    owner = path.parent
    assert path.name == "selection-frozen.json" and owner.parent == args.output
    assert owner.name.startswith("attempt-"), (
        "Selection must belong to a training attempt"
    )
    selected = json.loads(path.read_text())
    assert evaluate.checkpoint_source(selected) == owner
    checkpoint = evaluate.selected_checkpoint(
        selected, args.task, owner, verify_digest=False
    )
    config = asdict(trainer_config(args))
    del config["checkpoint_dir"]
    plan = selected["plan"]
    assert plan["trainer_config"] == config and plan["evaluation"] == args.evaluation, (
        "Frozen training plan differs from launcher options"
    )
    cli = json.loads((owner / "metadata.json").read_text())["cli"]
    for key in (
        "model_path",
        "dataset_path",
        "finetuning",
        "dtype",
        "lora_rank",
        "lora_alpha",
        "learning_rate",
        "critic_learning_rate",
    ):
        value = getattr(args, key)
        assert cli[key] == (str(value) if isinstance(value, Path) else value), (
            f"Frozen {key} differs from launcher options"
        )
    protocol = evaluate.protocol_for(args.backend) | {
        "seed": args.eval_seed,
        "batch_size": args.eval_batch_size,
    }
    assert selected["validation_protocol"] == protocol | {
        "rounds": args.validation_rounds
    }
    assert selected["test_protocol"] == protocol | {"rounds": 16}
    return selected, checkpoint


def completed_record(args, selected, result_path):
    result = json.loads(result_path.read_text())
    owner = args.resume_selection.parent
    evaluate.validate_results(result, args.task, owner, split="test")
    assert result["protocol"] == selected["test_protocol"], "Test protocol mismatch"
    assert result["dataset"] == selected["plan"]["test_dataset"], (
        "Test dataset mismatch"
    )
    best = selected["best"]
    for key in ("checkpoint", "checkpoint_source"):
        assert result[key] == best[key], f"Test {key} mismatch"
    return {
        "schema_version": 1,
        "selection": str(args.resume_selection),
        "selection_sha256": evaluate.file_sha256(args.resume_selection),
        "checkpoint_sha256": best["checkpoint"]["sha256"],
        "protocol": result["protocol"],
        "result": str(result_path),
        "result_sha256": evaluate.file_sha256(result_path),
    }


def finalize(args, *, prepare):
    selected, checkpoint = frozen_plan(args)
    validation_only = args.evaluation == "validation"
    completed = args.output / "test-complete.json"
    attempts = sorted(args.output.glob("test-attempt-*"))
    original = args.resume_selection.parent / "evaluations/test-selected"
    results = [
        p / "result.json" for p in [original, *attempts] if (p / "result.json").exists()
    ]
    if validation_only or results:
        # With no new evaluator, independently verify the selected bytes here.
        assert (
            evaluate.file_sha256(checkpoint) == selected["best"]["checkpoint"]["sha256"]
        ), "Selected checkpoint hash mismatch"
    if validation_only:
        return "complete"
    assert len(results) <= 1, "Multiple completed tests for the frozen run"
    if results:
        record = completed_record(args, selected, results[0])
        if completed.exists():
            assert json.loads(completed.read_text()) == record, (
                "Conflicting test completion"
            )
        else:
            evaluate.write_json(completed, record, exclusive=True)
        return "complete"
    assert not completed.exists(), "Test completion references a missing result"
    if prepare:
        return "run"
    output = args.output / f"test-attempt-{len(attempts):02d}"
    assert not output.exists(), "Test attempt path already exists"
    selection_bytes = args.resume_selection.read_bytes()
    command = [
        sys.executable,
        "-m",
        "experiments.training.evaluate",
        "--split",
        "test",
        "--training-output",
        str(args.resume_selection.parent),
        "--selection",
        str(args.resume_selection),
        "--output",
        str(output),
        "--backend",
        args.backend,
        "--rounds",
        "16",
        "--seed",
        str(args.eval_seed),
        "--batch-size",
        str(args.eval_batch_size),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-name",
        f"{args.wandb_name or args.resume_selection.parent.name}-{output.name}",
    ]
    for option in (
        "expected_gpu",
        "num_workers",
        "worker_dir",
        "wandb_entity",
        "wandb_group",
    ):
        value = getattr(args, option)
        if value is not None:
            command.extend((f"--{option.replace('_', '-')}", str(value)))
    subprocess.run(command, check=True, timeout=args.eval_timeout_seconds)
    assert args.resume_selection.read_bytes() == selection_bytes, (
        "Frozen selection changed"
    )
    evaluate.write_json(
        completed,
        completed_record(args, selected, output / "result.json"),
        exclusive=True,
    )
    return "complete"


if __name__ == "__main__":
    action, *argv = sys.argv[1:]
    assert action in ("prepare", "run")
    print(finalize(parse_args(argv), prepare=action == "prepare"))
