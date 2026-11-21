"""Synchronous held-out selection: validate before pruning, freeze before test.

The child owns all evaluation RNG/model state. Training remains resident and
idle on its allocated GPU; additional actor inference memory must fit explicitly.
"""

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from experiments.training.evaluate import (
    checkpoint_source,
    file_sha256,
    load_examples,
    protocol_for,
    selected_checkpoint,
    validate_validation,
    write_json,
)
from runtime.checkpoints import file_identity

ROOT = Path(__file__).resolve().parents[2]
TIE_RULE = "highest_validation_accuracy_then_earliest_iteration"


class CheckpointSelection:
    def __init__(self, args, checkpoint_every, start_iteration=0):
        assert args.evaluation in ("validation", "validation-test")
        self.args = args
        self.path = args.output / "selection.json"
        self.frozen_path = args.output / "selection-frozen.json"
        self.directory = args.output / "evaluations"
        self.directory.mkdir()
        protocol = protocol_for(args.backend)
        self.state = {
            "schema_version": 1,
            "frozen": False,
            "task": args.task,
            "training_output": str(args.output),
            "tie_rule": TIE_RULE,
            "validation_protocol": protocol
            | {
                "rounds": args.validation_rounds,
                "seed": args.eval_seed,
                "batch_size": args.eval_batch_size,
            },
            "test_protocol": protocol
            | {
                "rounds": 16,
                "seed": args.eval_seed,
                "batch_size": args.eval_batch_size,
            },
            "candidate_iterations": sorted(
                set(range(checkpoint_every, args.num_iterations + 1, checkpoint_every))
                | {args.num_iterations}
            ),
            "best": None,
            "candidates": [],
            # Unadapted iteration-0 actor: reported separately, never selectable.
            "baseline": None,
        }
        metadata = json.loads((args.output / "metadata.json").read_text())
        config = dict(metadata["trainer_config"])
        config.pop("checkpoint_dir")
        self.state["plan"] = {
            "trainer_config": config,
            "checkpoint_every": checkpoint_every,
            "evaluation": args.evaluation,
            **{
                key: metadata[key]
                for key in ("dataset", "model", "prompt", "tokenizer_sha256")
            },
            **{
                f"{split}_dataset": load_examples(args.dataset_path, args.task, split)[
                    1
                ]
                for split in ("validation", "test")
            },
        }
        self.state["checkpoint_source"] = {
            "path": str(args.output),
            "metadata_sha256": file_sha256(args.output / "metadata.json"),
        }
        self.pending_checkpoint = None
        if args.resume is not None:
            self._resume(start_iteration)
        else:
            assert args.resume_selection is None and start_iteration == 0
        assert not self.path.exists() and not self.frozen_path.exists()
        write_json(self.path, self.state)

    def _resume(self, iteration):
        args = self.args
        source_path = args.resume_selection
        source = json.loads(source_path.read_text())
        assert source["schema_version"] == 1
        assert source["frozen"] is False, "Frozen experiments cannot resume training"
        for key in (
            "task",
            "tie_rule",
            "plan",
            "validation_protocol",
            "test_protocol",
            "candidate_iterations",
        ):
            assert source[key] == self.state[key], f"Resume selection {key} mismatch"
        assert 0 < iteration <= args.num_iterations
        owner = checkpoint_source(source)
        assert str(owner) == source["training_output"]
        assert source_path.parent == owner, "Selection must belong to source output"
        assert args.resume.parent == owner / "checkpoints", (
            "Resume checkpoint owner mismatch"
        )
        candidates = source["candidates"]
        completed = [c["checkpoint"]["iteration"] for c in candidates]
        expected = [i for i in self.state["candidate_iterations"] if i <= iteration]
        assert completed == expected or (
            expected and expected[-1] == iteration and completed == expected[:-1]
        ), "Missing candidates or lookahead beyond resumed checkpoint"
        for candidate in candidates:
            validate_validation(candidate, args.task, owner)
            checkpoint_source(candidate)
            assert candidate["protocol"] == self.state["validation_protocol"]
            assert candidate["dataset"] == self.state["plan"]["validation_dataset"]
        best = (
            min(
                candidates, key=lambda c: (-c["accuracy"], c["checkpoint"]["iteration"])
            )
            if candidates
            else None
        )
        assert source["best"] == best, "Invalid prior best"
        if best is not None:
            assert (
                file_sha256(Path(best["checkpoint"]["path"]))
                == best["checkpoint"]["sha256"]
            )
        is_candidate = args.resume.name == f"iteration-{iteration}.pt"
        if iteration in completed:
            assert is_candidate, "Resume slot cannot replace a selection candidate"
            assert candidates[-1]["checkpoint"] == {
                "path": str(args.resume),
                "iteration": iteration,
                "sha256": file_sha256(args.resume),
            }, "Resume checkpoint differs from completed candidate"
        elif iteration in expected:
            assert is_candidate, "Selection needs the original candidate checkpoint"
            self.pending_checkpoint = args.resume
        # Re-home selection ownership, not actor/evaluation provenance or files.
        wandb = args.output / "wandb.json"
        run = {
            "path": str(args.output),
            "wandb": json.loads(wandb.read_text()) if wandb.exists() else None,
        }
        self.state["candidates"] = [c | {"training_run": run} for c in candidates]
        self.state["best"] = next(
            (
                c
                for c in self.state["candidates"]
                if best is not None and c["checkpoint"] == best["checkpoint"]
            ),
            None,
        )
        self.state["baseline"] = source["baseline"]
        self.state["resume"] = {
            "selection": str(source_path),
            "sha256": file_sha256(source_path),
            "training_output": str(owner),
            "checkpoint": str(args.resume),
            "iteration": iteration,
        }
        self.resume_source = source["checkpoint_source"]

    def recover(self, iteration):
        """Complete interrupted validation before the first resumed update."""
        if self.pending_checkpoint is not None:
            self.validate(self.pending_checkpoint, iteration)
            self.pending_checkpoint = None

    @property
    def best_path(self):
        best = self.state["best"]
        return None if best is None else Path(best["checkpoint"]["path"])

    def _evaluate(
        self, split, *, checkpoint=None, iteration=None, checkpoint_sha256=None
    ):
        name = f"validation-{iteration}" if split == "validation" else "test-selected"
        output = self.directory / name
        args = self.args
        command = [
            sys.executable,
            "-m",
            "experiments.training.evaluate",
            "--split",
            split,
            "--training-output",
            str(args.output),
            "--output",
            str(output),
            "--rounds",
            str(args.validation_rounds if split == "validation" else 16),
            "--seed",
            str(args.eval_seed),
            "--batch-size",
            str(args.eval_batch_size),
            "--wandb-mode",
            args.wandb_mode,
            "--wandb-project",
            args.wandb_project,
            "--wandb-name",
            f"{args.output.name}-{name}",
        ]
        if args.backend == "vllm":
            command += [
                "--backend",
                "vllm",
                "--num-workers",
                str(args.num_workers),
                "--worker-dir",
                str(args.worker_dir),
            ]
        if split == "validation" and checkpoint is None:
            command += ["--baseline"]
        elif split == "validation":
            command += [
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                checkpoint_sha256,
            ]
            if checkpoint == self.pending_checkpoint:
                command += [
                    "--checkpoint-source",
                    self.resume_source["path"],
                    "--checkpoint-source-sha256",
                    self.resume_source["metadata_sha256"],
                ]
        else:
            command += ["--selection", str(self.frozen_path)]
        for flag, value in (
            ("--expected-gpu", args.expected_gpu),
            ("--wandb-entity", args.wandb_entity),
            ("--wandb-group", args.wandb_group),
        ):
            if value is not None:
                command += [flag, value]
        with (self.directory / f"{name}-command.json").open("x") as stream:
            json.dump(command, stream, indent=2)
        # No shell, sampling in the parent, CPU fallback or automatic retries.
        with (self.directory / f"{name}.log").open("x") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                timeout=args.eval_timeout_seconds,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        result = json.loads((output / "result.json").read_text())
        assert result["complete"] is True and result["split"] == split
        return result

    def baseline(self):
        """Validate the initial actor once, with the candidates' exact protocol."""
        assert self.state["baseline"] is None and not self.state["candidates"]
        result = self._evaluate("validation", iteration=0)
        validate_validation(result, self.args.task, self.args.output)
        assert result["protocol"] == self.state["validation_protocol"]
        assert result["dataset"] == self.state["plan"]["validation_dataset"]
        self.state["baseline"] = result
        write_json(self.path, self.state)
        return result

    def validate(self, checkpoint, iteration):
        assert not self.state["frozen"] and not self.frozen_path.exists()
        candidates = self.state["candidates"]
        assert iteration == self.state["candidate_iterations"][len(candidates)], (
            "Missing, duplicate or out-of-order validation candidate"
        )
        checkpoint = checkpoint.resolve(strict=True)
        identity = file_identity(checkpoint)
        digest = file_sha256(checkpoint)
        result = self._evaluate(
            "validation",
            checkpoint=checkpoint,
            iteration=iteration,
            checkpoint_sha256=digest,
        )
        validate_validation(result, self.args.task, self.args.output)
        assert file_identity(checkpoint) == identity, "Validation checkpoint changed"
        assert result["checkpoint"] == {
            "path": str(checkpoint),
            "sha256": digest,
            "iteration": iteration,
        }, "Validation checkpoint identity mismatch"
        assert result["protocol"] == self.state["validation_protocol"]
        checkpoint_source(result)
        assert result["checkpoint_source"] == (
            self.resume_source
            if checkpoint == self.pending_checkpoint
            else self.state["checkpoint_source"]
        ), "Validation actor source mismatch"
        assert result["dataset"] == self.state["plan"]["validation_dataset"], (
            "Validation data changed"
        )
        candidates.append(result)
        self.state["best"] = min(
            candidates, key=lambda r: (-r["accuracy"], r["checkpoint"]["iteration"])
        )
        write_json(self.path, self.state)
        return {
            "eval/validation/pass_at_1": result["accuracy"],
            "eval/validation/round_std_population": result[
                "round_accuracy_std_population"
            ],
            "eval/validation/total_count": result["total_count"],
            "eval/validation/rounds_completed": result["rounds_completed"],
            "eval/validation/best_pass_at_1": self.state["best"]["accuracy"],
            "eval/validation/best_iteration": self.state["best"]["checkpoint"][
                "iteration"
            ],
        }

    def freeze(self):
        assert not self.state["frozen"] and not self.frozen_path.exists()
        assert [
            r["checkpoint"]["iteration"] for r in self.state["candidates"]
        ] == self.state["candidate_iterations"], (
            "Cannot freeze an incomplete validation schedule"
        )
        frozen = self.state | {"frozen": True}
        selected_checkpoint(frozen, self.args.task, self.args.output)
        write_json(self.frozen_path, frozen, exclusive=True)
        self.state = frozen
        write_json(self.path, self.state)
        return frozen

    def test(self):
        assert self.state["frozen"] and self.frozen_path.is_file()
        before = self.frozen_path.read_bytes()
        started = perf_counter()
        result = self._evaluate("test")
        assert self.frozen_path.read_bytes() == before, "Test mutated frozen selection"
        assert result["checkpoint"] == self.state["best"]["checkpoint"]
        assert result["protocol"] == self.state["test_protocol"]
        return result, perf_counter() - started
