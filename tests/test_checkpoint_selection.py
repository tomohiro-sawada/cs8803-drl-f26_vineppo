"""Selection orchestration is deterministic, split-safe and fail-loud."""

import json
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from experiments.training import evaluate as evaluation
from experiments.training import selection


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setitem(evaluation.EXPECTED_ROWS["validation"], "gsm8k", 2)
    args = SimpleNamespace(
        output=tmp_path,
        task="gsm8k",
        evaluation="validation-test",
        resume=None,
        resume_selection=None,
        dataset_path=tmp_path / "data",
        num_iterations=5,
        validation_rounds=2,
        eval_seed=42,
        eval_batch_size=4,
        eval_timeout_seconds=60,
        wandb_mode="disabled",
        wandb_project="test",
        expected_gpu="A40",
        wandb_entity=None,
        wandb_group=None,
        backend="hf",
    )
    evaluation.write_json(
        tmp_path / "metadata.json",
        {
            "trainer_config": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "num_iterations": 5,
            },
            "dataset": {"path": str(args.dataset_path), "hash": "train"},
            "model": {"path": "model"},
            "prompt": "prompt",
            "tokenizer_sha256": "tokenizer",
        },
    )
    monkeypatch.setattr(
        selection,
        "load_examples",
        lambda *a: (
            [],
            {
                "path": "data",
                "rows": 2,
                "fingerprint": "fixed",
                "examples_sha256": "fixed",
            },
        ),
    )
    (tmp_path / "checkpoints").mkdir()
    controller = selection.CheckpointSelection(args, checkpoint_every=2)
    return args, controller


def result(controller, path, iteration, correct=1):
    rows = [
        {
            "round": i,
            "seed": controller.args.eval_seed + i,
            "complete": True,
            "total_count": 2,
            "total_correct": correct,
            "accuracy": correct / 2,
        }
        for i in range(controller.args.validation_rounds)
    ]
    return {
        "schema_version": 1,
        "complete": True,
        "task": "gsm8k",
        "split": "validation",
        "checkpoint": {
            "path": str(path),
            "iteration": iteration,
            "sha256": selection.file_sha256(path),
        },
        "protocol": controller.state["validation_protocol"],
        "dataset": {
            "path": "data",
            "rows": 2,
            "fingerprint": "fixed",
            "examples_sha256": "fixed",
        },
        "training_run": {"path": str(controller.args.output), "wandb": None},
        "checkpoint_source": controller.state["checkpoint_source"],
        "accuracy": correct / 2,
        "total_correct": correct * len(rows),
        "total_count": 2 * len(rows),
        "num_examples": 2,
        "rounds_completed": len(rows),
        "round_accuracy_std_population": 0.0,
        "rounds": rows,
    }


def candidate(controller, iteration):
    path = controller.args.output / "checkpoints" / f"iteration-{iteration}.pt"
    path.write_bytes(f"weights {iteration}".encode())
    return path


def test_ties_choose_earliest_freeze_then_test_cannot_change_best(case, monkeypatch):
    _, controller = case
    state = random.getstate(), torch.get_rng_state()
    paths = [candidate(controller, i) for i in (2, 4, 5)]
    outcomes = [
        result(controller, p, i, n) for p, i, n in zip(paths, (2, 4, 5), (1, 2, 2))
    ]
    monkeypatch.setattr(controller, "_evaluate", Mock(side_effect=outcomes))
    for path, i in zip(paths, (2, 4, 5)):
        controller.validate(path, i)
    assert controller.best_path == paths[1]
    with monkeypatch.context() as patch:
        patch.setattr("os.link", Mock(side_effect=OSError("interrupted publication")))
        with pytest.raises(OSError, match="interrupted publication"):
            controller.freeze()
        assert not controller.frozen_path.exists()
        assert not controller.state["frozen"]
    frozen = controller.freeze()
    assert frozen["best"] == outcomes[1] and frozen["frozen"]
    original = controller.frozen_path.read_bytes()
    test = outcomes[1] | {
        "split": "test",
        "accuracy": 0.0,
        "protocol": frozen["test_protocol"],
    }
    monkeypatch.setattr(controller, "_evaluate", Mock(return_value=test))
    actual, seconds = controller.test()
    assert actual == test and seconds >= 0
    assert controller.frozen_path.read_bytes() == original
    assert controller.state["best"] == outcomes[1]
    assert random.getstate() == state[0] and torch.equal(
        torch.get_rng_state(), state[1]
    )
    with pytest.raises(AssertionError):
        controller.validate(paths[-1], 5)
    with pytest.raises(AssertionError):
        controller.freeze()


@pytest.mark.parametrize(
    "failure", ["incomplete", "test", "hash", "iteration", "protocol"]
)
def test_invalid_validation_cannot_select(case, monkeypatch, failure):
    _, controller = case
    path = candidate(controller, 2)
    evidence = result(controller, path, 2)
    if failure == "incomplete":
        evidence["complete"] = False
    if failure == "test":
        evidence["split"] = "test"
    if failure == "hash":
        evidence["checkpoint"]["sha256"] = "0" * 64
    if failure == "iteration":
        evidence["checkpoint"]["iteration"] = 3
    if failure == "protocol":
        evidence["protocol"] = evidence["protocol"] | {"batch_size": 8}
    monkeypatch.setattr(controller, "_evaluate", Mock(return_value=evidence))
    with pytest.raises(AssertionError):
        controller.validate(path, 2)
    assert controller.best_path is None and controller.state["candidates"] == []


def test_missing_schedule_and_changed_dataset_rejected(case, monkeypatch):
    _, controller = case
    path = candidate(controller, 2)
    monkeypatch.setattr(
        controller, "_evaluate", Mock(return_value=result(controller, path, 2))
    )
    with pytest.raises(AssertionError):
        controller.freeze()
    with pytest.raises(AssertionError):
        controller.validate(path, 4)
    controller.validate(path, 2)
    path2 = candidate(controller, 4)
    changed = result(controller, path2, 4)
    changed["dataset"]["fingerprint"] = "changed"
    monkeypatch.setattr(controller, "_evaluate", Mock(return_value=changed))
    with pytest.raises(AssertionError):
        controller.validate(path2, 4)
    assert controller.best_path == path


def test_subprocess_argv_timeout_and_failure_preserves_evidence(case, monkeypatch):
    _, controller = case
    path = candidate(controller, 2)
    error = subprocess.CalledProcessError(1, ["evaluation"])
    run = Mock(side_effect=error)
    monkeypatch.setattr(selection.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        controller.validate(path, 2)
    command = run.call_args.args[0]
    assert command[:3] == [
        selection.sys.executable,
        "-m",
        "experiments.training.evaluate",
    ]
    assert command[command.index("--split") + 1] == "validation"
    assert "--selection" not in command
    assert (
        run.call_args.kwargs["timeout"] == 60 and run.call_args.kwargs["check"] is True
    )
    assert "shell" not in run.call_args.kwargs
    assert path.exists() and controller.best_path is None
    assert (controller.directory / "validation-2.log").exists()
    assert (
        json.loads((controller.directory / "validation-2-command.json").read_text())
        == command
    )
    assert not controller.frozen_path.exists()


def resume_args(controller, checkpoint):
    args = SimpleNamespace(**vars(controller.args))
    args.resume = checkpoint
    args.resume_selection = controller.path
    args.output = controller.args.output / "resumed"
    args.output.mkdir()
    metadata = json.loads((controller.args.output / "metadata.json").read_text())
    metadata["trainer_config"]["checkpoint_dir"] = str(args.output / "checkpoints")
    evaluation.write_json(args.output / "metadata.json", metadata)
    return args


def test_resume_recovers_interrupted_validation_and_keeps_prior_best(case, monkeypatch):
    _, old = case
    first, interrupted = candidate(old, 2), candidate(old, 4)
    monkeypatch.setattr(old, "_evaluate", Mock(return_value=result(old, first, 2, 2)))
    old.validate(first, 2)
    before = old.path.read_bytes(), first.read_bytes(), interrupted.read_bytes()
    args = resume_args(old, interrupted)
    new = selection.CheckpointSelection(args, 2, 4)
    assert new.best_path == first
    assert new.state["candidates"][0]["training_run"]["path"] == str(args.output)
    assert new.state["candidates"][0]["checkpoint_source"]["path"] == str(
        old.args.output
    )
    evidence = result(new, interrupted, 4) | {
        "checkpoint_source": old.state["checkpoint_source"]
    }
    evaluate = Mock(return_value=evidence)
    monkeypatch.setattr(new, "_evaluate", evaluate)
    new.recover(4)
    evaluate.assert_called_once_with(
        "validation",
        checkpoint=interrupted,
        iteration=4,
        checkpoint_sha256=selection.file_sha256(interrupted),
    )
    new.recover(4)
    assert evaluate.call_count == 1
    last = args.output / "iteration-5.pt"
    last.write_bytes(b"last")
    monkeypatch.setattr(new, "_evaluate", Mock(return_value=result(new, last, 5)))
    new.validate(last, 5)
    frozen = new.freeze()
    assert frozen["best"]["checkpoint"]["path"] == str(first)
    assert selection.selected_checkpoint(frozen, "gsm8k", args.output) == first
    assert before == (
        old.path.read_bytes(),
        first.read_bytes(),
        interrupted.read_bytes(),
    )
    from experiments.training.tracking import WandBTracker

    tracked = SimpleNamespace(
        id="new-run",
        url=None,
        summary={},
        log=Mock(),
        define_metric=Mock(),
        finish=Mock(),
    )
    sdk = SimpleNamespace(init=Mock(return_value=tracked), Settings=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    args.wandb_mode = "offline"
    with WandBTracker(args, {"start_iteration": 4, "cli": vars(args)}) as tracker:
        tracker.log_selection(frozen)
        tracker.log_final_test(frozen["best"] | {"split": "test"})
    tracked.log.assert_not_called()  # The selected iteration 2 must not rewind history.
    assert sdk.init.call_args.kwargs["resume"] == "never"
    assert tracked.summary["selected_checkpoint"]["iteration"] == 2
    assert sdk.init.call_args.kwargs["config"]["cli"]["resume"] == str(interrupted)


@pytest.mark.parametrize(
    "mutation",
    ["lookahead", "schedule", "cadence", "protocol", "data", "metadata", "frozen"],
)
def test_resume_rejects_scope_and_evidence_tampering(case, monkeypatch, mutation):
    _, old = case
    first, second = candidate(old, 2), candidate(old, 4)
    monkeypatch.setattr(old, "_evaluate", Mock(return_value=result(old, first, 2)))
    old.validate(first, 2)
    if mutation == "lookahead":
        monkeypatch.setattr(old, "_evaluate", Mock(return_value=result(old, second, 4)))
        old.validate(second, 4)
    args = resume_args(old, first)
    if mutation == "schedule":
        args.num_iterations = 6
    if mutation == "protocol":
        args.eval_seed += 1
    if mutation == "data":
        monkeypatch.setattr(
            selection, "load_examples", lambda *a: ([], {"tampered": True})
        )
    if mutation == "metadata":
        (old.args.output / "metadata.json").write_text("{}")
    if mutation == "frozen":
        old.state["frozen"] = True
        evaluation.write_json(old.path, old.state)
    with pytest.raises(AssertionError):
        selection.CheckpointSelection(args, 1 if mutation == "cadence" else 2, 2)


def test_resume_completed_candidate_is_not_evaluated_again(case, monkeypatch):
    _, old = case
    checkpoint = candidate(old, 2)
    monkeypatch.setattr(old, "_evaluate", Mock(return_value=result(old, checkpoint, 2)))
    old.validate(checkpoint, 2)
    new = selection.CheckpointSelection(resume_args(old, checkpoint), 2, 2)
    evaluate = Mock(side_effect=AssertionError("must not evaluate twice"))
    monkeypatch.setattr(new, "_evaluate", evaluate)
    new.recover(2)
    evaluate.assert_not_called()
