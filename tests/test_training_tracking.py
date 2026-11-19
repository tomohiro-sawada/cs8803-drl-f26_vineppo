"""Tracker contracts using a fake SDK: no live runs, network, or .env reads."""

import builtins
import importlib.util
import json
import os
import socket
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/training/tracking.py"
spec = importlib.util.spec_from_file_location("training_tracking", SCRIPT)
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network or .env access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    # Even accidental direct reads must not touch the real project's .env.
    for owner, name in ((builtins, "open"), (Path, "open")):
        original = getattr(owner, name)

        def guarded(path, *args, _open=original, **kwargs):
            if Path(path).name == ".env":
                forbidden()
            return _open(path, *args, **kwargs)

        monkeypatch.setattr(owner, name, guarded)
    dotenv = SimpleNamespace(dotenv_values=Mock(side_effect=forbidden))
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    # Register restoration even if the key was originally absent and code adds it.
    monkeypatch.setenv("WANDB_API_KEY", "")
    monkeypatch.delenv("WANDB_API_KEY")
    return dotenv


@pytest.fixture
def sdk(monkeypatch):
    run = SimpleNamespace(
        id="fresh-id",
        url="https://wandb.invalid/run/fresh-id",
        summary={},
        define_metric=Mock(),
        log=Mock(),
        finish=Mock(),
    )
    sdk = SimpleNamespace(
        init=Mock(return_value=run),
        Settings=Mock(side_effect=lambda **kw: kw),
        util=SimpleNamespace(generate_id=Mock(return_value=run.id)),
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.setattr(
        tracking, "uuid4", lambda: SimpleNamespace(hex=sdk.util.generate_id())
    )
    return sdk


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(wandb_mode="offline", mode="train", output=tmp_path)


def row(iteration):
    return {
        "iteration": iteration,
        "actor/loss": 0.25,
        "critic/loss": 0.5,
        "actor/approx_kl": 0.01,
        "actor/ref_kl": 0.02,
        "actor/learning_rate": 1e-6,
        "warmup": True,
        "primary_tokens": 10,
        "mc_tokens": 20,
        "sampled_tokens": 30,
        "primary_continuations": 2,
        "mc_continuations": 4,
        "mc_expansions": 1,
        "iteration_seconds": 0.5,
    }


def test_disabled_is_noop_without_import_env_or_files(args, monkeypatch, sdk):
    args.wandb_mode = "disabled"
    args.output = args.output / "must-not-exist"
    original_import = builtins.__import__

    def guarded_import(name, *a, **kw):
        assert name not in ("wandb", "dotenv")
        return original_import(name, *a, **kw)

    class NoEnvironment:
        def __getattribute__(self, name):
            pytest.fail("Disabled mode accessed the environment")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded_import)
        patch.setattr(tracking.os, "environ", NoEnvironment())
        with tracking.WandBTracker(args, {}) as tracker:
            tracker.log_iteration({})
            tracker.finish_summary({})
            tracker.log_evaluation({}, selection={})
            tracker.log_selection({})
            tracker.log_final_test({})
    sdk.init.assert_not_called()
    assert not args.output.exists()


@pytest.mark.parametrize("mode", ["train", "benchmark"])
def test_offline_safe_config_settings_and_identity(args, sdk, isolated, mode):
    args.mode = mode
    metadata = {
        "cli": vars(args)
        | {
            "resume": Path("checkpoint.pt"),
            "api_key": "SECRET",
            "future_credential": "SECRET",
            "env_file": ".env",
            "evaluation": True,
            "validation_rounds": 2,
            "eval_batch_size": 4,
            "eval_seed": 42,
            "eval_timeout_seconds": 600,
        },
        "trainer_config": {"checkpoint_dir": Path("checkpoints")},
        "dataset": {"rows": 8},
        "gpu": {"name": "fake"},
        "software": {"python": "test"},
        "git": {"commit": "abc"},
        "model": {"path": Path("model")},
        "precision": "FP32",
        "start_iteration": 9,
        "evaluation": {"split": "validation", "rounds": 2},
        "training_run": {"path": "training"},
        "checkpoint": {"iteration": 9},
        "selection": {"frozen": True},
        "environment": {"API_KEY": "SECRET"},
        "dotenv": "SECRET",
        "unknown": "SECRET",
    }
    with tracking.WandBTracker(args, metadata):
        pass
    kw = sdk.init.call_args.kwargs
    assert {
        k: kw[k]
        for k in ("mode", "project", "entity", "group", "name", "dir", "resume", "id")
    } == {
        "mode": "offline",
        "project": "vineppo",
        "entity": None,
        "group": None,
        "name": None,
        "dir": str(args.output),
        "resume": "never",
        "id": "fresh-id",
    }
    assert kw["settings"] == {
        "console": "off",
        "disable_code": True,
        "disable_git": True,
        "x_disable_meta": True,
        "x_save_requirements": False,
        "stop_on_fatal_error": True,
        "finish_timeout": 60,
        "finish_timeout_raises": True,
    }
    config = kw["config"]
    assert set(config) == set(tracking.CONFIG_FIELDS) | {"budget_scope"}
    assert config["cli"]["resume"] == "checkpoint.pt"
    assert config["cli"]["mode"] == mode
    for key in (
        "evaluation",
        "validation_rounds",
        "eval_batch_size",
        "eval_seed",
        "eval_timeout_seconds",
    ):
        assert config["cli"][key] == metadata["cli"][key]
    assert config["trainer_config"]["checkpoint_dir"] == "checkpoints"
    assert config["model"]["path"] == "model"
    assert "SECRET" not in json.dumps(kw)
    assert config["budget_scope"] == "invocation"
    assert config["start_iteration"] == 9
    isolated.dotenv_values.assert_not_called()
    sdk.init.return_value.define_metric.assert_has_calls(
        [
            call("iteration"),
            call("*", step_metric="iteration"),
        ]
    )
    sdk.init.return_value.finish.assert_called_once_with(exit_code=0)
    assert json.loads((args.output / "wandb.json").read_text()) == {
        "id": "fresh-id",
        "url": None,
        "mode": "offline",
    }
    assert metadata["cli"]["api_key"] == "SECRET"


def test_resume_budget_resets_and_steps_increase(args, sdk):
    sdk.util.generate_id.side_effect = ["first", "second"]
    for directory, start in (("first", 0), ("resumed", 20)):
        args.output = args.output / directory
        args.output.mkdir()
        with tracking.WandBTracker(args, {"start_iteration": start}) as tracker:
            for offset in (1, 2):
                metrics = row(start + offset)
                tracker.log_iteration(metrics)
                logged = sdk.init.return_value.log.call_args
                assert logged.kwargs == {"step": start + offset}
                assert logged.args[0] == metrics | {
                    f"budget/{key}": metrics[key] * offset
                    for key in tracking.BUDGET_FIELDS
                }
                assert not any(key.startswith("budget/") for key in metrics)
            with pytest.raises(AssertionError, match="Iterations must increase"):
                tracker.log_iteration(row(start + 2))
    assert [c.kwargs["id"] for c in sdk.init.call_args_list] == ["first", "second"]
    assert all(c.kwargs["resume"] == "never" for c in sdk.init.call_args_list)


def test_summary_preserves_runner_schema(args, sdk):
    summary = {
        "schema_version": 1,
        "mode": "benchmark",
        "completed_iteration": 12,
        "iterations_executed": 2,
        "iterations_measured": 1,
        "warmup_iterations_excluded": 1,
        "iteration_seconds_total": 2.0,
        "iteration_seconds_mean": 2.0,
        "sampled_tokens": 30,
        "sampled_tokens_per_second": 15.0,
        "phase_seconds_total": {"actor_update": 1.0},
        "sampling_totals": {"primary_tokens": 10},
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "checkpoints": [{"path": Path("ckpt.pt"), "bytes": 10}],
    }
    assert set(summary) == tracking.SUMMARY_FIELDS
    with tracking.WandBTracker(args, {}) as tracker:
        tracker.finish_summary(summary | {"api_key": "SECRET", "environment": "SECRET"})
    assert sdk.init.return_value.summary == json.loads(json.dumps(summary, default=str))


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_training_failure_finishes_without_uploading_message(args, sdk, failure):
    error = failure("SECRET training error")
    with pytest.raises(failure) as raised, tracking.WandBTracker(args, {}):
        raise error
    assert raised.value is error
    assert sdk.init.return_value.summary == {"failure_type": failure.__name__}
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_finish_failure_propagates_on_success(args, sdk):
    sdk.init.return_value.finish.side_effect = TimeoutError("finish timed out")
    with pytest.raises(TimeoutError), tracking.WandBTracker(args, {}):
        pass


def test_finish_failure_does_not_mask_training_failure(args, sdk):
    sdk.init.return_value.finish.side_effect = TimeoutError("SECRET SDK failure")
    error = ValueError("SECRET training failure")
    with pytest.raises(ValueError) as raised, tracking.WandBTracker(args, {}):
        raise error
    assert raised.value is error
    assert error.__notes__ == ["W&B cleanup failed: TimeoutError"]
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_init_failure_propagates(args, sdk):
    sdk.init.side_effect = RuntimeError("init failed")
    with (
        pytest.raises(RuntimeError, match="init failed"),
        tracking.WandBTracker(args, {}),
    ):
        pytest.fail("Must not enter training")
    sdk.init.return_value.finish.assert_not_called()
    assert not (args.output / "wandb.json").exists()


def test_identity_write_is_exclusive_and_cleans_up(args, sdk):
    path = args.output / "wandb.json"
    path.write_text("existing identity")
    with pytest.raises(FileExistsError), tracking.WandBTracker(args, {}):
        pytest.fail("Must not overwrite identity")
    assert path.read_text() == "existing identity"
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("environment_key", [None, "env-key", ""])
def test_online_key_precedence(args, sdk, isolated, monkeypatch, environment_key):
    args.wandb_mode = "online"
    args.wandb_project, args.wandb_entity = "project", "entity"
    args.wandb_group, args.wandb_name = "group", "name"
    isolated.dotenv_values.side_effect = None
    isolated.dotenv_values.return_value = {
        "WANDB_API_KEY": "dotenv-key",
        "OTHER_SECRET": "SECRET",
    }
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    if environment_key is not None:
        monkeypatch.setenv("WANDB_API_KEY", environment_key)
    if environment_key == "":
        with (
            pytest.raises(RuntimeError, match="requires WANDB_API_KEY"),
            tracking.WandBTracker(args, {}),
        ):
            pytest.fail("Empty environment key must fail")
        isolated.dotenv_values.assert_not_called()
        sdk.init.assert_not_called()
        return
    with tracking.WandBTracker(args, {}):
        assert os.environ["WANDB_API_KEY"] == (environment_key or "dotenv-key")
        assert "OTHER_SECRET" not in os.environ
    if environment_key is None:
        isolated.dotenv_values.assert_called_once_with(
            tracking.ROOT / ".env", interpolate=False
        )
    else:
        isolated.dotenv_values.assert_not_called()
    kw = sdk.init.call_args.kwargs
    assert all(kw[key] == key for key in ("project", "entity", "group", "name"))
    assert "dotenv-key" not in json.dumps(kw)
    assert "env-key" not in json.dumps(kw)
    assert json.loads((args.output / "wandb.json").read_text()) == {
        "id": "fresh-id",
        "url": sdk.init.return_value.url,
        "mode": "online",
    }


@pytest.mark.parametrize("key", [None, "", "   "])
def test_online_missing_key_fails_without_prompt(args, sdk, isolated, key):
    args.wandb_mode = "online"
    isolated.dotenv_values.side_effect = None
    isolated.dotenv_values.return_value = {"WANDB_API_KEY": key}
    with (
        pytest.raises(RuntimeError, match="requires WANDB_API_KEY"),
        tracking.WandBTracker(args, {}),
    ):
        pytest.fail("Missing credentials must fail")
    sdk.init.assert_not_called()
    assert not (args.output / "wandb.json").exists()


@pytest.mark.parametrize(
    "update",
    [{"actor/loss": float("nan")}, {"actor/loss": "SECRET"}, {"mc_tokens": -1}],
)
def test_invalid_metrics_fail_loudly(args, sdk, update):
    with tracking.WandBTracker(args, {}) as tracker, pytest.raises(AssertionError):
        tracker.log_iteration(row(1) | update)
    sdk.init.return_value.log.assert_not_called()


def evaluation_result(split="validation"):
    return {
        "schema_version": 1,
        "complete": True,
        "task": "gsm8k",
        "split": split,
        "accuracy": 0.75,
        "round_accuracy_std_population": 0.25,
        "num_examples": 2,
        "total_count": 4,
        "total_correct": 3,
        "rounds_completed": 2,
        "dataset": {"rows": 2},
        "protocol": {"rounds": 2, "seed": 42},
        "checkpoint": {
            "path": "/training/checkpoint.pt",
            "sha256": "a" * 64,
            "iteration": 3,
        },
        "training_run": {
            "path": "/training",
            "wandb": {
                "id": "parent",
                "url": "https://wandb.invalid/parent",
                "mode": "online",
            },
        },
        "rounds": [
            {
                "round": i,
                "seed": 42 + i,
                "complete": True,
                "accuracy": correct / 2,
                "total_count": 2,
                "total_correct": correct,
                "samples": "SECRET",
            }
            for i, correct in enumerate((1, 2))
        ],
        "samples": "SECRET",
        "code": "SECRET",
        "text": "SECRET",
    }


def frozen_selection():
    best = evaluation_result()
    return {
        "schema_version": 1,
        "frozen": True,
        "task": "gsm8k",
        "training_output": "/training",
        "best": best,
        "candidates": [best],
        "tie_rule": "highest_validation_accuracy_then_earliest_iteration",
        "test_protocol": {"rounds": 2, "seed": 42},
    }


def evaluation_metrics(split):
    return {
        f"eval/{split}/pass_at_1": 0.75,
        f"eval/{split}/round_std_population": 0.25,
        f"eval/{split}/num_examples": 2,
        f"eval/{split}/total_count": 4,
        f"eval/{split}/rounds_completed": 2,
    }


@pytest.mark.parametrize("split", ["validation", "test"])
def test_child_evaluation_scalars_summary_and_privacy(args, sdk, isolated, split):
    result = evaluation_result(split)
    metadata = {key: result[key] for key in ("checkpoint", "training_run")}
    metadata["evaluation"] = {"split": split, **result["protocol"]}
    original = deepcopy(result)
    with tracking.WandBTracker(args, metadata) as tracker:
        tracker.log_evaluation(result)
        assert tracker._budget == dict.fromkeys(tracking.BUDGET_FIELDS, 0)
    run = sdk.init.return_value
    run.log.assert_called_once_with(
        evaluation_metrics(split) | {"iteration": 3}, step=3
    )
    assert run.summary == evaluation_metrics(split) | {
        "iteration": 3,
        "checkpoint": result["checkpoint"],
        "training_run": result["training_run"],
    }
    for key, value in metadata.items():
        assert sdk.init.call_args.kwargs["config"][key] == value
    assert "SECRET" not in json.dumps(run.summary)
    assert "SECRET" not in str(run.log.call_args_list)
    assert result == original
    isolated.dotenv_values.assert_not_called()


def test_test_child_includes_frozen_selection(args, sdk):
    selection = frozen_selection()
    with tracking.WandBTracker(args, {}) as tracker:
        tracker.log_evaluation(evaluation_result("test"), selection=selection)
    assert sdk.init.return_value.summary["eval/validation/best_pass_at_1"] == 0.75
    assert (
        sdk.init.return_value.summary["selected_checkpoint"]
        == selection["best"]["checkpoint"]
    )
    assert "SECRET" not in json.dumps(sdk.init.return_value.summary)


def test_parent_summaries_never_advance_history_or_budget(args, sdk):
    with tracking.WandBTracker(args, {}) as tracker:
        tracker.log_iteration(row(10) | evaluation_metrics("validation"))
        history = list(sdk.init.return_value.log.call_args_list)
        budget = dict(tracker._budget)
        tracker.log_selection(frozen_selection())
        tracker.log_final_test(evaluation_result("test"))
        assert sdk.init.return_value.log.call_args_list == history
        assert tracker._last_iteration == 10 and tracker._budget == budget
        assert sdk.init.return_value.summary == evaluation_metrics("test") | {
            "eval/validation/best_pass_at_1": 0.75,
            "selected_checkpoint": evaluation_result()["checkpoint"],
        }
        tracker.log_iteration(row(11))
    assert [c.kwargs["step"] for c in sdk.init.return_value.log.call_args_list] == [
        10,
        11,
    ]


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize(
    "update",
    [
        {"complete": False},
        {"split": "train"},
        {"accuracy": float("nan")},
        {"accuracy": float("inf")},
        {"accuracy": -0.1},
        {"accuracy": 1.1},
        {"accuracy": True},
        {"accuracy": 0.5},
        {"round_accuracy_std_population": float("nan")},
        {"round_accuracy_std_population": 0.5},
        {"num_examples": 0},
        {"total_count": 3},
        {"total_correct": 2},
        {"rounds_completed": 1},
        {"rounds_completed": True},
    ],
)
def test_invalid_evaluation_is_atomic(args, sdk, split, update):
    with tracking.WandBTracker(args, {}) as tracker:
        with pytest.raises(AssertionError):
            tracker.log_evaluation(evaluation_result(split) | update)
        assert sdk.init.return_value.summary == {}
        sdk.init.return_value.log.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("total_count", 1),
        ("total_correct", 3),
        ("accuracy", float("nan")),
        ("seed", 0),
        ("round", 2),
    ],
)
def test_invalid_round_evidence(args, sdk, field, value):
    result = evaluation_result()
    result["rounds"][0][field] = value
    with tracking.WandBTracker(args, {}) as tracker:
        with pytest.raises(AssertionError):
            tracker.log_evaluation(result)
        sdk.init.return_value.log.assert_not_called()
        assert sdk.init.return_value.summary == {}


@pytest.mark.parametrize(
    "invalid",
    ["unfrozen", "test_best", "incomplete", "checkpoint", "protocol", "wrong_best"],
)
def test_invalid_selection_is_atomic(args, sdk, invalid):
    selection, result = frozen_selection(), evaluation_result("test")
    if invalid == "unfrozen":
        selection["frozen"] = False
    elif invalid == "test_best":
        selection["best"]["split"] = "test"
    elif invalid == "incomplete":
        selection["best"]["complete"] = False
    elif invalid == "checkpoint":
        result["checkpoint"]["iteration"] += 1
    elif invalid == "protocol":
        selection["test_protocol"]["seed"] += 1
    else:
        earlier = deepcopy(selection["best"])
        earlier["checkpoint"].update(iteration=1, path="/training/earlier.pt")
        selection["candidates"].append(earlier)
    with tracking.WandBTracker(args, {}) as tracker:
        with pytest.raises(AssertionError):
            tracker.log_evaluation(result, selection=selection)
        if invalid not in ("checkpoint", "protocol"):
            with pytest.raises(AssertionError):
                tracker.log_selection(selection)
        sdk.init.return_value.log.assert_not_called()
        assert sdk.init.return_value.summary == {}


def test_final_test_rejects_validation_and_incomplete(args, sdk):
    with tracking.WandBTracker(args, {}) as tracker:
        for result in (
            evaluation_result(),
            evaluation_result("test") | {"complete": False},
        ):
            with pytest.raises(AssertionError):
                tracker.log_final_test(result)
        sdk.init.return_value.log.assert_not_called()
        assert sdk.init.return_value.summary == {}


@pytest.mark.parametrize(
    "method", ["log_evaluation", "log_selection", "log_final_test"]
)
def test_evaluation_apis_require_context(args, method):
    with pytest.raises(AssertionError, match="Enter the tracker"):
        getattr(tracking.WandBTracker(args, {}), method)({})


def test_offline_never_reads_environment(args, sdk, monkeypatch):
    class NoEnvironment:
        def __getattribute__(self, name):
            pytest.fail("Offline tracker accessed the environment")

    with monkeypatch.context() as patch:
        patch.setattr(tracking.os, "environ", NoEnvironment())
        with tracking.WandBTracker(args, {}) as tracker:
            tracker.log_iteration(row(1))
            tracker.log_evaluation(
                evaluation_result("test"), selection=frozen_selection()
            )
            tracker.log_selection(frozen_selection())
            tracker.log_final_test(evaluation_result("test"))
    sdk.init.return_value.finish.assert_called_once_with(exit_code=0)
