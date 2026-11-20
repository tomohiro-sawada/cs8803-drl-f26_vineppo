"""CPU contract tests: real local DatasetDict and verifier, no model/network/W&B."""

import copy
import json
import socket
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset, DatasetDict

from core.inference import TextCompletion
from experiments.training import evaluate as runner
from experiments.training import finalize as finalizer


def read(path):
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network, model, or tracker access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(runner, "load_checkpoint_actor", forbidden)
    monkeypatch.setattr(runner, "WandBTracker", forbidden)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    training = tmp_path / "training"
    training.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    dataset = tmp_path / "data"
    DatasetDict(
        {
            split: Dataset.from_list(
                [
                    {
                        "question": f" {split} Q{i} ",
                        "answer": f"private rationale #### {i}",
                    }
                    for i in range(3)
                ]
            )
            for split in ("train", "validation", "test")
        }
    ).save_to_disk(str(dataset))
    for split in runner.EXPECTED_ROWS:
        monkeypatch.setitem(runner.EXPECTED_ROWS[split], "gsm8k", 3)
    checkpoint = training / "iteration-1.pt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "cli": {
            "task": "gsm8k",
            "model_path": str(model),
            "dataset_path": str(dataset),
            "algorithm": "ppo",
            "dtype": "float32",
            "finetuning": "full",
        },
        "prompt": runner.PROMPT,
        "tokenizer_sha256": "tokenizer",
        "trainer_config": {"num_iterations": 2, "algorithm_config": {}},
        "model": {"path": str(model), "lora": None},
        "dataset": {
            "path": str(dataset),
            "split": "train",
            "rows": 3,
            "fingerprint": runner.load_from_disk(str(dataset))["train"]._fingerprint,
            "normalized_sha256": "a" * 64,
        },
    }
    runner.write_json(training / "metadata.json", metadata)
    runner.write_json(
        training / "wandb.json", {"id": "training-id", "url": None, "mode": "offline"}
    )
    info = {
        "checkpoint_sha256": runner.file_sha256(checkpoint),
        "iteration": 1,
        "tokenizer_sha256": "tokenizer",
        "dtype": "float32",
        "finetuning": "full",
        "lora": None,
        "evaluation_provenance": {
            "schema_version": 1,
            "task": "gsm8k",
            "prompt": runner.PROMPT,
            "tokenizer_sha256": "tokenizer",
            "dataset": copy.deepcopy(metadata["dataset"]),
        },
        "training_config": metadata["trainer_config"]
        | {"algorithm_type": "algorithms.ppo.PPOConfig"},
    }

    def load_actor(path, model_path, *, device, expected_sha256):
        assert expected_sha256 is None or runner.file_sha256(path) == expected_sha256
        return object(), object(), info

    loader = Mock(side_effect=load_actor)
    monkeypatch.setattr(runner, "load_checkpoint_actor", loader)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda i: "Mock GPU")
    monkeypatch.setattr(runner.torch.cuda, "set_device", Mock())
    seeds, prompts = [], []
    monkeypatch.setattr(runner, "set_seed", seeds.append)

    class Generator:
        closed = False
        calls = 0
        fail_at = None

        def generate(self, batch, **kwargs):
            self.calls += 1
            if self.calls == self.fail_at:
                raise RuntimeError("generation failed")
            assert kwargs == {
                "num_samples": 1,
                "max_new_tokens": 1024,
                "temperature": 0.35,
            }
            prompts.extend(batch)
            # Round 0 all right; round 1 all wrong. Truncation is not penalized.
            return [
                [TextCompletion(f"#### {i if seeds[-1] % 2 == 0 else 99}", "length")]
                for prompt in batch
                for i in [int(prompt.split(" Q")[1].split("\n")[0])]
            ]

        def close(self):
            self.closed = True

    generator = Generator()
    monkeypatch.setattr(runner, "PaperGenerator", lambda *args: generator)
    tracker = Mock()
    tracker.__enter__ = Mock(return_value=tracker)
    tracker.__exit__ = Mock(return_value=False)
    tracker_factory = Mock(return_value=tracker)
    monkeypatch.setattr(runner, "WandBTracker", tracker_factory)
    argv = [
        "--split",
        "validation",
        "--training-output",
        str(training),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(tmp_path / "evaluation"),
        "--rounds",
        "2",
        "--batch-size",
        "2",
    ]
    return SimpleNamespace(**locals())


def freeze(s, result, *, eval_seed=42, eval_batch_size=4):
    selection = {
        "schema_version": 1,
        "frozen": True,
        "plan": {
            "validation_dataset": copy.deepcopy(result["dataset"]),
            "test_dataset": runner.load_examples(s.dataset, "gsm8k", "test")[1],
        },
        "task": "gsm8k",
        "training_output": str(s.training),
        "best": result,
        "candidates": [result],
        "candidate_iterations": [result["checkpoint"]["iteration"]],
        "validation_protocol": copy.deepcopy(result["protocol"]),
        "tie_rule": "highest_validation_accuracy_then_earliest_iteration",
        "test_protocol": runner.PROTOCOL
        | {"rounds": 16, "seed": eval_seed, "batch_size": eval_batch_size},
    }
    path = s.tmp_path / "selection.json"
    runner.write_json(path, selection)
    return path, selection


@pytest.fixture
def frozen_run(setup):
    s = setup
    root = s.tmp_path / "resumable"
    root.mkdir()
    owner = root / "attempt-00"
    checkpoint = owner / s.checkpoint.name
    previous = s.training
    previous.rename(owner)
    s.argv = [
        {str(previous): str(owner), str(s.checkpoint): str(checkpoint)}.get(arg, arg)
        for arg in s.argv
    ]
    s.training, s.checkpoint = owner, checkpoint
    manifest = owner / "selection-frozen.json"
    manifest.touch()
    args = finalizer.parse_args(
        [
            "--mode",
            "evaluate",
            "--algorithm",
            "ppo",
            "--task",
            "gsm8k",
            "--model-path",
            str(s.model),
            "--dataset-path",
            str(s.dataset),
            "--output",
            str(root),
            "--resume-selection",
            str(manifest),
            "--evaluation",
            "validation-test",
            "--num-iterations",
            "1",
            "--validation-rounds",
            "2",
            "--eval-batch-size",
            "2",
        ]
    )
    config = asdict(finalizer.trainer_config(args))
    config["checkpoint_dir"] = str(owner / "checkpoints")
    s.metadata["trainer_config"] = config
    s.info["training_config"] = config | {"algorithm_type": "algorithms.ppo.PPOConfig"}
    for key in ("lora_rank", "lora_alpha", "learning_rate", "critic_learning_rate"):
        s.metadata["cli"][key] = getattr(args, key)
    runner.write_json(owner / "metadata.json", s.metadata)
    candidate = runner.main(s.argv)
    _, selection = freeze(s, candidate, eval_batch_size=2)
    selection["plan"].update(
        trainer_config={
            key: value for key, value in config.items() if key != "checkpoint_dir"
        },
        evaluation="validation-test",
    )
    selection["checkpoint_source"] = candidate["checkpoint_source"]
    runner.write_json(manifest, selection)
    return args, selection


@pytest.mark.parametrize("interrupt", ["before", "during", "after_result"])
def test_frozen_finalization_restarts_or_adopts_completed_test(
    frozen_run, monkeypatch, interrupt
):
    args, selection = frozen_run
    original = args.resume_selection.read_bytes()
    marker = args.output / "test-complete.json"
    failed = args.output / "test-attempt-00"
    if interrupt == "during":
        failed.mkdir()
        (failed / "progress.json").write_text('{"rounds_completed": 3}')
    calls = []

    def run_test(command, *, check, timeout):
        calls.append(command)
        assert check and timeout == args.eval_timeout_seconds
        runner.main(command[3:])

    monkeypatch.setattr(finalizer.subprocess, "run", run_test)
    if interrupt == "after_result":
        with monkeypatch.context() as patch:
            saved_writer = runner.write_json

            def preempt(path, data, **options):
                if path == marker:
                    raise RuntimeError("preempted before completion marker")
                saved_writer(path, data, **options)

            patch.setattr(finalizer.evaluate, "write_json", preempt)
            with pytest.raises(RuntimeError, match="preempted"):
                finalizer.finalize(args, prepare=False)
        assert not marker.exists()
        assert finalizer.finalize(args, prepare=True) == "complete"
    else:
        assert finalizer.finalize(args, prepare=True) == "run"
        assert finalizer.finalize(args, prepare=False) == "complete"
    assert len(calls) == 1
    assert finalizer.finalize(args, prepare=True) == "complete"
    assert len(calls) == 1
    record = read(marker)
    assert record["checkpoint_sha256"] == selection["best"]["checkpoint"]["sha256"]
    assert read(Path(record["result"]))["rounds_completed"] == 16
    assert args.resume_selection.read_bytes() == original
    if interrupt == "during":
        assert read(failed / "progress.json") == {"rounds_completed": 3}
    Path(selection["best"]["checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Selected checkpoint hash mismatch"):
        finalizer.finalize(args, prepare=True)


def test_finalization_rejects_changed_frozen_protocol(frozen_run):
    args, _ = frozen_run
    args.eval_seed += 1
    with pytest.raises(AssertionError):
        finalizer.finalize(args, prepare=True)


def test_full_validation_contract(setup):
    s = setup
    result = runner.main(s.argv)
    assert read(s.tmp_path / "evaluation/result.json") == result
    assert result["accuracy"] == 0.5 and result["round_accuracy_std_population"] == 0.5
    assert result["total_count"] == 6 and result["total_correct"] == 3
    assert result["rounds_completed"] == 2 and result["num_examples"] == 3
    assert result["checkpoint"] == {
        "path": str(s.checkpoint),
        "sha256": s.info["checkpoint_sha256"],
        "iteration": 1,
    }
    assert s.seeds == [42, 43] and s.generator.closed
    assert all(
        "validation Q" in prompt and "private" not in prompt for prompt in s.prompts
    )
    for index in range(2):
        records = [
            json.loads(line)
            for line in (s.tmp_path / f"evaluation/round-{index:02d}/records.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["example_id"] for r in records] == [
            f"gsm8k:validation:{i}" for i in range(3)
        ]
        assert all(
            r["sample_index"] == 0 and r["finish_reason"] == "length" for r in records
        )
    assert "paper_target_accuracy" not in result and "pass@k" not in result
    runner.validate_validation(result, "gsm8k", s.training)
    s.tracker.log_evaluation.assert_called_once_with(result, selection=None)
    config = s.tracker_factory.call_args.args[1]
    assert config["training_run"]["wandb"]["id"] == "training-id"
    assert "validation Q" not in json.dumps(config)
    assert "private rationale" not in json.dumps(config)
    provenance = read(s.tmp_path / "evaluation/provenance.json")
    assert provenance["actor"]["dtype"] == "float32"
    assert (
        provenance["evaluation_run"]
        == result["evaluation_run"]
        == {
            "path": str(s.tmp_path / "evaluation"),
            "wandb": None,
        }
    )
    deviations = " ".join(provenance["deviations"])
    for detail in (
        "HF",
        "trims the stop string",
        "Sequential seeded rounds",
        "dtype preserved",
    ):
        assert detail in deviations
    assert "server-side" not in deviations
    with pytest.raises(FileExistsError):
        runner.main(s.argv)


def test_tracked_remote_evaluation_records_the_child_identity(setup, monkeypatch):
    s = setup
    s.info["lora"] = s.metadata["model"]["lora"] = {"rank": 8}
    s.info["finetuning"] = s.metadata["cli"]["finetuning"] = "lora"
    runner.write_json(s.training / "metadata.json", s.metadata)
    workers = [{"gpu": {"name": "Mock GPU"}, "software": {"vllm": "test-version"}}]
    monkeypatch.setenv("VLLM_API_KEY", "placeholder")
    monkeypatch.setattr(
        runner.VLLMClient, "from_directory", lambda *a, **kw: (Mock(), workers)
    )
    monkeypatch.setattr(runner, "RemotePaperGenerator", lambda *a: s.generator)
    monkeypatch.setattr(runner, "publish_adapter", lambda *a: "checkpoint-actor")
    child = {"id": "evaluation-id", "url": None, "mode": "offline"}

    def enter():
        runner.write_json(s.tmp_path / "evaluation/wandb.json", child)
        return s.tracker

    s.tracker.__enter__.side_effect = enter
    result = runner.main(
        s.argv
        + [
            "--backend",
            "vllm",
            "--num-workers",
            "1",
            "--worker-dir",
            str(s.tmp_path),
            "--wandb-mode",
            "offline",
        ]
    )
    provenance = read(s.tmp_path / "evaluation/provenance.json")
    assert provenance["workers"] == workers
    assert (
        provenance["evaluation_run"]
        == result["evaluation_run"]
        == {
            "path": str(s.tmp_path / "evaluation"),
            "wandb": child,
        }
    )
    assert result["training_run"]["wandb"]["id"] == "training-id"
    deviations = " ".join(provenance["deviations"])
    assert "server-side stopping and per-request hashed seeds" in deviations
    assert "trims the stop string" not in deviations


def test_frozen_test_uses_only_test_and_does_not_modify_selection(setup):
    s = setup
    best = runner.main(s.argv)
    path, selection = freeze(s, best)
    before = path.read_bytes()
    s.prompts.clear()
    s.seeds.clear()
    result = runner.main(
        [
            "--split",
            "test",
            "--training-output",
            str(s.training),
            "--selection",
            str(path),
            "--output",
            str(s.tmp_path / "test-eval"),
        ]
    )
    assert result["total_count"] == 48 and result["rounds_completed"] == 16
    assert result["accuracy"] == 0.5 and result["split"] == "test"
    assert s.seeds == list(range(42, 58))
    assert all("test Q" in prompt for prompt in s.prompts)
    assert path.read_bytes() == before
    s.tracker.log_evaluation.assert_called_with(result, selection=selection)


@pytest.mark.parametrize(
    "mutation",
    [
        "frozen",
        "task",
        "training",
        "split",
        "complete",
        "count",
        "accuracy",
        "candidate",
        "hash",
        "iteration",
        "protocol",
    ],
)
def test_selection_rejected_before_generation(setup, mutation):
    s = setup
    best = runner.main(s.argv)
    path, selection = freeze(s, best)
    selected = selection["best"]
    if mutation == "frozen":
        selection["frozen"] = False
    elif mutation == "task":
        selection["task"] = "math"
    elif mutation == "training":
        selection["training_output"] = "/different"
    elif mutation == "split":
        selected["split"] = "test"
    elif mutation == "complete":
        selected["complete"] = False
    elif mutation == "count":
        selected["total_count"] -= 1
    elif mutation == "accuracy":
        selected["accuracy"] = 1.0
    elif mutation == "candidate":
        selection["candidates"] = []
    elif mutation == "hash":
        s.checkpoint.write_bytes(b"changed")
    elif mutation == "iteration":
        selected["checkpoint"]["iteration"] = 2
    else:
        selected["protocol"]["top_p"] = 1.0
    runner.write_json(path, selection)
    calls = s.generator.calls
    with pytest.raises(AssertionError):
        runner.main(
            [
                "--split",
                "test",
                "--training-output",
                str(s.training),
                "--selection",
                str(path),
                "--output",
                str(s.tmp_path / "rejected"),
            ]
        )
    assert s.generator.calls == calls
    assert not (s.tmp_path / "rejected/result.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing_hash", "hash", "task", "config", "algorithm", "dtype", "lora", "prompt"],
)
def test_training_identity_rejected(setup, mutation):
    s = setup
    metadata = copy.deepcopy(s.metadata)
    if mutation == "missing_hash":
        del metadata["tokenizer_sha256"]
    elif mutation == "hash":
        metadata["tokenizer_sha256"] = "wrong"
    elif mutation == "task":
        metadata["cli"]["task"] = "unknown"
    elif mutation == "config":
        metadata["trainer_config"]["num_iterations"] = 9
    elif mutation == "algorithm":
        metadata["cli"]["algorithm"] = "rloo"
    elif mutation == "dtype":
        metadata["cli"]["dtype"] = "bfloat16"
    elif mutation == "lora":
        metadata["model"]["lora"] = {}
    else:
        metadata["prompt"] = "wrong"
    runner.write_json(s.training / "metadata.json", metadata)
    with pytest.raises((AssertionError, KeyError)):
        runner.main(s.argv)
    assert s.generator.calls == 0
    assert not (s.tmp_path / "evaluation/result.json").exists()
    assert read(s.tmp_path / "evaluation/failure.json")["complete"] is False


@pytest.mark.parametrize("devices", [0, 2])
def test_exactly_one_cuda(setup, monkeypatch, devices):
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: devices)
    with pytest.raises(AssertionError, match="exactly one"):
        runner.main(setup.argv)
    setup.loader.assert_not_called()


def test_failure_retains_successful_batch(setup):
    s = setup
    s.generator.fail_at = 2
    with pytest.raises(RuntimeError, match="generation failed"):
        runner.main(s.argv)
    output = s.tmp_path / "evaluation"
    assert s.generator.closed and not (output / "result.json").exists()
    metrics = read(output / "round-00/metrics.json")
    assert metrics["total_count"] == 2 and metrics["complete"] is False
    assert len((output / "round-00/records.jsonl").read_text().splitlines()) == 2
    assert (output / "provenance.json").exists()
    s.tracker.log_evaluation.assert_not_called()


@pytest.mark.parametrize("tamper", [False, True])
def test_resumed_test_loads_selected_actor_source_metadata(setup, tamper):
    s = setup
    best = runner.main(s.argv)
    new = s.tmp_path / "resumed"
    new.mkdir()
    metadata = copy.deepcopy(s.metadata)
    metadata["trainer_config"]["checkpoint_dir"] = str(new / "checkpoints")
    runner.write_json(new / "metadata.json", metadata)
    runner.write_json(
        new / "wandb.json", {"id": "new-run", "url": None, "mode": "offline"}
    )
    best = best | {"training_run": {"path": str(new), "wandb": None}}
    path, selection = freeze(s, best)
    selection["training_output"] = str(new)
    runner.write_json(path, selection)
    original = s.checkpoint.read_bytes()
    if tamper:
        (s.training / "metadata.json").write_text("{}")
    argv = [
        "--split",
        "test",
        "--training-output",
        str(new),
        "--selection",
        str(path),
        "--output",
        str(s.tmp_path / "resumed-test"),
    ]
    if tamper:
        with pytest.raises(AssertionError, match="source metadata changed"):
            runner.main(argv)
    else:
        result = runner.main(argv)
        assert result["checkpoint_source"]["path"] == str(s.training)
        assert result["training_run"]["wandb"]["id"] == "new-run"
        assert result["checkpoint"] == best["checkpoint"]
        assert s.checkpoint.read_bytes() == original


def test_tracker_failure_no_final_result(setup):
    setup.tracker.log_evaluation.side_effect = RuntimeError("tracking failed")
    with pytest.raises(RuntimeError, match="tracking failed"):
        runner.main(setup.argv)
    assert setup.generator.closed
    assert not (setup.tmp_path / "evaluation/result.json").exists()


def test_math_solution_and_ordered_digest(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.EXPECTED_ROWS["validation"], "math", 2)
    path = tmp_path / "math"
    rows = [
        {
            "problem": f"P{i}",
            "solution": f"Reasoning: \\boxed{{{i}}}",
            "answer": "WRONG",
        }
        for i in range(2)
    ]
    DatasetDict(
        {
            "train": Dataset.from_list([{"unused": "bad"}]),
            "test": Dataset.from_list([{"unused": "bad"}]),
            "validation": Dataset.from_list(rows),
        }
    ).save_to_disk(str(path))
    examples, info = runner.load_examples(path, "math", "validation")
    assert examples[1].reference_answer == rows[1]["solution"]
    generator = Mock()
    generator.generate.return_value = [
        [TextCompletion("\\boxed{0}", "length")],
        [TextCompletion("\\boxed{1}", "stop")],
    ]
    monkeypatch.setattr(runner, "set_seed", Mock())
    output = tmp_path / "eval"
    output.mkdir()
    result = runner.run_rounds(
        SimpleNamespace(output=output, rounds=1, seed=42, batch_size=2),
        "math",
        examples,
        generator,
        {},
    )
    assert result["accuracy"] == 1.0
    other = tmp_path / "reordered"
    DatasetDict({"validation": Dataset.from_list(rows[::-1])}).save_to_disk(str(other))
    assert (
        runner.load_examples(other, "math", "validation")[1]["examples_sha256"]
        != info["examples_sha256"]
    )
    monkeypatch.setitem(runner.EXPECTED_ROWS["validation"], "math", 3)
    with pytest.raises(AssertionError, match="size"):
        runner.load_examples(path, "math", "validation")


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--checkpoint", "x", "--selection", "y"],
        ["--checkpoint", "x", "--rounds", "0"],
        ["--checkpoint", "x", "--batch-size", "0"],
        ["--checkpoint", "x", "--seed", "-1"],
        ["--checkpoint", "x", "--limit", "1"],
    ],
)
def test_invalid_validation_cli(tmp_path, extra):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--split",
                "validation",
                "--training-output",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                *extra,
            ]
        )


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--checkpoint", "x"],
        ["--selection", "x", "--rounds", "2"],
        ["--selection", "x", "--checkpoint", "y"],
    ],
)
def test_invalid_test_cli(tmp_path, extra):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--split",
                "test",
                "--training-output",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                *extra,
            ]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "tie_rule",
        "later_tie",
        "duplicate_iteration",
        "duplicate_path",
        "protocol",
        "dataset",
    ],
)
def test_candidate_selection_hardening(setup, mutation):
    s = setup
    best = runner.main(s.argv)
    _, selection = freeze(s, best)
    other = copy.deepcopy(best)
    other["checkpoint"] |= {"path": str(s.training / "iteration-2.pt"), "iteration": 2}
    selection["candidates"] = [other, best]
    selection["candidate_iterations"] = [2, 1]
    assert runner.selected_checkpoint(selection, "gsm8k", s.training) == s.checkpoint
    if mutation == "tie_rule":
        selection["tie_rule"] = "earliest iteration"
    elif mutation == "later_tie":
        selection["best"] = other
    elif mutation == "duplicate_iteration":
        other["checkpoint"]["iteration"] = 1
    elif mutation == "duplicate_path":
        other["checkpoint"]["path"] = best["checkpoint"]["path"]
    elif mutation == "protocol":
        other["protocol"]["batch_size"] += 1
    else:
        other["dataset"]["examples_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        runner.selected_checkpoint(selection, "gsm8k", s.training)


@pytest.mark.parametrize("extra", [["--seed", "43"], ["--batch-size", "2"]])
def test_frozen_test_cli_mismatch_before_loading(setup, extra):
    s = setup
    path, _ = freeze(s, runner.main(s.argv))
    s.loader.reset_mock()
    calls = s.generator.calls
    with pytest.raises(AssertionError, match="frozen selection"):
        runner.main(
            [
                "--split",
                "test",
                "--training-output",
                str(s.training),
                "--selection",
                str(path),
                "--output",
                str(s.tmp_path / "rejected"),
                *extra,
            ]
        )
    s.loader.assert_not_called()
    assert s.generator.calls == calls
    assert not (s.tmp_path / "rejected/result.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "temperature", "rounds", "extra"])
def test_frozen_test_protocol_requires_exact_match(setup, mutation):
    s = setup
    path, selection = freeze(s, runner.main(s.argv))
    if mutation == "missing":
        del selection["test_protocol"]
    else:
        selection["test_protocol"][mutation] = 1
    runner.write_json(path, selection)
    s.loader.reset_mock()
    with pytest.raises((AssertionError, KeyError)):
        runner.main(
            [
                "--split",
                "test",
                "--training-output",
                str(s.training),
                "--selection",
                str(path),
                "--output",
                str(s.tmp_path / "rejected"),
            ]
        )
    s.loader.assert_not_called()


def test_nondefault_frozen_test_protocol(setup):
    s = setup
    path, selection = freeze(s, runner.main(s.argv), eval_seed=100, eval_batch_size=2)
    s.seeds.clear()
    result = runner.main(
        [
            "--split",
            "test",
            "--training-output",
            str(s.training),
            "--selection",
            str(path),
            "--output",
            str(s.tmp_path / "test-eval"),
            "--seed",
            "100",
            "--batch-size",
            "2",
        ]
    )
    assert result["protocol"] == selection["test_protocol"]
    assert s.seeds == list(range(100, 116))


@pytest.mark.parametrize(
    "dtype,supported", [("bfloat16", False), ("bfloat16", True), ("float32", False)]
)
def test_native_bf16_requirement(setup, monkeypatch, dtype, supported):
    s = setup
    s.info["dtype"] = s.metadata["cli"]["dtype"] = dtype
    runner.write_json(s.training / "metadata.json", s.metadata)
    support = Mock(return_value=supported)
    monkeypatch.setattr(runner.torch.cuda, "is_bf16_supported", support)
    if dtype == "bfloat16" and not supported:
        with pytest.raises(AssertionError, match="Native BF16"):
            runner.main(s.argv)
        assert s.generator.calls == 0
        assert not (s.tmp_path / "evaluation/result.json").exists()
    else:
        assert runner.main(s.argv)["complete"]
        assert (
            read(s.tmp_path / "evaluation/provenance.json")["actor"]["dtype"] == dtype
        )
    if dtype == "bfloat16":
        support.assert_called_once_with(including_emulation=False)
    else:
        support.assert_not_called()


@pytest.mark.parametrize(
    "mutation", ["missing_candidate", "reordered", "validation_protocol"]
)
def test_planned_selection_rejected_before_model_loading(setup, mutation):
    s = setup
    best = runner.main(s.argv)
    path, selection = freeze(s, best)
    if mutation == "missing_candidate":
        selection["candidate_iterations"] = [1, 2]
    elif mutation == "reordered":
        other = copy.deepcopy(best)
        other["checkpoint"] |= {
            "path": str(s.training / "iteration-2.pt"),
            "iteration": 2,
        }
        selection["candidates"].append(other)
        selection["candidate_iterations"] = [2, 1]
    else:
        selection["validation_protocol"]["batch_size"] += 1
    runner.write_json(path, selection)
    s.loader.reset_mock()
    calls = s.generator.calls
    with pytest.raises(AssertionError):
        runner.main(
            [
                "--split",
                "test",
                "--training-output",
                str(s.training),
                "--selection",
                str(path),
                "--output",
                str(s.tmp_path / "rejected"),
            ]
        )
    s.loader.assert_not_called()
    assert s.generator.calls == calls
    assert not (s.tmp_path / "rejected/result.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "valid_task",
        "dataset",
        "missing",
        "checkpoint_task",
        "checkpoint_dataset",
        "extra",
    ],
)
def test_checkpoint_evaluation_provenance_binding(setup, monkeypatch, mutation):
    s = setup
    if mutation == "valid_task":
        s.metadata["cli"]["task"] = "math"
    elif mutation == "dataset":
        s.metadata["dataset"]["normalized_sha256"] = "b" * 64
    elif mutation == "missing":
        del s.info["evaluation_provenance"]
    elif mutation == "checkpoint_task":
        s.info["evaluation_provenance"]["task"] = "math"
    elif mutation == "checkpoint_dataset":
        s.info["evaluation_provenance"]["dataset"]["split"] = "test"
    else:
        s.info["evaluation_provenance"]["extra"] = True
    runner.write_json(s.training / "metadata.json", s.metadata)
    data_loader = Mock(side_effect=AssertionError("Must not load evaluation data"))
    monkeypatch.setattr(runner, "load_examples", data_loader)
    with pytest.raises(
        (AssertionError, KeyError), match="evaluation_provenance|evaluation provenance"
    ):
        runner.main(s.argv)
    data_loader.assert_not_called()
    assert s.generator.calls == 0
    assert not (s.tmp_path / "evaluation/result.json").exists()


def test_software_and_code_provenance(setup):
    runner.main(setup.argv)
    provenance = read(setup.tmp_path / "evaluation/provenance.json")
    assert provenance["software"] == {
        name: runner.version(name) for name in ("torch", "transformers", "peft")
    }
    assert provenance["cuda"] == runner.torch.version.cuda
    assert provenance["attention"] == "sdpa"
    assert provenance["evaluation_code_sha256"] == runner.file_sha256(
        Path(runner.__file__)
    )
    assert (
        provenance["actor"]["evaluation_provenance"]
        == setup.info["evaluation_provenance"]
    )
    assert provenance["checkpoint"]["sha256"] == runner.file_sha256(setup.checkpoint)


def test_defaults(tmp_path):
    args = runner.parse_args(
        [
            "--split",
            "validation",
            "--training-output",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--checkpoint",
            "x",
        ]
    )
    assert (args.rounds, args.seed, args.batch_size, args.wandb_mode) == (
        16,
        42,
        4,
        "disabled",
    )
