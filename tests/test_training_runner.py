"""Offline runner contracts: real local datasets, mocked models and CUDA clocks."""

import importlib.util
import io
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from datasets import Dataset, DatasetDict

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/training/run.py"
spec = importlib.util.spec_from_file_location("training_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network/model/CUDA access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    for loader in (runner.AutoTokenizer, runner.AutoModel, runner.AutoModelForCausalLM):
        monkeypatch.setattr(loader, "from_pretrained", forbidden)


@pytest.fixture
def cli(tmp_path):
    for name in ("model", "dataset"):
        (tmp_path / name).mkdir()
    return [
        "--mode",
        "benchmark",
        "--algorithm",
        "vineppo",
        "--task",
        "gsm8k",
        "--model-path",
        str(tmp_path / "model"),
        "--dataset-path",
        str(tmp_path / "dataset"),
        "--output",
        str(tmp_path / "output"),
    ]


@pytest.mark.parametrize("task", ["gsm8k", "math"])
def test_local_train_only_normalization_and_identity(tmp_path, task):
    question, answer = (
        ("question", "answer") if task == "gsm8k" else ("problem", "solution")
    )
    references = (
        ["reasoning#### 7", "work#### 8"]
        if task == "gsm8k"
        else [r"Work: \boxed{7} or \boxed{8}", r"\boxed{8}"]
    )
    train = Dataset.from_list(
        [{question: f" Q{i} ", answer: ref} for i, ref in enumerate(references)]
    )
    # Deliberately invalid held-out data must never be parsed.
    dataset = DatasetDict(
        {
            "train": train,
            "test": Dataset.from_list([{question: None, answer: "TEST LEAK"}]),
        }
    )
    path = tmp_path / "data"
    dataset.save_to_disk(str(path))
    examples, info = runner.load_examples(path, task)
    expected_refs = ["7", "8"] if task == "gsm8k" else references
    assert examples == [
        runner.Example(f"{task}:train:{i}", f"Q{i}", ref)
        for i, ref in enumerate(expected_refs)
    ]
    assert info["split"] == "train" and info["rows"] == 2
    assert info["fingerprint"] == runner.load_from_disk(str(path))["train"]._fingerprint
    assert len(info["normalized_sha256"]) == 64
    assert runner.load_examples(path, task) == (examples, info)
    assert (
        runner.paper_prompt(examples[0].question)
        == "[MATH_TASK] Problem:\nQ0\n\nSolution:"
    )
    dataset["test"] = Dataset.from_list(
        [{question: "changed test", answer: "still invalid"}]
    )
    dataset.save_to_disk(str(tmp_path / "changed-test"))
    assert (
        runner.load_examples(tmp_path / "changed-test", task)[1]["normalized_sha256"]
        == info["normalized_sha256"]
    )
    dataset["train"] = Dataset.from_list(
        [{question: "changed train", answer: references[0]}]
    )
    dataset.save_to_disk(str(tmp_path / "changed-train"))
    assert (
        runner.load_examples(tmp_path / "changed-train", task)[1]["normalized_sha256"]
        != info["normalized_sha256"]
    )


@pytest.mark.parametrize(
    "task,row,message",
    [
        ("gsm8k", {"question": "Q", "answer": "no delimiter"}, "Malformed GSM8K"),
        ("gsm8k", {"question": "Q", "answer": "a####1####2"}, "Malformed GSM8K"),
        ("gsm8k", {"question": "Q", "answer": "a#### "}, "Empty final"),
        ("gsm8k", {"question": " ", "answer": "a####1"}, "Empty question"),
        ("gsm8k", {"question": None, "answer": "a####1"}, "Empty question"),
        ("math", {"problem": "Q", "solution": " "}, "Empty reference"),
        ("math", {"problem": "Q", "solution": None}, "Empty reference"),
        ("math", {"problem": 42, "solution": "answer"}, "Empty question"),
    ],
)
def test_malformed_local_rows(tmp_path, task, row, message):
    DatasetDict({"train": Dataset.from_list([row])}).save_to_disk(str(tmp_path))
    with pytest.raises(AssertionError, match=message):
        runner.load_examples(tmp_path, task)


@pytest.mark.parametrize(
    "kind,error,message",
    [
        ("bare", AssertionError, "DatasetDict"),
        ("test-only", KeyError, "train"),
        ("empty", AssertionError, "Empty train"),
        ("missing-column", KeyError, "answer"),
    ],
)
def test_invalid_dataset_structure(tmp_path, kind, error, message):
    rows = Dataset.from_dict({"question": ["Q"], "answer": ["work####1"]})
    data = {
        "bare": rows,
        "test-only": DatasetDict({"test": rows}),
        "empty": DatasetDict({"train": rows.select([])}),
        "missing-column": DatasetDict({"train": rows.remove_columns("answer")}),
    }[kind]
    # Explicit shards keep empty splits loadable in datasets 5.x.
    shards = {name: 1 for name in data} if isinstance(data, DatasetDict) else 1
    data.save_to_disk(str(tmp_path), num_shards=shards)
    with pytest.raises(error, match=message):
        runner.load_examples(tmp_path, "gsm8k")


@pytest.mark.parametrize(
    "option",
    [
        "num-iterations",
        "prompts-per-iteration",
        "num-samples",
        "mc-num-samples",
        "update-epochs",
        "minibatch-size",
        "microbatch-size",
        "max-sequence-length",
        "max-new-tokens",
        "keep-checkpoints",
        "checkpoint-every",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_cli_counts(cli, option, value):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(cli + [f"--{option}", value])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "option,value",
    [
        ("learning-rate", "nan"),
        ("learning-rate", "inf"),
        ("learning-rate", "0"),
        ("critic-learning-rate", "-1"),
        ("critic-learning-rate", "-inf"),
        ("warmup-iterations", "-1"),
        ("warmup-iterations", "3"),
        ("seed", "-1"),
        ("seed", str(2**64)),
        ("algorithm", "unknown"),
        ("finetuning", "unknown"),
    ],
)
def test_invalid_cli_ranges(cli, option, value):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(cli + [f"--{option}", value])
    assert exc.value.code == 2


@pytest.mark.parametrize("algorithm", ["vineppo", "ppo", "grpo", "rloo"])
@pytest.mark.parametrize("num_samples", [1, 2, 4])
def test_cli_algorithm_samples(cli, capsys, algorithm, num_samples):
    options = ["--algorithm", algorithm, "--num-samples", str(num_samples)]
    if algorithm in ("grpo", "rloo") and num_samples == 1:
        for sample_options in ([], options[2:]):
            with pytest.raises(SystemExit) as exc:
                runner.parse_args(cli + options[:2] + sample_options)
            assert exc.value.code == 2
            assert "require --num-samples >= 2" in capsys.readouterr().err
    else:
        args = runner.parse_args(cli + options)
        assert (args.algorithm, args.num_samples) == (algorithm, num_samples)


@pytest.mark.parametrize("finetuning", ["full", "lora"])
@pytest.mark.parametrize("learning_rate", [None, "3e-7"])
def test_cli_finetuning_defaults_and_lr_override(cli, finetuning, learning_rate):
    options = ["--finetuning", finetuning]
    if learning_rate is not None:
        options += ["--learning-rate", learning_rate]
    args = runner.parse_args(cli + options)
    assert args.finetuning == finetuning
    assert (args.lora_rank, args.lora_alpha) == (
        (16, 32.0) if finetuning == "lora" else (None, None)
    )
    assert args.learning_rate == (
        float(learning_rate)
        if learning_rate is not None
        else 1e-5
        if finetuning == "lora"
        else 1e-6
    )


@pytest.mark.parametrize("option,value", [("lora-rank", "8"), ("lora-alpha", "12.5")])
def test_cli_explicit_lora_options_require_lora(cli, capsys, option, value):
    for mode in ([], ["--finetuning", "full"]):
        with pytest.raises(SystemExit) as exc:
            runner.parse_args(cli + mode + [f"--{option}", value])
        assert exc.value.code == 2
        assert "LoRA options require" in capsys.readouterr().err
    args = runner.parse_args(cli + ["--finetuning", "lora", f"--{option}", value])
    assert (args.lora_rank, args.lora_alpha) == (
        (8, 32.0) if option == "lora-rank" else (16, 12.5)
    )


@pytest.mark.parametrize(
    "option,values",
    [
        ("lora-rank", ["0", "-1", "1.5"]),
        ("lora-alpha", ["0", "-1", "nan", "inf", "-inf"]),
    ],
)
def test_cli_invalid_lora_values(cli, option, values):
    for value in values:
        with pytest.raises(SystemExit) as exc:
            runner.parse_args(cli + ["--finetuning", "lora", f"--{option}={value}"])
        assert exc.value.code == 2


def test_cli_defaults_collision_and_resume(cli, tmp_path):
    args = runner.parse_args(cli + ["--seed", str(2**64 - 1)])
    assert (args.num_iterations, args.warmup_iterations, args.dtype) == (
        3,
        1,
        "float32",
    )
    assert (args.finetuning, args.lora_rank, args.lora_alpha, args.learning_rate) == (
        "full",
        None,
        None,
        1e-6,
    )
    assert args.checkpoint_every is None
    assert args.model_path == (tmp_path / "model").resolve()
    resume = tmp_path / "iteration-9.pt"
    resume.write_bytes(b"original")
    assert runner.parse_args(cli + ["--resume", str(resume)]).resume == resume.resolve()
    args.output.mkdir()
    with pytest.raises(AssertionError, match="new directory"):
        runner.parse_args(cli + ["--resume", str(resume)])
    assert resume.read_bytes() == b"original"


@pytest.mark.parametrize(
    "option,kind,error",
    [
        ("model-path", "missing", FileNotFoundError),
        ("dataset-path", "missing", FileNotFoundError),
        ("model-path", "file", AssertionError),
        ("dataset-path", "file", AssertionError),
        ("resume", "directory", AssertionError),
    ],
)
def test_cli_requires_local_paths(cli, tmp_path, option, kind, error):
    path = tmp_path / "invalid"
    if kind == "file":
        path.write_text("not a directory")
    if kind == "directory":
        path.mkdir()
    with pytest.raises(error):
        runner.parse_args(cli + [f"--{option}", str(path)])


def test_main_requires_cuda_without_creating_output(cli, tmp_path):
    with pytest.raises(AssertionError, match="CUDA required"):
        runner.main(cli)
    assert not (tmp_path / "output").exists()


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sync=Mock(), reset=Mock())
    monkeypatch.setattr(runner, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(torch.cuda, "synchronize", clock.sync)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", clock.reset)
    for name, value in (
        ("max_memory_allocated", 100),
        ("max_memory_reserved", 200),
        ("memory_allocated", 30),
        ("memory_reserved", 40),
    ):
        monkeypatch.setattr(torch.cuda, name, lambda value=value: value)
    return clock


def test_meter_nested_exception_and_reset(clock):
    meter = runner.Meter()
    meter.start()
    with meter.phase("outer"):
        clock.now = 1
        with pytest.raises(ValueError), meter.phase("inner"):
            clock.now = 3
            raise ValueError("generation failed")
        assert meter.active == "outer"
        clock.now = 5
    assert meter.active is None
    assert dict(meter.seconds) == {"inner": 2, "outer": 5}
    assert clock.sync.call_count == 5
    meter.counts["primary_tokens"] = 9
    meter.start()
    assert not meter.seconds and not meter.counts
    assert meter.started == 5 and clock.reset.call_count == 2


@pytest.mark.parametrize(
    "phase,kind", [("primary_generation", "primary"), ("mc_estimation", "mc")]
)
def test_inference_counts_all_continuations_and_eos(monkeypatch, clock, phase, kind):
    inference = object.__new__(runner.MeasuredInference)
    inference.meter = runner.Meter()
    result = [
        [SimpleNamespace(token_ids=[7, 2]), SimpleNamespace(token_ids=[8])],
        [SimpleNamespace(token_ids=[])],
    ]

    def generate(self, prefixes, **kwargs):
        assert prefixes == [[1], [1, 3]] and kwargs == {"num_samples": 2}
        clock.now += 4
        return result

    monkeypatch.setattr(runner.HFInference, "generate", generate)
    with inference.meter.phase(phase):
        assert inference.generate([[1], [1, 3]], num_samples=2) is result
        assert inference.meter.active == phase
    assert inference.meter.counts[f"{kind}_tokens"] == 3
    assert inference.meter.counts[f"{kind}_continuations"] == 3
    assert inference.meter.counts["mc_expansions"] == (2 if kind == "mc" else 0)
    assert inference.meter.seconds[f"{kind}_generate_call"] == 4
    with pytest.raises(AssertionError):
        inference.generate([[1]])


def measured_trainer():
    trainer = object.__new__(runner.MeasuredTrainer)
    trainer.meter = runner.Meter()
    trainer.stream = io.StringIO()
    trainer.warmup_iterations = 1
    trainer.keep_checkpoints = 2
    trainer.rows, trainer.saved = [], []
    trainer.pending = None
    trainer.tracker = None
    trainer.selection = None
    return trainer


def test_retention_only_prunes_owned_files_after_success(monkeypatch, tmp_path, clock):
    trainer = measured_trainer()
    resume = tmp_path / "resume" / "iteration-0.pt"
    resume.parent.mkdir()
    resume.write_bytes(b"resume")
    output = tmp_path / "new"
    output.mkdir()
    unrelated = output / "iteration-99.pt"
    unrelated.write_bytes(b"not owned")
    created = []

    def save(self):
        assert all(path.exists() for path in self.saved)
        path = output / f"iteration-{len(created) + 1}.pt"
        path.write_bytes(b"checkpoint")
        created.append(path)
        return path

    monkeypatch.setattr(runner.PPOTrainer, "save_checkpoint", save)
    trainer.meter.start()
    for _ in range(3):
        trainer.pending = {"loss": 0.25}
        clock.now += 1
        assert trainer.save_checkpoint() == created[-1]
    assert trainer.saved == created[1:]
    assert not created[0].exists() and all(path.exists() for path in created[1:])
    assert all(row["checkpoint_bytes"] == 10 for row in trainer.rows)
    assert resume.read_bytes() == b"resume" and unrelated.read_bytes() == b"not owned"
    monkeypatch.setattr(
        runner.PPOTrainer, "save_checkpoint", Mock(side_effect=OSError("disk full"))
    )
    trainer.pending = {"loss": 1}
    with pytest.raises(OSError, match="disk full"):
        trainer.save_checkpoint()
    assert trainer.saved == created[1:] and all(path.exists() for path in trainer.saved)
    assert trainer.pending == {"loss": 1}
    with pytest.raises(AssertionError, match="this run"):
        trainer.save_checkpoint(resume)


@pytest.mark.parametrize(
    "algorithm,task",
    [("vineppo", "gsm8k"), ("ppo", "math"), ("grpo", "gsm8k"), ("rloo", "math")],
)
@pytest.mark.parametrize(
    "finetuning,dtype", [("full", "float32"), ("lora", "bfloat16")]
)
@pytest.mark.parametrize("tracking_mode", ["disabled", "offline"])
def test_main_local_model_assembly_and_real_warmup_summary(
    cli, tmp_path, monkeypatch, clock, algorithm, task, finetuning, dtype, tracking_mode
):
    tracked_run = SimpleNamespace(
        id="fake-run",
        url=None,
        summary={},
        define_metric=Mock(),
        log=Mock(),
        finish=Mock(),
    )
    sdk = SimpleNamespace(
        init=Mock(return_value=tracked_run), Settings=lambda **kwargs: kwargs
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    model_path = tmp_path / "model"
    (model_path / "config.json").write_text("{}")
    row = (
        {"question": "Q", "answer": "work####2"}
        if task == "gsm8k"
        else {"problem": "Q", "solution": r"\boxed{2}"}
    )
    DatasetDict({"train": Dataset.from_list([row, row])}).save_to_disk(
        str(tmp_path / "dataset")
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="fake A40", total_memory=1000, major=8, minor=6),
    )
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    bf16_supported = Mock(return_value=True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", bf16_supported)
    monkeypatch.setattr(torch, "manual_seed", Mock())
    monkeypatch.setattr(runner, "git", lambda *args: "test-commit")
    tokenizer = SimpleNamespace(
        is_fast=True,
        eos_token_id=2,
        eos_token="eos",
        pad_token_id=None,
        backend_tokenizer=SimpleNamespace(to_str=lambda: "{}"),
    )
    load_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", load_tokenizer)

    def model():
        value = Mock()
        value.to.return_value = value
        value.requires_grad_.return_value = value
        value.config.to_dict.return_value = {"hidden_size": 1}
        value.parameters.return_value = [
            torch.nn.Parameter(torch.ones(5, dtype=getattr(torch, dtype)))
        ]
        return value

    actor, reference, backbone, critic = [model() for _ in range(4)]

    def freeze_reference(enabled):
        reference.parameters()[0].requires_grad_(enabled)
        return reference

    reference.requires_grad_.side_effect = freeze_reference
    wrapped = model()
    adapters = [torch.nn.Parameter(torch.ones(n)) for n in (2, 3)]

    def wrap(base, **kwargs):
        assert base is actor
        base.parameters()[0].requires_grad_(False)
        wrapped.parameters.return_value = [*base.parameters(), *adapters]
        return wrapped

    apply_lora = Mock(side_effect=wrap)
    identity = {"rank": 8, "alpha": 12.5, "target_modules": ["q_proj", "v_proj"]}
    lora_identity = Mock(return_value=identity)
    monkeypatch.setattr(runner, "apply_lora", apply_lora)
    monkeypatch.setattr(runner, "lora_identity", lora_identity)
    load_actor = Mock(side_effect=[actor, reference])
    load_backbone = Mock(return_value=backbone)
    make_critic = Mock(return_value=critic)
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", load_actor)
    monkeypatch.setattr(runner.AutoModel, "from_pretrained", load_backbone)
    monkeypatch.setattr(runner, "TokenCritic", make_critic)
    optimizer = Mock()
    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    inference = Mock()
    inference.__enter__ = Mock(return_value=inference)
    inference.__exit__ = Mock(return_value=False)
    make_inference = Mock(return_value=inference)
    monkeypatch.setattr(runner, "MeasuredInference", make_inference)
    trainer = measured_trainer()

    def assemble(*args, **kwargs):
        trainer.meter = kwargs["meter"]
        trainer.stream = kwargs["stream"]
        trainer.warmup_iterations = kwargs["warmup_iterations"]
        trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-5}])
        trainer.critic_optimizer = None
        return trainer

    assembly = Mock(side_effect=assemble)
    monkeypatch.setattr(runner, "MeasuredTrainer", assembly)
    trainer.iteration = 0

    def train(**inputs):
        assert inputs["task"] == task
        assert inputs["reference_is_solution"] is (task == "math")
        assert [e.id for e in inputs["examples"]] == [
            f"{task}:train:0",
            f"{task}:train:1",
        ]
        trainer.meter.start()
        # Real finish_iteration and summary code; unequal durations detect mean-of-rates bugs.
        for elapsed, primary, mc in [(100, 900, 100), (2, 6, 4), (4, 18, 12)]:
            trainer.iteration += 1
            trainer.pending = {"iteration": trainer.iteration, "loss": 0.5}
            uses_mc = algorithm == "vineppo"
            trainer.meter.seconds.update(
                primary_generation=0.5,
                mc_estimation=1 if uses_mc else 0,
                mc_generate_call=0.75 if uses_mc else 0,
            )
            trainer.meter.counts.update(
                primary_tokens=primary,
                mc_tokens=mc if uses_mc else 0,
                primary_continuations=2,
                mc_continuations=4 if uses_mc else 0,
                mc_expansions=2 if uses_mc else 0,
            )
            clock.now += elapsed
            trainer.finish_iteration()

    trainer.train = train
    options = [
        "--algorithm",
        algorithm,
        "--task",
        task,
        "--num-samples",
        "2",
        "--finetuning",
        finetuning,
        "--dtype",
        dtype,
    ]
    if finetuning == "lora":
        options += ["--lora-rank", "8", "--lora-alpha", "12.5"]
    runner.main(cli + options + ["--wandb-mode", tracking_mode])
    if tracking_mode == "offline":
        sdk.init.assert_called_once()
        tracked_run.finish.assert_called_once_with(exit_code=0)
        assert tracked_run.log.call_count == 3
        assert [c.kwargs["step"] for c in tracked_run.log.call_args_list] == [1, 2, 3]
        assert tracked_run.summary["iterations_measured"] == 2
        assert (
            json.loads((tmp_path / "output" / "wandb.json").read_text())["id"]
            == "fake-run"
        )
        assert tracked_run.log.call_args.args[0]["budget/primary_tokens"] == 924
    else:
        sdk.init.assert_not_called()
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": getattr(torch, dtype),
        "attn_implementation": "sdpa",
        "use_safetensors": True,
    }
    assert load_actor.call_args_list == [call(str(model_path.resolve()), **kwargs)] * 2
    load_tokenizer.assert_called_once_with(
        str(model_path.resolve()),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    assert tokenizer.pad_token == tokenizer.eos_token
    actor.to.assert_called_once_with("cuda:0")
    reference.requires_grad_.assert_called_once_with(False)
    reference.eval.assert_called_once_with()
    policy = wrapped if finetuning == "lora" else actor
    assert assembly.call_args.args[1] is policy
    assert make_inference.call_args.args[0] is policy
    assert assembly.call_args.kwargs["reference_policy"] is reference
    assert not reference.parameters()[0].requires_grad
    expected_params = adapters if finetuning == "lora" else actor.parameters()
    actual_params = list(optimizer.call_args_list[0].args[0])
    assert [id(p) for p in actual_params] == [id(p) for p in expected_params]
    assert optimizer.call_args_list[0].kwargs == {
        "lr": 1e-5 if finetuning == "lora" else 1e-6,
        "weight_decay": 0.0,
    }
    if finetuning == "lora":
        apply_lora.assert_called_once_with(actor, rank=8, alpha=12.5)
        lora_identity.assert_called_once_with(wrapped)
        bf16_supported.assert_called_once_with(including_emulation=False)
    else:
        apply_lora.assert_not_called()
        lora_identity.assert_not_called()
        bf16_supported.assert_not_called()
    metadata = json.loads((tmp_path / "output/metadata.json").read_text())
    assert metadata["cli"]["finetuning"] == finetuning
    assert metadata["model"]["parameters"] == (10 if finetuning == "lora" else 5)
    assert metadata["model"]["trainable_parameters"] == 5
    assert metadata["model"]["trainable_dtypes"] == ["torch.float32"]
    assert metadata["model"]["lora"] == (identity if finetuning == "lora" else None)
    assert metadata["software"]["peft"] == runner.version("peft")
    assert metadata["precision"].startswith(
        "FP32 actor adapters" if finetuning == "lora" else "Native dtype"
    )
    config = assembly.call_args.args[0]
    assert config.checkpoint_every == 3
    assert isinstance(
        config.algorithm_config,
        {
            "vineppo": runner.VinePPOConfig,
            "ppo": runner.PPOConfig,
            "grpo": runner.GRPOConfig,
            "rloo": runner.RLOOConfig,
        }[algorithm],
    )
    assert assembly.call_args.kwargs["warmup_iterations"] == 1
    if algorithm == "ppo":
        load_backbone.assert_called_once_with(str(model_path.resolve()), **kwargs)
        make_critic.assert_called_once_with(backbone)
        assert assembly.call_args.kwargs["critic"] is critic
        assert optimizer.call_count == 2
        assert all(p.requires_grad for p in critic.parameters())
        assert [id(p) for p in optimizer.call_args_list[1].args[0]] == [
            id(p) for p in critic.parameters()
        ]
        assert optimizer.call_args_list[1].kwargs == {"lr": 1e-5, "weight_decay": 0.0}
    else:
        load_backbone.assert_not_called()
        make_critic.assert_not_called()
        assert assembly.call_args.kwargs["critic"] is None
        assert assembly.call_args.kwargs["critic_optimizer"] is None
        assert optimizer.call_count == 1
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert (
        summary["iterations_executed"],
        summary["iterations_measured"],
        summary["warmup_iterations_excluded"],
    ) == (3, 2, 1)
    assert summary["iteration_seconds_total"] == 6
    assert summary["iteration_seconds_mean"] == 3
    uses_mc = algorithm == "vineppo"
    tokens = 40 if uses_mc else 24
    assert summary["sampled_tokens"] == tokens
    assert summary["sampled_tokens_per_second"] == pytest.approx(tokens / 6)
    assert summary["phase_seconds_total"]["mc_estimation"] == (2 if uses_mc else 0)
    assert summary["sampling_totals"]["mc_tokens"] == (16 if uses_mc else 0)
    assert summary["sampling_totals"]["mc_continuations"] == (8 if uses_mc else 0)
    assert summary["sampling_totals"]["mc_expansions"] == (4 if uses_mc else 0)
    assert summary["peak_allocated_bytes"] == 100
    assert summary["peak_reserved_bytes"] == 200
    rows = [
        json.loads(line)
        for line in (tmp_path / "output/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["warmup"] for row in rows] == [True, False, False]
    assert rows[1]["other_seconds"] == (
        0.5 if uses_mc else 1.5
    )  # Nested MC generation is not double-counted.
    assert rows[1]["mc_generation_seconds"] == (0.75 if uses_mc else 0)
    assert trainer.pending is None
    inference.__exit__.assert_called_once()


def test_iteration_is_logged_after_local_write_with_learning_rates(clock):
    trainer = measured_trainer()
    tracker = Mock()
    trainer.tracker = tracker
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-5}])
    trainer.critic_optimizer = SimpleNamespace(param_groups=[{"lr": 2e-5}])
    trainer.pending = {"iteration": 7, "loss": 0.5}
    trainer.meter.start()
    clock.now += 2

    def capture(row):
        assert json.loads(trainer.stream.getvalue()) == row
        assert row["actor_learning_rate"] == 1e-5
        assert row["critic_learning_rate"] == 2e-5

    tracker.log_iteration.side_effect = capture
    trainer.finish_iteration()
    tracker.log_iteration.assert_called_once_with(trainer.rows[0])


def test_tracking_failure_propagates_after_preserving_local_row(clock):
    trainer = measured_trainer()
    trainer.tracker = Mock()
    trainer.tracker.log_iteration.side_effect = RuntimeError("logging failed")
    trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-5}])
    trainer.critic_optimizer = None
    trainer.pending = {"iteration": 1, "loss": 0.5}
    trainer.meter.start()
    clock.now += 1
    with pytest.raises(RuntimeError, match="logging failed"):
        trainer.finish_iteration()
    assert json.loads(trainer.stream.getvalue())["iteration"] == 1


def test_selection_precedes_pruning_and_protects_old_best(monkeypatch, tmp_path, clock):
    trainer = measured_trainer()
    selection = SimpleNamespace(best_path=None)
    trainer.selection = selection
    created = []

    def save(self, path=None):
        saved = tmp_path / f"iteration-{self.iteration}.pt"
        saved.write_bytes(b"checkpoint")
        created.append(saved)
        return saved

    def validate(saved, iteration):
        assert saved.exists()
        if iteration == 3:
            assert created[0].exists(), "Best was pruned before validation"
        if iteration in (1, 4):
            selection.best_path = saved
        return {"eval/validation/pass_at_1": iteration / 10}

    selection.validate = validate
    monkeypatch.setattr(runner.PPOTrainer, "save_checkpoint", save)
    trainer.meter.start()
    for iteration in range(1, 5):
        trainer.iteration = iteration
        trainer.pending = {"iteration": iteration, "loss": 0.25}
        clock.now += 1
        trainer.save_checkpoint()
        if iteration == 3:
            assert trainer.saved == created and len(trainer.saved) == 3
    assert trainer.saved == created[-2:]
    assert not created[0].exists() and not created[1].exists()
    assert trainer.rows[-1]["eval/validation/pass_at_1"] == 0.4
    assert "validation_seconds" in trainer.rows[-1]
    selection.validate = Mock(side_effect=RuntimeError("validation failed"))
    trainer.iteration = 5
    trainer.pending = {"iteration": 5}
    with pytest.raises(RuntimeError, match="validation failed"):
        trainer.save_checkpoint()
    assert all(path.exists() for path in created[-3:])
    assert len(trainer.rows) == 4


@pytest.mark.parametrize(
    "extra",
    [
        ["--evaluation", "validation"],
        [
            "--mode",
            "train",
            "--evaluation",
            "validation-test",
            "--resume",
            "CHECKPOINT",
        ],
        ["--eval-seed", "-1"],
        ["--validation-rounds", "0"],
        ["--eval-timeout-seconds", "0"],
        ["--eval-batch-size", "0"],
    ],
)
def test_invalid_automatic_selection_cli(cli, extra):
    with pytest.raises(SystemExit):
        runner.parse_args(cli + extra)


def test_selection_resume_cli(cli, tmp_path):
    checkpoint, manifest = tmp_path / "iteration-3.pt", tmp_path / "selection.json"
    checkpoint.touch()
    manifest.write_text("{}")
    args = runner.parse_args(
        cli
        + [
            "--mode",
            "train",
            "--evaluation",
            "validation",
            "--resume",
            str(checkpoint),
            "--resume-selection",
            str(manifest),
        ]
    )
    assert args.resume_selection == manifest and args.resume == checkpoint
    with pytest.raises(SystemExit):
        runner.parse_args(cli + ["--resume-selection", str(manifest)])


@pytest.mark.parametrize("start", [2, 3])
def test_main_recovers_validation_before_updates_or_finalization(
    cli, tmp_path, monkeypatch, clock, start
):
    checkpoint, manifest = tmp_path / "iteration.pt", tmp_path / "selection.json"
    checkpoint.touch()
    manifest.write_text("{}")
    (tmp_path / "model/config.json").write_text("{}")
    monkeypatch.setattr(
        runner, "load_examples", lambda *a: ([runner.Example("a", "Q", "1")] * 2, {})
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="fake", total_memory=1, major=8, minor=6),
    )
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    monkeypatch.setattr(torch, "manual_seed", Mock())
    monkeypatch.setattr(runner, "git", lambda *a: "test")
    tokenizer = SimpleNamespace(
        is_fast=True,
        eos_token_id=2,
        pad_token_id=2,
        backend_tokenizer=SimpleNamespace(to_str=lambda: "{}"),
    )
    monkeypatch.setattr(
        runner.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer)
    )
    actor = Mock()
    actor.to.return_value = actor
    actor.parameters.return_value = [torch.nn.Parameter(torch.ones(1))]
    actor.config.to_dict.return_value = {}
    monkeypatch.setattr(
        runner.AutoModelForCausalLM, "from_pretrained", Mock(return_value=actor)
    )
    inference = Mock(__enter__=Mock(), __exit__=Mock(return_value=False))
    monkeypatch.setattr(runner, "MeasuredInference", Mock(return_value=inference))
    trainer = measured_trainer()
    events = []

    def assemble(*a, **kw):
        trainer.stream, trainer.meter = kw["stream"], kw["meter"]
        trainer.warmup_iterations = kw["warmup_iterations"]
        trainer.optimizer = kw["optimizer"]
        return trainer

    def load(*a, **kw):
        trainer.iteration = start

    def train(**kw):
        assert events == ["recover"]
        events.append("train")
        trainer.iteration = 3
        trainer.pending = {"iteration": 3}
        trainer.meter.start()
        clock.now += 1
        trainer.finish_iteration()

    trainer.load_checkpoint = load
    trainer.train = train
    monkeypatch.setattr(runner, "MeasuredTrainer", assemble)
    controller = Mock()
    controller.recover.side_effect = lambda i: events.append("recover")
    controller.freeze.side_effect = lambda: events.append("freeze") or {}
    monkeypatch.setattr(runner, "CheckpointSelection", Mock(return_value=controller))
    tracker = Mock()
    tracker.__enter__ = Mock(return_value=tracker)
    tracker.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(runner, "WandBTracker", Mock(return_value=tracker))
    runner.main(
        cli
        + [
            "--mode",
            "train",
            "--evaluation",
            "validation",
            "--resume",
            str(checkpoint),
            "--resume-selection",
            str(manifest),
        ]
    )
    assert events == (
        ["recover", "train", "freeze"] if start == 2 else ["recover", "freeze"]
    )
    tracker.log_iteration.assert_not_called()
    tracker.log_selection.assert_called_once()
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["iterations_executed"] == (1 if start == 2 else 0)
    assert (summary["iteration_seconds_mean"] is None) == (start == 3)
