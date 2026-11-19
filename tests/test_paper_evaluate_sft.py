"""CPU-only runner tests; no model downloads or CUDA jobs."""

import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset, DatasetDict

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/paper_artifact_replications/evaluate_sft.py"
)
spec = importlib.util.spec_from_file_location("paper_evaluate_sft", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
load_generator = runner.load_generator


@pytest.fixture(autouse=True)
def no_network_or_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network/model/GPU access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "load_generator", forbidden)


def read_json(path):
    return json.loads(path.read_text())


def test_cli_defaults_and_smoke(tmp_path):
    config = runner.parse_args(["--task", "gsm8k", "--output", str(tmp_path)])
    assert (config.rounds, config.seed, config.batch_size) == (16, 42, 8)
    with pytest.raises(AssertionError, match="requires --smoke"):
        runner.parse_args(["--task", "math", "--output", str(tmp_path), "--limit", "2"])
    smoke = runner.parse_args(
        [
            "--task",
            "math",
            "--output",
            str(tmp_path),
            "--limit",
            "2",
            "--smoke",
            "--expected-gpu",
            "A40",
        ]
    )
    assert smoke.smoke and smoke.limit == 2 and smoke.expected_gpu == "A40"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rounds": 0},
        {"batch_size": -1},
        {"limit": 0, "smoke": True},
        {"seed": -1},
        {"seed": 2**32},
    ],
)
def test_invalid_config(tmp_path, kwargs):
    with pytest.raises(AssertionError):
        runner.Config("math", tmp_path, **kwargs)


def test_cuda_and_gpu_name(monkeypatch):
    with pytest.raises(AssertionError, match="CUDA is required"):
        runner.require_cuda(None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA A40")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert runner.require_cuda("a40") == "NVIDIA A40"
    with pytest.raises(AssertionError, match="Expected GPU"):
        runner.require_cuda("H100")


@pytest.fixture
def manifest(tmp_path):
    model = tmp_path / "model"
    dataset = tmp_path / "dataset"
    model.mkdir()
    dataset.mkdir()
    data = {
        "schema_version": 1,
        "upstream_commit": runner.UPSTREAM_COMMIT,
        "artifacts": [
            {
                "kind": "model",
                "task": "gsm8k",
                "path": str(model),
                "repo_id": runner.MODELS["gsm8k"][0],
                "revision": runner.MODELS["gsm8k"][1],
            },
            {
                "kind": "dataset",
                "task": "gsm8k",
                "path": str(dataset),
                "sha256": runner.DATASET_SHA256,
                "splits": {"test": {"rows": 1319}},
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data


def test_manifest_pins_and_paths(manifest):
    path, data = manifest
    model, dataset = runner.read_artifacts(path, "gsm8k")
    assert model["revision"] == runner.MODELS["gsm8k"][1]
    assert dataset["sha256"] == runner.DATASET_SHA256
    data["artifacts"][0]["revision"] = "main"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="Unpinned"):
        runner.read_artifacts(path, "gsm8k")


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "path", "checksum", "rows"]
)
def test_bad_manifest(manifest, mutation):
    path, data = manifest
    if mutation == "missing":
        data["artifacts"].pop()
    elif mutation == "duplicate":
        data["artifacts"].append(data["artifacts"][0])
    elif mutation == "path":
        data["artifacts"][0]["path"] = "not-a-local-absolute-directory"
    elif mutation == "checksum":
        data["artifacts"][1]["sha256"] = "bad"
    else:
        data["artifacts"][1]["splits"]["test"]["rows"] = 500
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        runner.read_artifacts(path, "gsm8k")


def saved_test(tmp_path, rows):
    DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(str(tmp_path))
    return tmp_path


def test_local_gsm8k_selection(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.TEST_ROWS, "gsm8k", 3)
    path = saved_test(
        tmp_path,
        [
            {"question": f" Q{i} ", "answer": f"secret reasoning\n#### {i}"}
            for i in range(3)
        ],
    )
    examples = runner.load_examples(path, "gsm8k", 2)
    assert examples == [
        runner.Example("gsm8k:test:0", "Q0", "0"),
        runner.Example("gsm8k:test:1", "Q1", "1"),
    ]
    assert (
        runner.paper_prompt(examples[0].question)
        == "[MATH_TASK] Problem:\nQ0\n\nSolution:"
    )
    assert "secret" not in runner.paper_prompt(examples[0].question)
    assert runner.load_examples(path, "gsm8k", None)[:2] == examples
    with pytest.raises(AssertionError, match="exceeds"):
        runner.load_examples(path, "gsm8k", 4)


def test_math_full_solution_including_multiple_boxes(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.TEST_ROWS, "math", 2)
    path = saved_test(
        tmp_path,
        [
            {
                "problem": "Compute a fraction.",
                "solution": r"\boxed{15\frac{39}{40}} or \boxed{\frac{639}{40}}",
                "answer": r"\frac{639}{40}",
            },
            {"problem": "Compute.", "solution": r"\boxed{2}", "answer": "2"},
        ],
    )
    examples = runner.load_examples(path, "math", None)
    assert (
        examples[0].reference_answer
        == r"\boxed{15\frac{39}{40}} or \boxed{\frac{639}{40}}"
    )
    assert examples[1].reference_answer == r"\boxed{2}"
    assert examples[1].id == "math:test:1"


@pytest.mark.parametrize(
    "task,row",
    [
        ("gsm8k", {"question": "Q", "answer": "no delimiter"}),
        ("gsm8k", {"question": "Q", "answer": "work#### "}),
        ("math", {"problem": "Q", "solution": ""}),
    ],
)
def test_malformed_rows_fail(tmp_path, monkeypatch, task, row):
    monkeypatch.setitem(runner.TEST_ROWS, task, 1)
    path = saved_test(tmp_path, [row])
    with pytest.raises(AssertionError):
        runner.load_examples(path, task, None)


class FakeModel:
    config = SimpleNamespace(
        is_encoder_decoder=False, vocab_size=32, max_position_embeddings=2048
    )
    generation_config = SimpleNamespace(eos_token_id=2)
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 0 and kwargs["do_sample"]
        assert kwargs["temperature"] == 0.35 and kwargs["num_return_sequences"] == 1
        tokens = torch.tensor([[7, 2]] * len(kwargs["input_ids"]))
        return SimpleNamespace(
            sequences=torch.cat([kwargs["input_ids"], tokens], dim=1)
        )


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, prompts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "padding": False}
        return {"input_ids": [[1] + [3] * len(p) for p in prompts]}

    def decode(self, tokens, **kwargs):
        assert tokens == [7, 2]
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return "#### 2" + runner.STOP + "unscored text"


def test_real_hf_adapter_cpu_padding_decoding_and_close():
    model = FakeModel()
    generator = runner.PaperGenerator(model, FakeTokenizer())
    result = generator.generate(
        ["abc", "a"], num_samples=1, max_new_tokens=1024, temperature=0.35
    )
    assert result == [[runner.TextCompletion("#### 2", "stop")]] * 2
    assert model.calls[0]["input_ids"].tolist() == [[1, 3, 3, 3], [2, 2, 1, 3]]
    assert model.calls[0]["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    generator.close()
    assert generator.inference.model is None
    with pytest.raises(RuntimeError, match="closed"):
        generator.generate(["a"], num_samples=1, max_new_tokens=1024, temperature=0.35)


def test_long_prompt_budget_does_not_shrink_batch_peers():
    model = FakeModel()
    generator = runner.PaperGenerator(model, FakeTokenizer())
    generator.generate(
        ["a", "b" * 1100, "c"], num_samples=1, max_new_tokens=1024, temperature=0.35
    )
    assert [c["max_new_tokens"] for c in model.calls] == [1024, 946, 1024]
    with pytest.raises(AssertionError, match="context"):
        generator.generate(
            ["x" * 2047], num_samples=1, max_new_tokens=1024, temperature=0.35
        )


class Answers:
    def __init__(self, output, fail=False):
        self.output, self.fail, self.calls = output, fail, 0
        self.closed = False

    def generate(self, prompts, **kwargs):
        self.calls += 1
        assert kwargs == {"num_samples": 1, "max_new_tokens": 1024, "temperature": 0.35}
        if self.calls == 2:
            saved = self.output / "round-00/records.jsonl"
            assert len(saved.read_text().splitlines()) == 2
            assert read_json(self.output / "round-00/metrics.json")["complete"] is False
            if self.fail:
                raise RuntimeError("backend failure")
        return [[runner.TextCompletion("#### 2", "length")] for _ in prompts]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fail", [False, True])
def test_rounds_incremental_metrics_failure_and_cleanup(tmp_path, monkeypatch, fail):
    config = runner.Config("gsm8k", tmp_path, rounds=2, batch_size=2, seed=10)
    examples = [runner.Example(str(i), "Q", str(i)) for i in range(3)]
    seeds = []
    monkeypatch.setattr(runner, "set_seed", seeds.append)
    generator = Answers(tmp_path, fail)
    if fail:
        with (
            pytest.raises(RuntimeError, match="backend failure"),
            runner.inference_context(lambda: generator) as owned,
        ):
            runner.run_rounds(config, examples, owned)
        assert not (tmp_path / "aggregate.json").exists()
        assert read_json(tmp_path / "round-00/metrics.json")["total_count"] == 2
        assert seeds == [10]
    else:
        with runner.inference_context(lambda: generator) as owned:
            metrics = runner.run_rounds(config, examples, owned)
        assert seeds == [10, 11]
        assert metrics["accuracy"] == pytest.approx(1 / 3)
        assert metrics["total_count"] == 6 and metrics["total_correct"] == 2
        assert metrics["round_accuracy_std_population"] == 0
        assert metrics["complete"] and not metrics["paper_protocol_size"]
        assert generator.calls == 4
        for i in range(2):
            rows = [
                json.loads(line)
                for line in (tmp_path / f"round-{i:02d}/records.jsonl")
                .read_text()
                .splitlines()
            ]
            assert [r["example_id"] for r in rows] == ["0", "1", "2"]
            assert all(r["round"] == i and r["seed"] == 10 + i for r in rows)
            assert all(r["sample_index"] == 0 for r in rows)
    assert generator.closed


def test_loading_is_local_and_explicitly_cuda(tmp_path, monkeypatch):
    model = FakeModel()
    model.to = Mock(return_value=model)
    tokenizer = FakeTokenizer()
    load_model = Mock(return_value=model)
    load_tokenizer = Mock(return_value=tokenizer)
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", load_tokenizer)
    generator = load_generator(tmp_path)
    load_model.assert_called_once_with(
        str(tmp_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_safetensors=True,
    )
    load_tokenizer.assert_called_once_with(
        str(tmp_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model.to.assert_called_once_with("cuda:0")
    generator.close()


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_adapter_preserves_finish_reason_without_stop_text(monkeypatch, finish_reason):
    generator = runner.PaperGenerator(FakeModel(), FakeTokenizer())
    monkeypatch.setattr(
        generator.inference,
        "generate",
        lambda *args, **kwargs: [
            [SimpleNamespace(token_ids=[7, 2], finish_reason=finish_reason)]
        ],
    )
    monkeypatch.setattr(generator.tokenizer, "decode", lambda *args, **kwargs: "#### 2")
    assert generator.generate(
        ["Q"], num_samples=1, max_new_tokens=1024, temperature=0.35
    ) == [[runner.TextCompletion("#### 2", finish_reason)]]


def test_math_round_uses_full_solution_reference(tmp_path, monkeypatch):
    config = runner.Config("math", tmp_path, rounds=1)
    example = runner.Example("math:test:0", "Find both.", r"\boxed{1} and \boxed{2}")
    generator = Mock()
    generator.generate.return_value = [[runner.TextCompletion(r"\boxed{2}", "stop")]]
    monkeypatch.setattr(runner, "set_seed", lambda seed: None)
    metrics = runner.run_rounds(config, [example], generator)
    assert metrics["accuracy"] == 0
    record = json.loads((tmp_path / "round-00/records.jsonl").read_text())
    assert record["reference_answer"] == example.reference_answer
    assert record["prompt"] == runner.paper_prompt(example.question)


def test_main_assembly_and_existing_output(tmp_path, monkeypatch, manifest):
    manifest_path, _ = manifest
    output = tmp_path / "run"
    config = runner.Config("gsm8k", output, manifest=manifest_path, rounds=1)
    monkeypatch.setattr(runner, "require_cuda", lambda expected: "mock A40")
    monkeypatch.setattr(
        runner, "load_examples", lambda *args: [runner.Example("id", "Q", "2")]
    )
    generator = Answers(output)
    monkeypatch.setattr(runner, "load_generator", lambda path: generator)
    monkeypatch.setattr(runner, "set_seed", lambda seed: None)
    assert runner.main(config)["accuracy"] == 1
    assert generator.closed
    assert read_json(output / "config.json")["top_p"] == 0.9
    provenance = read_json(output / "provenance.json")
    assert provenance["model"]["revision"] == runner.MODELS["gsm8k"][1]
    assert len(provenance["manifest_sha256"]) == 64
    assert provenance["reference_is_solution"] is False
    with pytest.raises(FileExistsError):
        runner.main(config)
