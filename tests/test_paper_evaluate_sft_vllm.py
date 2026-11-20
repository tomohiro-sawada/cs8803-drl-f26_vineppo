"""CPU-only tests using fake vLLM and the real shared evaluator/verifier."""

import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/paper_artifact_replications/evaluate_sft_vllm.py"
)
spec = importlib.util.spec_from_file_location("paper_evaluate_sft_vllm", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def no_network_or_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network/GPU/model access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(runner.shared.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner.shared.AutoTokenizer, "from_pretrained", forbidden)


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, prompts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "padding": False}
        return {"input_ids": [[1] + [3] * len(p) for p in prompts]}


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=Mock()),
            vllm_config=SimpleNamespace(
                attention_config=SimpleNamespace(
                    backend=SimpleNamespace(name="FLASH_ATTN")
                )
            ),
        )
        self.before_generate = lambda: None

    def generate(self, prompts, *, sampling_params, use_tqdm):
        self.before_generate()
        assert use_tqdm is False
        self.calls.append((prompts, sampling_params))
        return [
            SimpleNamespace(
                finished=True,
                prompt_token_ids=p["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        index=i,
                        text="#### 2" if i == 0 else "#### 0",
                        finish_reason="length" if i == 0 else "stop",
                    )
                    for i in reversed(range(params.n))
                ],
            )
            for p, params in zip(prompts, sampling_params, strict=True)
        ]


def generator(seed=42):
    return runner.PaperVLLMGenerator(FakeLLM(), Tokenizer(), SimpleNamespace, seed)


def read_json(path):
    return json.loads(path.read_text())


def test_cli(tmp_path):
    config = runner.parse_args(["--task", "math", "--output", str(tmp_path)])
    assert (config.batch_size, config.num_samples, config.seed) == (32, 16, 42)
    with pytest.raises(AssertionError, match="requires --smoke"):
        runner.parse_args(["--task", "math", "--output", str(tmp_path), "--limit", "2"])
    config = runner.parse_args(
        [
            "--task",
            "gsm8k",
            "--output",
            str(tmp_path),
            "--smoke",
            "--limit",
            "2",
            "--expected-gpu",
            "A40",
        ]
    )
    assert config.smoke and config.limit == 2 and config.expected_gpu == "A40"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 0},
        {"batch_size": 0},
        {"seed": -1},
        {"seed": 2**32},
        {"smoke": True, "limit": 0},
    ],
)
def test_invalid_config(tmp_path, kwargs):
    with pytest.raises(AssertionError):
        runner.Config("math", tmp_path, **kwargs)


@pytest.mark.parametrize("task", ["math", "gsm8k"])
@pytest.mark.parametrize("num_samples", [1, 3, 16])
def test_seed_bounds_cover_all_child_seeds(tmp_path, task, num_samples):
    rows = runner.shared.TEST_ROWS[task]
    max_seed = 2**32 - rows * num_samples
    config = runner.Config(task, tmp_path, num_samples=num_samples, seed=max_seed)
    assert config.seed + (rows - 1) * num_samples + (num_samples - 1) == 2**32 - 1
    with pytest.raises(AssertionError, match="Seed out of range"):
        runner.Config(task, tmp_path, num_samples=num_samples, seed=max_seed + 1)


def test_sampling_bos_budgets_seeds_and_sample_order():
    gen = generator(10)
    result = gen.generate(
        ["short", "x" * 1100], num_samples=16, max_new_tokens=1024, temperature=0.35
    )
    prompts, params = gen.llm.calls[0]
    assert prompts[0]["prompt_token_ids"][0] == 1
    assert [p.max_tokens for p in params] == [1024, 946]
    assert [p.seed for p in params] == [10, 26]
    for p in params:
        assert p.n == 16 and p.temperature == 0.35 and p.top_p == 0.9 and p.top_k == -1
        assert p.stop == ["\n\n\nProblem:"]
        assert (
            not p.ignore_eos
            and not p.include_stop_str_in_output
            and p.skip_special_tokens
        )
    assert result[0][0] == runner.TextCompletion("#### 2", "length")
    assert result[0][1] == runner.TextCompletion("#### 0", "stop")
    gen.generate(["next"], num_samples=16, max_new_tokens=1024, temperature=0.35)
    assert gen.llm.calls[1][1][0].seed == 42
    engine = gen.llm.llm_engine.engine_core
    gen.close()
    gen.close()
    engine.shutdown.assert_called_once()
    with pytest.raises(AssertionError, match="closed"):
        gen.generate(["next"], num_samples=16, max_new_tokens=1024, temperature=0.35)


def test_context_exhaustion_and_missing_bos():
    gen = generator()
    with pytest.raises(AssertionError, match="context"):
        gen.generate(
            ["x" * 2047], num_samples=16, max_new_tokens=1024, temperature=0.35
        )
    assert not gen.llm.calls and gen.prompt_count == 0
    gen.tokenizer = Mock(return_value={"input_ids": [[3]]}, bos_token_id=1)
    with pytest.raises(AssertionError, match="BOS"):
        gen.generate(["x"], num_samples=16, max_new_tokens=1024, temperature=0.35)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("num_samples", [1, 3, 16])
def test_real_evaluator_sample_mean_not_pass16(tmp_path, batch_size, num_samples):
    config = runner.Config(
        "gsm8k", tmp_path, batch_size=batch_size, num_samples=num_samples
    )
    examples = [runner.shared.Example(str(i), "Q", "2") for i in range(3)]
    gen = generator()
    metrics = runner.run_batches(config, examples, gen)
    assert metrics["accuracy"] == 1 / num_samples
    assert metrics["total_count"] == 3 * num_samples and metrics["total_correct"] == 3
    assert metrics["num_examples_completed"] == 3
    assert metrics["complete"] is False  # only main can attest cleanup
    assert not metrics["paper_protocol_size"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert [r["seed"] for r in rows[::num_samples]] == [
        42 + i * num_samples for i in range(3)
    ]
    assert [r["sample_index"] for r in rows[:num_samples]] == list(range(num_samples))
    child_seeds = [r["seed"] + r["sample_index"] for r in rows]
    assert child_seeds == list(range(42, 42 + 3 * num_samples))
    assert len(set(child_seeds)) == len(rows)
    assert all(r["prompt"] == runner.shared.paper_prompt("Q") for r in rows)


def test_failure_keeps_incremental_records_and_closes(tmp_path):
    gen = generator()
    engine = gen.llm.llm_engine.engine_core

    def fail_after_first():
        if gen.llm.calls:
            assert len((tmp_path / "records.jsonl").read_text().splitlines()) == 16
            assert read_json(tmp_path / "metrics.json")["total_count"] == 16
            raise RuntimeError("backend failure")

    gen.llm.before_generate = fail_after_first
    with (
        pytest.raises(RuntimeError, match="backend failure"),
        runner.inference_context(lambda: gen),
    ):
        runner.run_batches(
            runner.Config("gsm8k", tmp_path, batch_size=1),
            [runner.shared.Example(str(i), "Q", "2") for i in range(2)],
            gen,
        )
    assert read_json(tmp_path / "metrics.json")["complete"] is False
    engine.shutdown.assert_called_once()


def test_math_uses_solution_verifier(tmp_path):
    gen = generator()
    example = runner.shared.Example(
        "math:test:0", "Find both.", r"\boxed{1} and \boxed{2}"
    )
    gen.llm.generate = lambda prompts, **kwargs: [
        SimpleNamespace(
            finished=True,
            prompt_token_ids=prompts[0]["prompt_token_ids"],
            outputs=[SimpleNamespace(index=0, text=r"\boxed{2}", finish_reason="stop")],
        )
    ]
    metrics = runner.run_batches(
        runner.Config("math", tmp_path, num_samples=1), [example], gen
    )
    assert metrics["accuracy"] == 0
    assert (
        read_json(tmp_path / "records.jsonl")["reference_answer"]
        == example.reference_answer
    )


def test_local_loader_engine_settings_and_backend(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=SimpleNamespace),
    )
    load = Mock(return_value=Tokenizer())
    monkeypatch.setattr(runner.shared.AutoTokenizer, "from_pretrained", load)
    gen = runner.load_generator(tmp_path, 42)
    load.assert_called_once_with(
        str(tmp_path), local_files_only=True, trust_remote_code=False
    )
    assert gen.llm.kwargs == {
        "model": str(tmp_path),
        "tokenizer": str(tmp_path),
        "seed": 42,
        **runner.ENGINE,
    }
    assert runner.ENGINE == {
        "max_model_len": 2048,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_num_seqs": 64,
        "enforce_eager": True,
        "generation_config": "vllm",
        "attention_config": {"backend": "FLASH_ATTN"},
        "trust_remote_code": False,
    }
    gen.close()
    with pytest.raises(AssertionError, match="local"):
        runner.load_generator(Path("remote/model"), 42)


def test_main_provenance_and_existing_output(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = runner.Config("gsm8k", tmp_path / "run", manifest=manifest)
    monkeypatch.setattr(
        runner.shared,
        "read_artifacts",
        lambda *args: ({"path": str(tmp_path)}, {"path": str(tmp_path)}),
    )
    monkeypatch.setattr(
        runner.shared,
        "load_examples",
        lambda *args: [runner.shared.Example("id", "Q", "2")],
    )
    monkeypatch.setattr(runner.shared, "require_cuda", lambda expected: "fake A40")
    monkeypatch.setattr(runner, "version", lambda name: "test-version")
    gen = generator()
    monkeypatch.setattr(runner, "load_generator", lambda *args: gen)
    result = runner.main(config)
    assert result["complete"] and gen.llm is None
    assert result["paper_target_accuracy"] == 0.403
    provenance = read_json(config.output / "provenance.json")
    assert provenance["attention_backend"] == "FLASH_ATTN"
    assert len(provenance["manifest_sha256"]) == 64
    assert all(len(h) == 64 for h in provenance["source_sha256"].values())
    assert provenance["versions"]["vllm"] == "test-version"
    assert "disjoint child seed ranges" in " ".join(provenance["deviations"])
    assert provenance["seed_policy"] == runner.SEED_POLICY
    assert "example index * num_samples" in provenance["seed_policy"]
    assert read_json(config.output / "config.json")["seed_policy"] == runner.SEED_POLICY
    with pytest.raises(FileExistsError):
        runner.main(config)
