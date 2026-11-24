"""CPU-only diagnostic tests: production segmentation, fake tokens/completions."""

import importlib
import json
import random
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest
import torch
from datasets import Dataset, DatasetDict

from core.inference import Completion

with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    sampler = importlib.import_module("scripts.diagnostics.sample_segmentation")


class Tokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = (1, 2, 3)
    added_tokens_decoder = {}

    def __init__(self, *, prefixes=None):
        self.prefixes = prefixes

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        vocabulary = {1: "<bos>", 2: "<eos>", 3: "<pad>", 4: "a\nb", 5: "ab"}
        return "".join(
            vocabulary[token] if token < 10 else chr(token - 10) for token in ids
        )

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        padding,
        truncation,
        return_offsets_mapping=False,
    ):
        assert isinstance(text, list) and add_special_tokens
        assert not padding and not truncation and not return_offsets_mapping
        return {
            "input_ids": self.prefixes
            if self.prefixes is not None
            else [[1] + tokens(t) for t in text]
        }


def tokens(text):
    return [ord(char) + 10 for char in text]


@pytest.fixture(autouse=True)
def no_network_or_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network/model/GPU access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sampler.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(sampler.AutoModelForCausalLM, "from_pretrained", forbidden)


def test_defaults_and_required_arguments(tmp_path):
    config = sampler.parse_args(["--task", "math", "--output", str(tmp_path)])
    assert (
        config.num_prompts,
        config.num_samples,
        config.batch_size,
        config.seed,
        config.max_new_tokens,
    ) == (32, 2, 8, 42, 1024)
    assert config.manifest == sampler.ROOT / ".cache/download_manifest.json"
    for args in ([], ["--task", "math"], ["--output", str(tmp_path)]):
        with pytest.raises(SystemExit):
            sampler.parse_args(args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_prompts": 0},
        {"num_samples": -1},
        {"batch_size": 0},
        {"max_new_tokens": 0},
        {"seed": -1},
        {"seed": 2**32},
    ],
)
def test_bad_config(tmp_path, kwargs):
    with pytest.raises(AssertionError):
        sampler.Config("gsm8k", tmp_path, **kwargs)


@pytest.mark.parametrize("task,column", [("gsm8k", "question"), ("math", "problem")])
def test_selection_uses_train_original_indices_and_local_rng(task, column, tmp_path):
    data = DatasetDict(
        {
            "train": Dataset.from_dict({column: [f"  train {i}  " for i in range(20)]}),
            "test": Dataset.from_dict({column: ["DO NOT SELECT"]}),
        }
    )
    data.save_to_disk(tmp_path / "dataset")
    loaded = sampler.load_from_disk(str(tmp_path / "dataset"))
    state = random.getstate()
    selected = sampler.select_prompts(loaded, task, 5, 42)
    assert random.getstate() == state
    expected = sorted(random.Random(42).sample(range(20), 5))
    assert [row["index"] for row in selected] == expected
    assert selected == sampler.select_prompts(loaded, task, 5, 42)
    for row, index in zip(selected, expected, strict=True):
        question = f"train {index}" if task == "gsm8k" else f"  train {index}  "
        assert row == {
            "id": f"{task}:train:{index}",
            "index": index,
            "prompt": sampler.paper_prompt(question),
        }
    with pytest.raises(AssertionError, match="exceed"):
        sampler.select_prompts(loaded, task, 21, 42)
    with pytest.raises(AssertionError, match="DatasetDict"):
        sampler.select_prompts(loaded["train"], task, 1, 42)
    with pytest.raises(KeyError):
        sampler.select_prompts(DatasetDict({"test": loaded["test"]}), task, 1, 42)


@pytest.mark.parametrize("task", ["gsm8k", "math"])
@pytest.mark.parametrize("eos", [False, True])
def test_assess_success_keeps_exact_ids(task, eos):
    ids = tokens("First step.\nSecond step.") + ([2] if eos else [])
    original = ids.copy()
    result = sampler.assess(ids, Tokenizer(), task=task, eos_token_ids=[2])
    assert result["success"] and result["category"] == "success"
    assert result["boundaries"][0] == 0 and result["boundaries"][-1] == len(ids)
    assert ids == original


@pytest.mark.parametrize(
    "ids,tokenizer,task,category,exception",
    [
        ([3, 5], Tokenizer(), "gsm8k", "special", "AssertionError"),
        ([2, 2], Tokenizer(), "gsm8k", "special", "AssertionError"),
        ([2], Tokenizer(), "gsm8k", "empty", "AssertionError"),
        ([], Tokenizer(), "gsm8k", "empty", "AssertionError"),
        (tokens("x"), Tokenizer(), "math", "upstream_or_span", "AssertionError"),
        (tokens("\n\n"), Tokenizer(), "gsm8k", "upstream_or_span", "AssertionError"),
    ],
)
def test_observed_failure_categories(ids, tokenizer, task, category, exception):
    result = sampler.assess(ids, tokenizer, task=task, eos_token_ids=[2])
    assert result["success"] is False
    assert result["category"] == category
    assert result["exception_type"] == exception
    assert isinstance(result["exception_message"], str)
    assert "boundaries" not in result


def test_only_production_assertion_and_valueerror_are_caught(monkeypatch):
    production = Mock(side_effect=RuntimeError("infrastructure"))
    monkeypatch.setattr(sampler, "step_boundaries_from_tokens", production)
    with pytest.raises(RuntimeError, match="infrastructure"):
        sampler.assess([2], Tokenizer(), task="gsm8k", eos_token_ids=[2])
    production.assert_called_once_with(
        [2],
        ANY,
        dataset="gsm8k",
        eos_token_ids=[2],
        context_ids=(),
        context_special_ids=(),
    )


def test_generation_eos_ids_are_authoritative():
    result = sampler.assess([5], Tokenizer(), task="gsm8k", eos_token_ids=[5])
    assert result["category"] == "empty"


def test_empty_does_not_override_an_earlier_production_failure():
    tokenizer = Tokenizer()
    tokenizer.is_fast = False
    result = sampler.assess([], tokenizer, task="gsm8k", eos_token_ids=[2])
    assert result["category"] == "upstream_or_span"
    assert result["exception_message"] == "Segmentation requires a fast tokenizer"


def read_records(path):
    return [
        json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()
    ]


def prompts(count):
    return [
        {"id": f"gsm8k:train:{i}", "index": i, "prompt": f"p{i}"} for i in range(count)
    ]


def test_batches_budgets_all_samples_and_metrics(tmp_path):
    config = sampler.Config(
        "gsm8k", tmp_path, num_prompts=4, num_samples=2, batch_size=3
    )
    prefixes = [[1] + tokens("p") * (length - 1) for length in (3, 1500, 3, 3)]
    tokenizer = Tokenizer(prefixes=prefixes)

    def generate(prefixes, **kwargs):
        assert kwargs["temperature"] == 1.0 and kwargs["num_samples"] == 2
        return [
            [Completion(tokens("ok") + [2], "stop"), Completion([4], "length")]
            for _ in prefixes
        ]

    inference = Mock(generate=Mock(side_effect=generate))
    metrics = sampler.run_samples(config, prompts(4), tokenizer, inference, [2])
    assert [
        call.kwargs["max_new_tokens"] for call in inference.generate.call_args_list
    ] == [1024, 548, 1024, 1024]
    records = read_records(tmp_path)
    assert len(records) == 8
    for index, record in enumerate(records):
        assert record["index"] == index // 2 and record["sample_index"] == index % 2
        assert record["prompt_ids"] == prefixes[index // 2]
        assert record["token_ids"] == (tokens("ok") + [2] if index % 2 == 0 else [4])
        assert record["text"] == ("ok<eos>" if index % 2 == 0 else "a\nb")
    assert metrics["complete"] and metrics["total_samples"] == 8
    assert metrics["success"] == metrics["failure"] == 4
    assert metrics["rejection_frequency"] == 0.5
    assert metrics["by_finish_reason"]["stop"]["success"] == 4
    assert metrics["by_finish_reason"]["length"]["by_category"]["inside_token"] == 4
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert not (tmp_path / "metrics.tmp").exists()


def test_equal_budgets_share_a_batch(tmp_path):
    inference = Mock()
    inference.generate.return_value = [[Completion(tokens("ok"), "length")]] * 2
    sampler.run_samples(
        sampler.Config("gsm8k", tmp_path, num_prompts=2, num_samples=1),
        prompts(2),
        Tokenizer(),
        inference,
        [2],
    )
    assert inference.generate.call_count == 1
    assert len(inference.generate.call_args.args[0]) == 2


@pytest.mark.parametrize(
    "error",
    [
        AssertionError("generation"),
        ValueError("generation"),
        RuntimeError("generation"),
    ],
)
def test_generation_failure_propagates_with_previous_batch_flushed(tmp_path, error):
    inference = Mock()

    def generate(*args, **kwargs):
        if inference.generate.call_count == 2:
            assert len(read_records(tmp_path)) == 1
            saved = json.loads((tmp_path / "metrics.json").read_text())
            assert saved["total_samples"] == 1 and not saved["complete"]
            raise error
        return [[Completion([2], "stop")]]

    inference.generate.side_effect = generate
    with pytest.raises(type(error), match="generation"):
        sampler.run_samples(
            sampler.Config(
                "gsm8k", tmp_path, num_prompts=2, num_samples=1, batch_size=1
            ),
            prompts(2),
            Tokenizer(),
            inference,
            [2],
        )
    assert len(read_records(tmp_path)) == 1


@pytest.mark.parametrize("result", [[], [[]], [[Completion([2], "unexpected")]]])
def test_malformed_generation_is_not_a_segmentation_rejection(tmp_path, result):
    inference = Mock()
    inference.generate.return_value = result
    with pytest.raises(AssertionError):
        sampler.run_samples(
            sampler.Config("gsm8k", tmp_path, num_prompts=1, num_samples=1),
            prompts(1),
            Tokenizer(),
            inference,
            [2],
        )
    assert read_records(tmp_path) == []
    assert not json.loads((tmp_path / "metrics.json").read_text())["complete"]


def test_context_exhaustion_rejected_before_generation(tmp_path):
    inference = Mock()
    with pytest.raises(AssertionError, match="exhausts"):
        sampler.run_samples(
            sampler.Config("gsm8k", tmp_path, num_prompts=1),
            prompts(1),
            Tokenizer(prefixes=[[1] * 2048]),
            inference,
            [2],
        )
    inference.generate.assert_not_called()


def test_existing_output_rejected_before_cuda(tmp_path):
    with pytest.raises(FileExistsError):
        sampler.main(sampler.Config("gsm8k", tmp_path))


def test_main_pinned_train_model_loading_and_provenance(tmp_path, monkeypatch):
    output = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    data = DatasetDict({"train": Dataset.from_dict({"question": ["first", "second"]})})
    entries = (
        {"path": "local-model", "revision": "pinned"},
        {"path": "local-data", "splits": {"train": {"rows": 2}}},
    )
    monkeypatch.setattr(sampler, "require_cuda", lambda _: "fake GPU")
    artifacts = Mock(return_value=entries)
    monkeypatch.setattr(sampler, "read_artifacts", artifacts)
    # Source inspection still needs the real pinned helper, not the test mock.
    getfile = sampler.inspect.getfile
    monkeypatch.setattr(
        sampler.inspect,
        "getfile",
        lambda obj: getfile(sampler.paper_prompt) if obj is artifacts else getfile(obj),
    )
    monkeypatch.setattr(sampler, "load_from_disk", Mock(return_value=data))
    tokenizer = Tokenizer()
    token_loader = Mock(return_value=tokenizer)
    model = Mock(
        config=SimpleNamespace(
            max_position_embeddings=2048, is_encoder_decoder=False, vocab_size=1000
        ),
        generation_config=SimpleNamespace(eos_token_id=2),
    )
    model.to.return_value = model
    loader = Mock(return_value=model)
    monkeypatch.setattr(sampler.AutoTokenizer, "from_pretrained", token_loader)
    monkeypatch.setattr(sampler.AutoModelForCausalLM, "from_pretrained", loader)
    # Keep HFInference itself for provenance; bypass only its generation method.

    monkeypatch.setattr(
        sampler.HFInference,
        "generate",
        lambda self, prefixes, **kwargs: [
            [Completion([2], "stop") for _ in range(kwargs["num_samples"])]
            for _ in prefixes
        ],
    )
    config = sampler.Config("gsm8k", output, manifest=manifest, num_prompts=2)
    result = sampler.main(config)
    artifacts.assert_called_once_with(manifest, "gsm8k")
    assert loader.call_args.kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "use_safetensors": True,
    }
    model.to.assert_called_once_with("cuda:0")
    assert result["complete"] and result["failure"] == 4
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["split"] == "train" and provenance["model"] == entries[0]
    assert len(provenance["source_sha256"]) == 5 and provenance["git_commit"]
    settings = json.loads((output / "config.json").read_text())
    assert (settings["top_p"], settings["temperature"], settings["context_size"]) == (
        1.0,
        1.0,
        2048,
    )
