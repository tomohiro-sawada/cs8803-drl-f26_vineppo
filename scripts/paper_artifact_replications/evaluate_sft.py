"""Offline, CUDA-only evaluation of the paper's initial Rho SFT checkpoints."""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from statistics import mean, pstdev

# Set before importing HF libraries, including when launched as a standalone script.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from datasets import DatasetDict, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from core.data import Example
from core.inference import HFInference, TextCompletion
from core.verifier import UPSTREAM_COMMIT, VERIFIER_POLICY, VERIFIER_PROVENANCE
from runtime.evaluate import evaluate, inference_context

ROOT = Path(__file__).resolve().parents[2]
PAPER = "https://arxiv.org/html/2410.01679v2"
PROMPT = "[MATH_TASK] Problem:\n{question}\n\nSolution:"
STOP = "\n\n\nProblem:"
DATASET_SHA256 = "c25d3977febe21260a1d6fe484ef368774e3d99ef10ef3f07a4531f4cd02a503"
MODELS = {
    "gsm8k": (
        "realtreetune/rho-1b-sft-GSM8K",
        "b28fda3216f8178c1f37c3c764196e3ca2439a30",
    ),
    "math": (
        "realtreetune/rho-1b-sft-MATH",
        "f4d0f86bfce159f0042cca9116a52c0b972f34aa",
    ),
}
TEST_ROWS = {"gsm8k": 1319, "math": 500}
TARGETS = {"gsm8k": 0.403, "math": 0.155}


@dataclass(frozen=True)
class Config:
    task: str
    output: Path
    manifest: Path = ROOT / ".cache/download_manifest.json"
    batch_size: int = 8
    rounds: int = 16
    limit: int | None = None
    smoke: bool = False
    seed: int = 42
    expected_gpu: str | None = None

    def __post_init__(self):
        assert self.task in MODELS, "Unsupported task"
        assert self.batch_size > 0 and self.rounds > 0, (
            "Positive batch size/rounds required"
        )
        assert 0 <= self.seed <= 2**32 - self.rounds, (
            "Seed range exceeds NumPy seed bounds"
        )
        assert self.limit is None or self.smoke, "--limit requires --smoke"
        assert self.limit is None or self.limit > 0, "Positive limit required"


def paper_prompt(question: str) -> str:
    return PROMPT.format(question=question)


def read_artifacts(manifest: Path, task: str) -> tuple[dict, dict]:
    data = json.loads(manifest.read_text())
    assert data["schema_version"] == 1 and data["upstream_commit"] == UPSTREAM_COMMIT
    selected = {}
    for kind in ("model", "dataset"):
        entries = [
            e for e in data["artifacts"] if e["task"] == task and e["kind"] == kind
        ]
        assert len(entries) == 1, f"Manifest must contain exactly one {task}/{kind}"
        entry = entries[0]
        path = Path(entry["path"])
        assert path.is_absolute() and path.is_dir(), f"Missing local artifact: {path}"
        selected[kind] = entry
    model, dataset = selected["model"], selected["dataset"]
    assert (model["repo_id"], model["revision"]) == MODELS[task], (
        "Unpinned model revision"
    )
    assert dataset["sha256"] == DATASET_SHA256, "Not the pinned paper dataset archive"
    assert dataset["splits"]["test"]["rows"] == TEST_ROWS[task], "Wrong test split"
    return model, dataset


def load_examples(path: Path, task: str, limit: int | None) -> list[Example]:
    dataset = load_from_disk(str(path))
    assert isinstance(dataset, DatasetDict), "Expected the original saved DatasetDict"
    test = dataset["test"]
    assert len(test) == TEST_ROWS[task], "Wrong paper test set size"
    assert limit is None or limit <= len(test), "Limit exceeds test split"
    examples = []
    for index, row in enumerate(test):
        if limit is not None and index == limit:
            break
        question = row["question"].strip() if task == "gsm8k" else row["problem"]
        answer = row["answer"] if task == "gsm8k" else row["solution"]
        assert isinstance(question, str) and question.strip(), (
            f"Empty question at {index}"
        )
        assert isinstance(answer, str) and answer.strip(), f"Empty answer at {index}"
        if task == "gsm8k":
            parts = answer.split("####")
            assert len(parts) == 2, f"Malformed GSM8K reference at {index}"
            answer = parts[1].strip()

        assert answer, f"Empty final answer at {index}"
        examples.append(Example(f"{task}:test:{index}", question, answer))
    assert examples, "Empty selection"
    return examples


class PaperSamplingModel:
    """Evaluation-only facade: retain HFInference mechanics, override only nucleus p.

    HFInference deliberately disables nucleus filtering for training. Do not mutate
    the shared implementation or monkeypatch the actual model's generate method.
    """

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.generation_config = model.generation_config
        self.device = model.device

    def eval(self):
        self.model.eval()

    def generate(self, **kwargs):
        assert kwargs["top_p"] == 1.0 and not kwargs["output_scores"]
        return self.model.generate(**(kwargs | {"top_p": 0.9}))


class PaperGenerator:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
        assert model.generation_config.eos_token_id == tokenizer.eos_token_id
        self.inference = HFInference(
            PaperSamplingModel(model), pad_token_id=tokenizer.eos_token_id
        )

    def generate(self, prompts, *, num_samples, max_new_tokens, temperature):
        assert num_samples == 1 and max_new_tokens == 1024 and temperature == 0.35
        prefixes = self.tokenizer(prompts, add_special_tokens=True, padding=False)[
            "input_ids"
        ]
        assert all(p[0] == self.tokenizer.bos_token_id for p in prefixes), (
            "Expected BOS"
        )
        # Upstream reserves one context token and shrinks each long prompt's budget.
        # Group adjacent equal budgets; never shrink short prompts due to batch peers.
        budgets = [min(max_new_tokens, 2047 - len(p)) for p in prefixes]
        assert all(b > 0 for b in budgets), "Prompt exceeds paper context size"
        results = []
        for budget, group in groupby(zip(prefixes, budgets), key=lambda pair: pair[1]):
            completions = self.inference.generate(
                [prefix for prefix, _ in group],
                num_samples=1,
                max_new_tokens=budget,
                temperature=temperature,
            )
            for (completion,) in completions:
                text = self.tokenizer.decode(
                    completion.token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # HFInference only stops at EOS/budget. Exclude the upstream stop
                # string and all following text from scoring (extra compute only).
                before, separator, _ = text.partition(STOP)
                results.append(
                    [
                        TextCompletion(
                            before, "stop" if separator else completion.finish_reason
                        )
                    ]
                )
        return results

    def close(self):
        self.inference.close()


def require_cuda(expected_gpu: str | None) -> str:
    assert torch.cuda.is_available(), "CUDA is required; no CPU/MPS fallback"
    name = torch.cuda.get_device_name(0)
    assert expected_gpu is None or expected_gpu.casefold() in name.casefold(), (
        f"Expected GPU {expected_gpu!r}, found {name!r}"
    )
    assert torch.cuda.is_bf16_supported(), "CUDA device must support bfloat16"
    return name


def load_generator(path: Path) -> PaperGenerator:
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_safetensors=True,
    ).to("cuda:0")
    model.eval()
    return PaperGenerator(model, tokenizer)


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_rounds(config: Config, examples: list[Example], generator) -> dict:
    """Borrow the backend; persist each successful batch before starting the next."""
    completed = []
    for round_index in range(config.rounds):
        seed = config.seed + round_index
        set_seed(seed)
        directory = config.output / f"round-{round_index:02d}"
        directory.mkdir()
        correct = count = 0
        with (
            (directory / "records.jsonl").open("x") as stream,
            tqdm(
                total=len(examples),
                desc=f"{config.task} round {round_index + 1}/{config.rounds}",
                unit="example",
            ) as bar,
        ):
            for start in range(0, len(examples), config.batch_size):
                records, metrics = evaluate(
                    examples[start : start + config.batch_size],
                    generator,
                    task=config.task,
                    batch_size=config.batch_size,
                    num_samples=1,
                    max_new_tokens=1024,
                    temperature=0.35,
                    formatter=paper_prompt,
                    reference_is_solution=config.task == "math",
                )
                for record in records:
                    stream.write(
                        json.dumps(
                            asdict(record) | {"round": round_index, "seed": seed}
                        )
                        + "\n"
                    )
                stream.flush()
                correct += metrics["total_correct"]
                count += metrics["total_count"]
                round_metrics = {
                    "round": round_index,
                    "seed": seed,
                    "total_correct": correct,
                    "total_count": count,
                    "accuracy": correct / count,
                    "complete": count == len(examples),
                }
                write_json(directory / "metrics.json", round_metrics)
                bar.update(len(records))
                bar.set_postfix(accuracy=f"{correct / count:.2%}")
        completed.append(round_metrics)
        accuracies = [r["accuracy"] for r in completed]
        aggregate = {
            "complete": len(completed) == config.rounds,
            "smoke": config.smoke,
            "rounds_completed": len(completed),
            "rounds_requested": config.rounds,
            "num_examples": len(examples),
            "total_count": sum(r["total_count"] for r in completed),
            "total_correct": sum(r["total_correct"] for r in completed),
            "accuracy": mean(accuracies),
            "round_accuracy_std_population": pstdev(accuracies),
            "paper_target_accuracy": TARGETS[config.task],
            "difference_percentage_points": 100
            * (mean(accuracies) - TARGETS[config.task]),
            "paper_protocol_size": not config.smoke and config.rounds == 16,
            "rounds": completed,
        }
        write_json(config.output / "aggregate.json", aggregate)
        tqdm.write(
            f"Completed {len(completed)} rounds: mean accuracy {aggregate['accuracy']:.2%}"
        )
    return aggregate


def main(config: Config) -> dict:
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite {config.output}")
    gpu = require_cuda(config.expected_gpu)
    model, dataset = read_artifacts(config.manifest, config.task)
    examples = load_examples(Path(dataset["path"]), config.task, config.limit)
    config.output.mkdir(parents=True, exist_ok=False)
    write_json(
        config.output / "config.json",
        {
            **asdict(config),
            "output": str(config.output.resolve()),
            "manifest": str(config.manifest.resolve()),
            "temperature": 0.35,
            "max_new_tokens": 1024,
            "top_p": 0.9,
            "top_k": 0,
            "prompt": PROMPT,
            "stop": STOP,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "device": "cuda:0",
            "context_size": 2047,
            "seed_policy": "seed + zero-based round index",
        },
    )
    write_json(
        config.output / "provenance.json",
        {
            "paper": PAPER,
            "upstream_commit": UPSTREAM_COMMIT,
            "model": model,
            "dataset": dataset,
            "manifest_sha256": hashlib.sha256(config.manifest.read_bytes()).hexdigest(),
            "selection_sha256": hashlib.sha256(
                json.dumps([asdict(e) for e in examples]).encode()
            ).hexdigest(),
            "verifier_policy": VERIFIER_POLICY,
            "verifier_provenance": VERIFIER_PROVENANCE,
            "reference_is_solution": config.task == "math",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {
                p: version(p)
                for p in ("torch", "transformers", "datasets", "tokenizers")
            },
            "gpu": gpu,
            "cuda": torch.version.cuda,
            "deviations": [
                "HF SDPA rather than upstream vLLM; no bitwise parity claim",
                "Stop string removed after generation; extra draws affect subsequent RNG state",
                "Sequential seeded rounds rather than upstream batched n=16 completions",
                "Long-prompt budget counts BOS; upstream tokenize() does not",
                "Manifest attests downloads, not post-download local file integrity",
                "Shared safe verifier errors abort instead of counting infrastructure failures as wrong",
            ],
        },
    )
    with inference_context(lambda: load_generator(Path(model["path"]))) as generator:
        return run_rounds(config, examples, generator)


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(MODELS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / ".cache/download_manifest.json"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, help="First N test rows; requires --smoke")
    parser.add_argument(
        "--expected-gpu", help="Case-insensitive CUDA:0 name substring, e.g. A40"
    )
    return Config(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main(parse_args())
