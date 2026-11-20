"""Offline CUDA preflight of exact-token segmentation, not paper evaluation.

Run from the repository root with python -m scripts.diagnostics.sample_segmentation.
Records preserve all sampled tokens, including EOS; no text/trajectory repair is
performed. A rejection is evidence for a policy decision, not a skipped sample.
"""

import argparse
import hashlib
import inspect
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import groupby
from pathlib import Path

# This import sets HF offline mode before importing the HF libraries.
from scripts.paper_artifact_replications.evaluate_sft import (
    MODELS,
    PROMPT,
    ROOT,
    paper_prompt,
    read_artifacts,
    require_cuda,
    write_json,
)

# isort: split
import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from core.inference import HFInference
from core.segmentation import (
    MATH_SEGMENTATION_POLICY,
    split_solution_inplace,
    step_boundaries_from_tokens,
)

CONTEXT_SIZE = 2048
CATEGORIES = (
    "success",
    "inside_token",
    "special",
    "empty",
    "upstream_or_span",
)


@dataclass(frozen=True)
class Config:
    task: str
    output: Path
    manifest: Path = ROOT / ".cache/download_manifest.json"
    num_prompts: int = 32
    num_samples: int = 2
    batch_size: int = 8
    seed: int = 42
    max_new_tokens: int = 1024

    def __post_init__(self):
        assert self.task in MODELS, "Unsupported task"
        assert all(
            type(n) is int and n > 0
            for n in (
                self.num_prompts,
                self.num_samples,
                self.batch_size,
                self.max_new_tokens,
            )
        ), "Counts and token budget must be positive integers"
        assert 0 <= self.seed < 2**32, "Seed outside NumPy bounds"


def select_prompts(
    dataset: DatasetDict, task: str, count: int, seed: int
) -> list[dict]:
    """Select train rows without replacement; retain their original row indices."""
    assert isinstance(dataset, DatasetDict), "Expected the original saved DatasetDict"
    assert task in MODELS, "Unsupported task"
    train = dataset["train"]
    assert 0 < count <= len(train), "Requested prompts exceed train split or are empty"
    indices = sorted(random.Random(seed).sample(range(len(train)), count))
    selected = []
    for index in indices:
        question = train[index]["question" if task == "gsm8k" else "problem"]
        assert isinstance(question, str) and question.strip(), (
            f"Empty question at {index}"
        )
        if task == "gsm8k":
            question = question.strip()
        selected.append(
            {
                "id": f"{task}:train:{index}",
                "index": index,
                "prompt": paper_prompt(question),
            }
        )
    return selected


def assess(
    response_ids, tokenizer, *, task: str, eos_token_ids, context_ids=()
) -> dict:
    """Run production first; labels describe its first failure, never a repair.

    Unknown/assertion-without-message upstream or span failures stay grouped.
    In particular, do not probe later checks or infer hypothetical extra failures.
    """
    try:
        boundaries = step_boundaries_from_tokens(
            response_ids,
            tokenizer,
            dataset=task,
            eos_token_ids=eos_token_ids,
            context_ids=context_ids,
        )
    except (AssertionError, ValueError) as error:
        message = str(error)
        category = "upstream_or_span"
        for fragment, label in (
            ("inside a sampled token", "inside_token"),
            ("other special tokens", "special"),
            ("Text and sampled response must be nonempty", "empty"),
        ):
            if fragment in message:
                category = label
                break
        if not response_ids and not message:
            category = "empty"
        return {
            "success": False,
            "category": category,
            "exception_type": type(error).__name__,
            "exception_message": message,
        }
    return {"success": True, "category": "success", "boundaries": list(boundaries)}


def run_samples(
    config: Config, prompts: list[dict], tokenizer, inference, eos_ids
) -> dict:
    """Persist each generation sub-batch; generation/decoding errors propagate."""
    assert len(prompts) == config.num_prompts
    prefixes = tokenizer(
        [row["prompt"] for row in prompts],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    assert len(prefixes) == len(prompts)
    assert tokenizer.bos_token_id is not None
    assert all(ids and ids[0] == tokenizer.bos_token_id for ids in prefixes), (
        "Expected BOS"
    )
    budgets = [min(config.max_new_tokens, CONTEXT_SIZE - len(ids)) for ids in prefixes]
    assert all(b > 0 for b in budgets), "Prompt exhausts total context budget"
    metrics = {
        "expected_samples": config.num_prompts * config.num_samples,
        "total_samples": 0,
        "success": 0,
        "failure": 0,
        "by_category": dict.fromkeys(CATEGORIES, 0),
        "by_finish_reason": {
            reason: {
                "total": 0,
                "success": 0,
                "failure": 0,
                "by_category": dict.fromkeys(CATEGORIES, 0),
            }
            for reason in ("stop", "length")
        },
        "complete": False,
    }
    write_json(config.output / "metrics.json", metrics)
    set_seed(config.seed)
    with (config.output / "records.jsonl").open("x") as stream:
        for start in range(0, len(prompts), config.batch_size):
            batch = list(zip(prompts, prefixes, budgets))[
                start : start + config.batch_size
            ]
            # Adjacent equal budgets preserve selection order and prevent a long
            # peer from reducing another prompt's continuation budget.
            for budget, group in groupby(batch, key=lambda row: row[2]):
                rows = list(group)
                completions = inference.generate(
                    [ids for _, ids, _ in rows],
                    num_samples=config.num_samples,
                    max_new_tokens=budget,
                    temperature=1.0,
                )
                assert len(completions) == len(rows), "Wrong prompt cardinality"
                assert all(
                    len(samples) == config.num_samples for samples in completions
                ), "Wrong sample cardinality"
                for (prompt, ids, _), samples in zip(rows, completions, strict=True):
                    for sample_index, completion in enumerate(samples):
                        assert completion.finish_reason in ("stop", "length"), (
                            "Unknown finish reason"
                        )
                        result = assess(
                            completion.token_ids,
                            tokenizer,
                            task=config.task,
                            eos_token_ids=eos_ids,
                            context_ids=ids,
                        )
                        record = prompt | {
                            "sample_index": sample_index,
                            "prompt_ids": ids,
                            "token_ids": completion.token_ids,
                            "text": tokenizer.decode(
                                completion.token_ids,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                            "finish_reason": completion.finish_reason,
                            "max_new_tokens": budget,
                            "segmentation": result,
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        status = "success" if result["success"] else "failure"
                        metrics["total_samples"] += 1
                        metrics[status] += 1
                        metrics["by_category"][result["category"]] += 1
                        finish = metrics["by_finish_reason"][completion.finish_reason]
                        finish["total"] += 1
                        finish[status] += 1
                        finish["by_category"][result["category"]] += 1
                stream.flush()
                metrics["rejection_frequency"] = (
                    metrics["failure"] / metrics["total_samples"]
                )
                metrics["complete"] = (
                    metrics["total_samples"] == metrics["expected_samples"]
                )
                write_json(config.output / "metrics.json", metrics)
    return metrics


def main(config: Config) -> dict:
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite {config.output}")
    gpu = require_cuda(None)
    model_entry, dataset_entry = read_artifacts(config.manifest, config.task)
    dataset = load_from_disk(dataset_entry["path"])
    prompts = select_prompts(dataset, config.task, config.num_prompts, config.seed)
    assert len(dataset["train"]) == dataset_entry["splits"]["train"]["rows"], (
        "Wrong train split size"
    )
    config.output.mkdir(parents=True, exist_ok=False)
    write_json(
        config.output / "config.json",
        asdict(config)
        | {
            "output": str(config.output.resolve()),
            "manifest": str(config.manifest.resolve()),
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "context_size": CONTEXT_SIZE,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "device": "cuda:0",
            "prompt": PROMPT,
            "stop_strings": None,
            "selection": "sorted(random.Random(seed).sample(range(len(train)), num_prompts))",
            "generation_seed": "set_seed(seed) once before batches; batch/group order affects RNG",
            "math_segmentation_policy": MATH_SEGMENTATION_POLICY
            if config.task == "math"
            else None,
        },
    )
    sources = [
        Path(__file__),
        *(
            Path(inspect.getfile(obj))
            for obj in (
                HFInference,
                step_boundaries_from_tokens,
                split_solution_inplace,
                read_artifacts,
            )
        ),
    ]
    write_json(
        config.output / "provenance.json",
        {
            "model": model_entry,
            "dataset": dataset_entry,
            "split": "train",
            "train_fingerprint": dataset["train"]._fingerprint,
            "manifest_sha256": hashlib.sha256(config.manifest.read_bytes()).hexdigest(),
            "selection": prompts,
            "selection_sha256": hashlib.sha256(
                json.dumps(prompts, sort_keys=True).encode()
            ).hexdigest(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "--no-optional-locks", "status", "--short"], cwd=ROOT, text=True
            ),
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sources
            },
            "versions": {
                p: version(p)
                for p in ("torch", "transformers", "datasets", "tokenizers")
            },
            "gpu": gpu,
            "cuda": torch.version.cuda,
            "artifact_integrity": "Manifest pins downloads; local artifact bytes are not rehashed.",
            "classification": "Observed first production failure only; upstream_or_span includes unclassified assertions/ValueErrors. No repairs.",
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_entry["path"], local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_entry["path"],
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_safetensors=True,
    ).to("cuda:0")
    model.eval()
    assert model.config.max_position_embeddings >= CONTEXT_SIZE
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if type(eos) is int else eos
    assert isinstance(eos_ids, list) and eos_ids, "Configure model EOS IDs"
    assert tokenizer.eos_token_id in eos_ids, "Tokenizer/model EOS mismatch"
    write_json(
        config.output / "tokenizer.json",
        {
            "class": type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_ids": eos_ids,
            "pad_token_id": tokenizer.eos_token_id,
            "all_special_ids": tokenizer.all_special_ids,
        },
    )
    with HFInference(model, pad_token_id=tokenizer.eos_token_id) as inference:
        return run_samples(config, prompts, tokenizer, inference, eos_ids)


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(MODELS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / ".cache/download_manifest.json"
    )
    for flag, default in (
        ("num-prompts", 32),
        ("num-samples", 2),
        ("batch-size", 8),
        ("seed", 42),
        ("max-new-tokens", 1024),
    ):
        parser.add_argument(f"--{flag}", type=int, default=default)
    return Config(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main(parse_args())
