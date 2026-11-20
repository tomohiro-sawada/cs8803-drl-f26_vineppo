"""Offline Figure 4 Rho SFT evaluation: batched n=16 sample-mean pass@1.

Run with the inference environment, e.g. python -m
scripts.paper_artifact_replications.evaluate_sft_vllm --task math --output NEW_DIR.
No downloads, chat template, pass@16, or majority vote.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

# Also support direct script execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from core.inference import TextCompletion
from runtime.evaluate import evaluate, inference_context
from scripts.paper_artifact_replications import evaluate_sft as shared

ENGINE = {
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
SEED_POLICY = "base seed + zero-based test example index * num_samples; one seeded n-sample request per prompt"
BUDGET_POLICY = "min(1024, 2047 - prompt tokens including BOS); no truncation or batch-peer shrinking"


@dataclass(frozen=True)
class Config:
    task: str
    output: Path
    manifest: Path = shared.ROOT / ".cache/download_manifest.json"
    batch_size: int = 32
    num_samples: int = 16
    smoke: bool = False
    limit: int | None = None
    expected_gpu: str | None = None
    seed: int = 42

    def __post_init__(self):
        assert self.task in shared.MODELS, "Unsupported task"
        assert self.batch_size > 0 and self.num_samples > 0, (
            "Positive batch size/samples required"
        )
        assert self.limit is None or self.smoke, "--limit requires --smoke"
        assert self.limit is None or self.limit > 0, "Positive limit required"
        assert (
            0 <= self.seed <= 2**32 - shared.TEST_ROWS[self.task] * self.num_samples
        ), "Seed out of range"


class PaperVLLMGenerator:
    """Sequential evaluator adapter; its prompt counter persists across batches."""

    def __init__(self, llm, tokenizer, sampling_params, seed):
        assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
        self.llm, self.tokenizer = llm, tokenizer
        self.sampling_params, self.seed = sampling_params, seed
        self.prompt_count = 0
        self.last_metadata = []

    def generate(self, prompts, *, num_samples, max_new_tokens, temperature):
        assert self.llm is not None, "Generator is closed"
        assert max_new_tokens == 1024 and temperature == 0.35
        prefixes = self.tokenizer(prompts, add_special_tokens=True, padding=False)[
            "input_ids"
        ]
        assert len(prefixes) == len(prompts)
        assert all(p and p[0] == self.tokenizer.bos_token_id for p in prefixes), (
            "Expected BOS"
        )
        budgets = [min(max_new_tokens, 2047 - len(p)) for p in prefixes]
        assert all(b > 0 for b in budgets), "Prompt exceeds paper context size"
        metadata = [
            {
                "example_index": self.prompt_count + i,
                "seed": self.seed + (self.prompt_count + i) * num_samples,
                "prompt_tokens": len(p),
                "max_tokens": budget,
            }
            for i, (p, budget) in enumerate(zip(prefixes, budgets, strict=True))
        ]
        params = [
            self.sampling_params(
                n=num_samples,
                temperature=temperature,
                top_p=0.9,
                top_k=-1,
                max_tokens=m["max_tokens"],
                seed=m["seed"],
                stop=[shared.STOP],
                include_stop_str_in_output=False,
                ignore_eos=False,
                skip_special_tokens=True,
            )
            for m in metadata
        ]
        outputs = self.llm.generate(
            [{"prompt_token_ids": p} for p in prefixes],
            sampling_params=params,
            use_tqdm=False,
        )
        assert len(outputs) == len(prefixes), "Request count mismatch"
        results = []
        for output, prefix in zip(outputs, prefixes, strict=True):
            assert output.finished and list(output.prompt_token_ids) == prefix, (
                "Request order/incomplete output"
            )
            samples = sorted(output.outputs, key=lambda sample: sample.index)
            assert [s.index for s in samples] == list(range(num_samples)), (
                "Sample indices mismatch"
            )
            assert all(s.finish_reason in ("stop", "length") for s in samples), (
                "Unexpected finish reason"
            )
            results.append([TextCompletion(s.text, s.finish_reason) for s in samples])
        self.prompt_count += len(prompts)
        self.last_metadata = metadata
        return results

    def close(self):
        if self.llm is not None:
            # vLLM 0.34's LLM has no public close(); stop its owned engine workers.
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None


def load_generator(path: Path, seed: int) -> PaperVLLMGenerator:
    assert path.is_absolute() and path.is_dir(), "Expected local model directory"
    from vllm import LLM, SamplingParams

    tokenizer = shared.AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    llm = LLM(model=str(path), tokenizer=str(path), seed=seed, **ENGINE)
    generator = PaperVLLMGenerator(llm, tokenizer, SamplingParams, seed)
    backend = llm.llm_engine.vllm_config.attention_config.backend
    if backend.name != "FLASH_ATTN":
        generator.close()
        raise AssertionError(f"Expected FLASH_ATTN, got {backend}")
    return generator


def initial_metrics(config: Config, count: int) -> dict:
    return {
        "complete": False,
        "smoke": config.smoke,
        "metric": "sample-mean pass@1 (not pass@16)",
        "num_examples_requested": count,
        "num_examples_completed": 0,
        "num_samples": config.num_samples,
        "total_count": 0,
        "total_correct": 0,
        "accuracy": None,
        "paper_target_accuracy": shared.TARGETS[config.task],
        "difference_percentage_points": None,
        "paper_protocol_size": not config.smoke
        and count == shared.TEST_ROWS[config.task]
        and config.num_samples == 16,
        "context_budget_policy": BUDGET_POLICY,
        "reduced_budget_prompts": 0,
        "min_max_tokens": None,
        "max_prompt_tokens": 0,
    }


def run_batches(config: Config, examples, generator) -> dict:
    metrics = initial_metrics(config, len(examples))
    shared.write_json(config.output / "metrics.json", metrics)
    with (config.output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for start in range(0, len(examples), config.batch_size):
            batch = examples[start : start + config.batch_size]
            assert generator.prompt_count == start, (
                "Generator must start at example zero"
            )
            records, batch_metrics = evaluate(
                batch,
                generator,
                task=config.task,
                batch_size=config.batch_size,
                num_samples=config.num_samples,
                max_new_tokens=1024,
                temperature=0.35,
                formatter=shared.paper_prompt,
                reference_is_solution=config.task == "math",
            )
            metadata = generator.last_metadata
            assert len(metadata) == len(batch)
            for i, record in enumerate(records):
                stream.write(
                    json.dumps(
                        asdict(record) | metadata[i // config.num_samples],
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            stream.flush()
            metrics["total_count"] += batch_metrics["total_count"]
            metrics["total_correct"] += batch_metrics["total_correct"]
            metrics["num_examples_completed"] += len(batch)
            metrics["accuracy"] = metrics["total_correct"] / metrics["total_count"]
            metrics["difference_percentage_points"] = 100 * (
                metrics["accuracy"] - shared.TARGETS[config.task]
            )
            metrics["reduced_budget_prompts"] += sum(
                m["max_tokens"] < 1024 for m in metadata
            )
            metrics["min_max_tokens"] = min(
                [m["max_tokens"] for m in metadata]
                + [metrics["min_max_tokens"] or 1024]
            )
            metrics["max_prompt_tokens"] = max(
                [m["prompt_tokens"] for m in metadata] + [metrics["max_prompt_tokens"]]
            )
            shared.write_json(config.output / "metrics.json", metrics)
            print(
                f"{config.task}: {metrics['num_examples_completed']}/{len(examples)} prompts, pass@1={metrics['accuracy']:.4%}",
                flush=True,
            )
    # main marks completion only after engine cleanup also succeeds.
    return metrics


def main(config: Config) -> dict:
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite {config.output}")
    model, dataset = shared.read_artifacts(config.manifest, config.task)
    examples = shared.load_examples(Path(dataset["path"]), config.task, config.limit)
    gpu = shared.require_cuda(config.expected_gpu)
    config.output.mkdir(parents=True, exist_ok=False)
    shared.write_json(
        config.output / "metrics.json", initial_metrics(config, len(examples))
    )
    shared.write_json(
        config.output / "config.json",
        {
            **asdict(config),
            "output": str(config.output.resolve()),
            "manifest": str(config.manifest.resolve()),
            "engine": ENGINE,
            "temperature": 0.35,
            "top_p": 0.9,
            "top_k": -1,
            "max_tokens": 1024,
            "prompt": shared.PROMPT,
            "stop": shared.STOP,
            "seed_policy": SEED_POLICY,
            "context_budget_policy": BUDGET_POLICY,
        },
    )
    sources = {
        "runner": Path(__file__),
        "shared_runner": Path(shared.__file__),
        "evaluator": shared.ROOT / "src/runtime/evaluate.py",
        "verifier": shared.ROOT / "src/core/verifier.py",
    }
    shared.write_json(
        config.output / "provenance.json",
        {
            "paper": shared.PAPER,
            "upstream_commit": shared.UPSTREAM_COMMIT,
            "model": model,
            "dataset": dataset,
            "manifest_sha256": hashlib.sha256(config.manifest.read_bytes()).hexdigest(),
            "selection_sha256": hashlib.sha256(
                json.dumps([asdict(e) for e in examples]).encode()
            ).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in sources.items()
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=shared.ROOT, text=True
            ).strip(),
            "versions": {
                p: version(p)
                for p in ("vllm", "torch", "transformers", "datasets", "tokenizers")
            },
            "gpu": gpu,
            "cuda": shared.torch.version.cuda,
            "attention_backend": "FLASH_ATTN",
            "verifier_policy": shared.VERIFIER_POLICY,
            "verifier_provenance": shared.VERIFIER_PROVENANCE,
            "reference_is_solution": config.task == "math",
            "seed_policy": SEED_POLICY,
            "deviations": [
                "Modern vLLM/torch rather than upstream versions; no bitwise parity claim",
                "Explicit per-prompt base+index*num_samples seeds rather than upstream global RNG; stable assignment across batches, not guaranteed bitwise batch invariance",
                "vLLM 0.34 derives child seeds as prompt seed + sample index; num_samples stride gives disjoint child seed ranges across prompts",
                "Long-prompt budget includes BOS; upstream tokenize() does not",
                "Manifest attests downloads, not post-download local file integrity",
                "Shared safe verifier errors abort instead of counting infrastructure failures as wrong",
            ],
        },
    )
    with inference_context(
        lambda: load_generator(Path(model["path"]), config.seed)
    ) as generator:
        metrics = run_batches(config, examples, generator)
    metrics["complete"] = True
    shared.write_json(config.output / "metrics.json", metrics)
    return metrics


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(shared.MODELS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=shared.ROOT / ".cache/download_manifest.json"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, help="First N test rows; requires --smoke")
    parser.add_argument("--expected-gpu", help="Case-insensitive CUDA:0 name substring")
    parser.add_argument("--seed", type=int, default=42)
    return Config(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main(parse_args())
