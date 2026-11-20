"""Offline checkpoint evaluation; validation selects, frozen selection gates test.

--backend hf loads the actor on this process's GPU. --backend vllm publishes the
checkpoint's LoRA adapter, under a name embedding its content hash, to the idle
training replicas and samples there; the training actor stays loaded and is
verified untouched afterwards. --baseline evaluates the unadapted base model
(iteration 0), which is reported separately and is never a selection candidate.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from core.data import Example
from core.inference import TextCompletion
from core.verifier import VERIFIER_POLICY, VERIFIER_PROVENANCE
from core.vllm_inference import (
    BASE_MODEL,
    VLLMClient,
    publish_adapter,
    retire_adapter,
    stream_seed,
)
from experiments.training.tracking import WandBTracker
from runtime.checkpoint_actor import load_checkpoint_actor
from runtime.evaluate import evaluate, inference_context
from scripts.paper_artifact_replications.evaluate_sft import (
    PROMPT,
    STOP,
    PaperGenerator,
    paper_prompt,
)

EXPECTED_ROWS = {
    "validation": {"gsm8k": 373, "math": 500},
    "test": {"gsm8k": 1319, "math": 500},
}
PROTOCOL = {
    "temperature": 0.35,
    "top_p": 0.9,
    "top_k": 0,
    "max_new_tokens": 1024,
    "context_size": 2047,
    "num_samples": 1,
    "prompt": PROMPT,
    "stop": STOP,
    "stop_policy": "trim after generation",
    "seed_policy": "seed + zero-based round index",
    "accuracy_policy": "mean Pass@1; no truncation penalty",
}
VLLM_PROTOCOL = PROTOCOL | {
    "backend": "vllm",
    "stop_policy": "server-side stop string",
    "seed_policy": "sha256(seed, question counter across rounds) per request",
}


def protocol_for(backend):
    return VLLM_PROTOCOL if backend == "vllm" else PROTOCOL


class RemotePaperGenerator:
    """Paper decoding on vLLM replicas serving `model`; one seeded request per prompt."""

    def __init__(self, client, tokenizer, model, seed, release):
        assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
        self.client, self.tokenizer, self.model = client, tokenizer, model
        self.seed, self.release, self.count = seed, release, 0

    def generate(self, prompts, *, num_samples, max_new_tokens, temperature):
        assert num_samples == 1 and max_new_tokens == 1024 and temperature == 0.35
        prefixes = self.tokenizer(prompts, add_special_tokens=True, padding=False)[
            "input_ids"
        ]
        assert all(p[0] == self.tokenizer.bos_token_id for p in prefixes), (
            "Expected BOS"
        )
        budgets = [min(max_new_tokens, 2047 - len(p)) for p in prefixes]
        assert all(b > 0 for b in budgets), "Prompt exceeds paper context size"
        results = self.client.complete(
            [
                {
                    "model": self.model,
                    "prompt": prefix,
                    "n": 1,
                    "seed": stream_seed(self.seed, self.count + index),
                    "max_tokens": budget,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 0,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "stop": [STOP],
                    "add_special_tokens": False,
                    "skip_special_tokens": True,
                    "return_token_ids": True,
                }
                for index, (prefix, budget) in enumerate(
                    zip(prefixes, budgets, strict=True)
                )
            ]
        )
        self.count += len(prompts)
        return [
            [TextCompletion(choice["text"], choice["finish_reason"])]
            for (choice,) in results
        ]

    def close(self):
        self.release()
        self.client.close()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    for name in ("training-output", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--checkpoint-source", type=Path)
    parser.add_argument("--checkpoint-source-sha256")
    parser.add_argument("--rounds", type=positive_int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--expected-gpu")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--num-workers", type=positive_int)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="vineppo")
    for name in ("entity", "group", "name"):
        parser.add_argument(f"--wandb-{name}")
    args = parser.parse_args(argv)
    if args.split == "validation" and (
        (args.checkpoint is None) != args.baseline or args.selection is not None
    ):
        parser.error(
            "validation requires exactly one of --checkpoint/--baseline, no --selection"
        )
    if args.baseline and (args.split == "test" or args.checkpoint_source is not None):
        parser.error("--baseline is a validation of the unadapted base model")
    if (args.backend == "vllm") != (args.num_workers is not None) or (
        args.backend == "vllm"
    ) != (args.worker_dir is not None):
        parser.error(
            "--backend vllm requires --num-workers and --worker-dir, hf neither"
        )
    if args.split == "test" and (
        args.selection is None or args.checkpoint is not None or args.rounds != 16
    ):
        parser.error("test requires --selection, --rounds 16, and forbids --checkpoint")
    if (args.checkpoint_source is None) != (args.checkpoint_source_sha256 is None):
        parser.error("checkpoint source requires path and metadata SHA256 together")
    if args.split == "test" and args.checkpoint_source is not None:
        parser.error("test checkpoint source comes from frozen selection")
    if not 0 <= args.seed <= 2**32 - args.rounds:
        parser.error("seed range exceeds NumPy bounds")
    args.training_output = args.training_output.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    return args


def load_examples(path, task, split):
    data = load_from_disk(str(path))
    assert isinstance(data, DatasetDict), "Expected local DatasetDict"
    selected = data[split]
    assert len(selected) == EXPECTED_ROWS[split][task], "Wrong paper split size"
    examples = []
    digest = hashlib.sha256()
    for index, row in enumerate(selected):
        question = row["question" if task == "gsm8k" else "problem"]
        answer = row["answer" if task == "gsm8k" else "solution"]
        assert isinstance(question, str) and question.strip(), "Empty question"
        assert isinstance(answer, str) and answer.strip(), "Empty reference"
        if task == "gsm8k":
            parts = answer.split("####")
            assert len(parts) == 2 and parts[1].strip(), "Malformed GSM8K reference"
            question, answer = question.strip(), parts[1].strip()
        example = Example(f"{task}:{split}:{index}", question, answer)
        examples.append(example)
        digest.update(
            (
                json.dumps(asdict(example), sort_keys=True, ensure_ascii=False) + "\n"
            ).encode()
        )
    return examples, {
        "path": str(path),
        "fingerprint": selected._fingerprint,
        "rows": len(selected),
        "examples_sha256": digest.hexdigest(),
    }


def validate_validation(result, task, training_output):
    """Validate the complete evidence, not just a caller-supplied accuracy."""
    assert result["schema_version"] == 1 and result["complete"] is True
    assert result["task"] == task and result["split"] == "validation"
    assert result["training_run"]["path"] == str(training_output)
    protocol = result["protocol"]
    assert all(
        protocol[key] == value
        for key, value in protocol_for(protocol.get("backend")).items()
    )
    n = protocol["rounds"]
    assert type(n) is int and n > 0
    assert type(protocol["batch_size"]) is int and protocol["batch_size"] > 0
    assert type(protocol["seed"]) is int and 0 <= protocol["seed"] <= 2**32 - n
    rows = EXPECTED_ROWS["validation"][task]
    assert result["dataset"]["rows"] == result["num_examples"] == rows
    assert result["rounds_completed"] == len(result["rounds"]) == n
    for index, row in enumerate(result["rounds"]):
        assert row["complete"] is True and row["round"] == index
        assert row["seed"] == protocol["seed"] + index
        assert type(row["total_count"]) is int and row["total_count"] == rows
        assert type(row["total_correct"]) is int and 0 <= row["total_correct"] <= rows
        assert row["accuracy"] == row["total_correct"] / rows
    accuracies = [row["accuracy"] for row in result["rounds"]]
    assert result["total_count"] == rows * n
    assert result["total_correct"] == sum(
        row["total_correct"] for row in result["rounds"]
    )
    assert result["accuracy"] == mean(accuracies)
    assert result["round_accuracy_std_population"] == pstdev(accuracies)
    checkpoint = result["checkpoint"]
    assert Path(checkpoint["path"]).is_absolute()
    assert type(checkpoint["iteration"]) is int and checkpoint["iteration"] >= 0
    assert len(checkpoint["sha256"]) == 64 and all(
        c in "0123456789abcdef" for c in checkpoint["sha256"]
    )


def checkpoint_source(candidate):
    """Resolve actor metadata separately from the selection invocation's owner."""
    source = candidate["checkpoint_source"]
    path = Path(source["path"]).resolve(strict=True)
    assert str(path) == source["path"]
    assert file_sha256(path / "metadata.json") == source["metadata_sha256"], (
        "Checkpoint source metadata changed"
    )
    return path


def selected_checkpoint(selection, task, training_output):
    assert selection["schema_version"] == 1 and selection["frozen"] is True
    assert selection["task"] == task and selection["training_output"] == str(
        training_output
    )
    assert (
        selection["tie_rule"] == "highest_validation_accuracy_then_earliest_iteration"
    ), "Unsupported tie rule"
    best, candidates = selection["best"], selection["candidates"]
    assert candidates and sum(candidate == best for candidate in candidates) == 1, (
        "Best must equal exactly one candidate"
    )
    assert [c["checkpoint"]["iteration"] for c in candidates] == selection[
        "candidate_iterations"
    ], "Candidates do not match the planned iteration schedule"
    for candidate in candidates:
        validate_validation(candidate, task, training_output)
        checkpoint_source(candidate)
        assert (
            candidate["dataset"]
            == best["dataset"]
            == selection["plan"]["validation_dataset"]
        )
        assert candidate["protocol"] == selection["validation_protocol"], (
            "Candidate protocol differs from frozen validation protocol"
        )
    for key in ("path", "iteration"):
        assert len({c["checkpoint"][key] for c in candidates}) == len(candidates), (
            f"Duplicate checkpoint {key}"
        )
    assert best == min(
        candidates, key=lambda c: (-c["accuracy"], c["checkpoint"]["iteration"])
    ), "Best violates selection tie rule"
    checkpoint = Path(best["checkpoint"]["path"]).resolve(strict=True)
    assert file_sha256(checkpoint) == best["checkpoint"]["sha256"], (
        "Selected checkpoint hash mismatch"
    )
    return checkpoint


def run_rounds(args, task, examples, generator, identity):
    rounds = []
    for index in range(args.rounds):
        seed = args.seed + index
        set_seed(seed)
        directory = args.output / f"round-{index:02d}"
        directory.mkdir()
        row = {
            "round": index,
            "seed": seed,
            "total_correct": 0,
            "total_count": 0,
            "accuracy": 0.0,
            "complete": False,
        }
        write_json(directory / "metrics.json", row)
        with (directory / "records.jsonl").open("x") as stream:
            for start in range(0, len(examples), args.batch_size):
                records, metrics = evaluate(
                    examples[start : start + args.batch_size],
                    generator,
                    task=task,
                    batch_size=args.batch_size,
                    num_samples=1,
                    max_new_tokens=1024,
                    temperature=0.35,
                    formatter=paper_prompt,
                    reference_is_solution=task == "math",
                )
                for record in records:
                    stream.write(
                        json.dumps(
                            asdict(record) | {"round": index, "seed": seed} | identity
                        )
                        + "\n"
                    )
                stream.flush()
                row["total_count"] += metrics["total_count"]
                row["total_correct"] += metrics["total_correct"]
                row["accuracy"] = row["total_correct"] / row["total_count"]
                row["complete"] = row["total_count"] == len(examples)
                write_json(directory / "metrics.json", row)
        assert row["complete"], "Incomplete round coverage"
        rounds.append(row)
        write_json(
            args.output / "progress.json",
            {"complete": False, "rounds_completed": len(rounds), "rounds": rounds},
        )
    accuracies = [row["accuracy"] for row in rounds]
    return {
        "complete": True,
        "accuracy": mean(accuracies),
        "total_count": sum(row["total_count"] for row in rounds),
        "total_correct": sum(row["total_correct"] for row in rounds),
        "num_examples": len(examples),
        "rounds_completed": len(rounds),
        "round_accuracy_std_population": pstdev(accuracies),
        "rounds": rounds,
    }


def main(argv=None):
    """Return the final result dict; write result.json only after successful cleanup."""
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    write_json(
        args.output / "progress.json",
        {"complete": False, "rounds_completed": 0, "rounds": []},
    )
    try:
        metadata = json.loads((args.training_output / "metadata.json").read_text())
        cli = metadata["cli"]
        task = cli["task"]
        assert task in EXPECTED_ROWS[args.split], "Invalid training task"
        assert metadata["prompt"] == PROMPT, "Training prompt mismatch"
        assert metadata["tokenizer_sha256"], "Missing training tokenizer hash"
        model_path = Path(cli["model_path"]).resolve(strict=True)
        dataset_path = Path(cli["dataset_path"]).resolve(strict=True)
        assert model_path.is_dir() and dataset_path.is_dir(), "Local artifacts required"
        protocol = protocol_for(args.backend) | {
            "rounds": args.rounds,
            "seed": args.seed,
            "batch_size": args.batch_size,
        }
        selection = None
        if args.split == "test":
            selection = json.loads(args.selection.read_text())
            assert selection["test_protocol"] == protocol, (
                "Test CLI protocol differs from frozen selection"
            )
            checkpoint = selected_checkpoint(selection, task, args.training_output)
        elif args.baseline:
            checkpoint = model_path / "model.safetensors"
        else:
            checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        source = (
            selection["best"]["checkpoint_source"]
            if selection is not None
            else {
                "path": str(args.training_output),
                "metadata_sha256": file_sha256(args.training_output / "metadata.json"),
            }
            if args.checkpoint_source is None
            else {
                "path": str(args.checkpoint_source.resolve(strict=True)),
                "metadata_sha256": args.checkpoint_source_sha256,
            }
        )
        source_path = checkpoint_source({"checkpoint_source": source})
        source_metadata = json.loads((source_path / "metadata.json").read_text())
        for key in ("dataset", "model", "prompt", "tokenizer_sha256"):
            assert source_metadata[key] == metadata[key], f"Source {key} mismatch"
        assert source_metadata["cli"]["task"] == task
        metadata = source_metadata
        cli = metadata["cli"]
        client = workers = None
        if args.backend == "vllm":
            client, workers = VLLMClient.from_directory(
                args.worker_dir,
                count=args.num_workers,
                api_key=os.environ["VLLM_API_KEY"],
                model_path=model_path,
            )
            gpu = [worker["gpu"] for worker in workers]
        else:
            assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
            assert torch.cuda.device_count() == 1, "Expose exactly one CUDA device"
            gpu = torch.cuda.get_device_name(0)
            assert (
                args.expected_gpu is None
                or args.expected_gpu.casefold() in gpu.casefold()
            ), "Expected GPU mismatch"
            torch.cuda.set_device(0)
        device = "cpu" if args.backend == "vllm" else "cuda:0"
        if args.baseline:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=False
            )
            assert (
                hashlib.sha256(
                    tokenizer.backend_tokenizer.to_str().encode("utf-8")
                ).hexdigest()
                == metadata["tokenizer_sha256"]
            ), "Tokenizer hash mismatch"
            info = {"baseline": True, "dtype": cli["dtype"]}
            checkpoint_info = {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "iteration": 0,
            }
            model = None
            if args.backend == "hf":
                model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=False,
                        dtype=getattr(torch, cli["dtype"]),
                        attn_implementation="sdpa",
                        use_safetensors=True,
                    )
                    .to(device)
                    .eval()
                )
        else:
            model, tokenizer, info = load_checkpoint_actor(
                checkpoint, model_path, device=device
            )
            if info["dtype"] == "bfloat16":
                assert torch.cuda.is_bf16_supported(including_emulation=False), (
                    "Native BF16 CUDA support required"
                )
            assert info["tokenizer_sha256"] == metadata["tokenizer_sha256"], (
                "Tokenizer hash mismatch"
            )
            expected_evaluation_provenance = {
                "schema_version": 1,
                "task": cli["task"],
                "prompt": metadata["prompt"],
                "tokenizer_sha256": metadata["tokenizer_sha256"],
                "dataset": metadata["dataset"],
            }
            assert info["evaluation_provenance"] == expected_evaluation_provenance, (
                "Checkpoint evaluation provenance mismatch"
            )
            config = dict(info["training_config"])
            algorithm = config.pop("algorithm_type")
            assert cli["algorithm"] in ("ppo", "vineppo", "grpo", "rloo")
            assert (
                algorithm
                == f"algorithms.{cli['algorithm']}.{ {'ppo': 'PPOConfig', 'vineppo': 'VinePPOConfig', 'grpo': 'GRPOConfig', 'rloo': 'RLOOConfig'}[cli['algorithm']] }"
            ), "Algorithm identity mismatch"
            assert config == metadata["trainer_config"], "Training config mismatch"
            for key in ("dtype", "finetuning"):
                assert info[key] == cli[key], f"Training {key} mismatch"
            assert info["lora"] == metadata["model"]["lora"], "Training LoRA mismatch"
            assert file_sha256(checkpoint) == info["checkpoint_sha256"], (
                "Checkpoint changed during loading"
            )
            checkpoint_info = {
                "path": str(checkpoint),
                "sha256": info["checkpoint_sha256"],
                "iteration": info["iteration"],
            }
        identity = {}
        if args.backend == "vllm":
            served = client.served()
            actor, release = BASE_MODEL, lambda: None
            if not args.baseline:
                assert info["lora"] is not None, "Replicas serve LoRA actors"
                actor = publish_adapter(
                    client,
                    model,
                    args.output / "actor",
                    f"eval-{checkpoint_info['iteration']}",
                    {"checkpoint": checkpoint_info},
                )
                release = lambda: retire_adapter(client, actor, args.output / "actor")
            identity = {"actor": actor}

            def make_generator():
                def restore():
                    release()
                    assert client.served() == served, (
                        "Evaluation changed the replicas' training actors"
                    )

                return RemotePaperGenerator(
                    client, tokenizer, actor, args.seed, restore
                )
        else:

            def make_generator():
                return PaperGenerator(model, tokenizer)

        if selection is not None:
            assert checkpoint_info == selection["best"]["checkpoint"], (
                "Selected checkpoint identity mismatch"
            )
        examples, dataset = load_examples(dataset_path, task, args.split)
        if selection is not None:
            assert dataset == selection["plan"]["test_dataset"], "Test data changed"
        wandb_path = args.training_output / "wandb.json"
        training_run = {
            "path": str(args.training_output),
            "wandb": json.loads(wandb_path.read_text())
            if wandb_path.exists()
            else None,
        }
        base = {
            "schema_version": 1,
            "task": task,
            "split": args.split,
            "checkpoint": checkpoint_info,
            "protocol": protocol,
            "dataset": dataset,
            "training_run": training_run,
            "checkpoint_source": source,
            "evaluation_run": training_run,
        } | identity
        write_json(
            args.output / "provenance.json",
            base
            | {
                "model": metadata["model"],
                "model_path": str(model_path),
                "actor": info,
                "metadata_sha256": file_sha256(source_path / "metadata.json"),
                "selection": selection,
                "gpu": gpu,
                "software": {
                    name: version(name) for name in ("torch", "transformers", "peft")
                },
                "cuda": torch.version.cuda,
                "attention": "sdpa" if args.backend == "hf" else "vllm",
                "workers": workers,
                "evaluation_code_sha256": file_sha256(Path(__file__)),
                "verifier_policy": VERIFIER_POLICY,
                "verifier_provenance": VERIFIER_PROVENANCE,
                "deviations": [
                    "HF rather than upstream vLLM; no bitwise parity claim",
                    "Stop string trimmed after generation; extra draws affect subsequent RNG state",
                    "Sequential seeded rounds rather than upstream batched n=16 completions",
                    "Checkpoint dtype preserved rather than forced BF16",
                    "Long-prompt budget counts BOS; upstream tokenize() does not",
                ],
            },
        )
        with WandBTracker(
            args,
            {
                "evaluation": {"task": task, "split": args.split, **protocol},
                "training_run": training_run,
                "checkpoint": checkpoint_info,
            },
        ) as tracker:
            with inference_context(make_generator) as generator:
                result = base | run_rounds(args, task, examples, generator, identity)
            tracker.log_evaluation(result, selection=selection)
        write_json(args.output / "result.json", result)
        return result
    except BaseException as error:
        write_json(
            args.output / "failure.json",
            {"complete": False, "failure_type": type(error).__name__},
        )
        raise


if __name__ == "__main__":
    main()
