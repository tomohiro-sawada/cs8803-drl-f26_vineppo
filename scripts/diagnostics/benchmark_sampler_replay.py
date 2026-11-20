"""Frozen sampler replay: time K-sibling MC sampling on identical recorded prefixes.

Takes the training runner's CLI through launch.sh
(TRAINING_ENTRY=scripts/diagnostics/benchmark_sampler_replay.py --backend vllm).
No weights change. The first run samples primary responses from the SFT actor
on the learner, segments them, and records every nonterminal prefix with its
exact budget in <output>/prefixes.json; set REPLAY_PREFIXES to that file to
replay the same workload under another layout. Each configuration is timed by
the coordinator from first request to last result and appended to
<output>/replay.jsonl. HF_PREFIX_LIMIT bounds the sequential HF control to the
first N prefixes (its rate is reported for exactly those); REPLAY_MAX_SEQUENCES
lists the in-flight sequence bounds to sweep on the replicas.
"""

import json
import os
import sys
from pathlib import Path
from statistics import quantiles
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from algorithms.vineppo import _get_prefixes, continuation_budget
from core.data import Episode
from core.inference import HFInference
from core.lora import apply_lora
from core.segmentation import step_boundaries_from_tokens
from core.vllm_inference import VLLMClient, VLLMInference, gpu_identity
from experiments.training.run import load_examples, paper_prompt, parse_args


def record_prefixes(args, actor, tokenizer):
    examples, _ = load_examples(args.dataset_path, args.task)
    order = torch.randperm(
        len(examples), generator=torch.Generator().manual_seed(args.seed)
    )[: args.prompts_per_iteration].tolist()
    prompts = [
        tokenizer(paper_prompt(examples[i].question))["input_ids"] for i in order
    ]
    primary = HFInference(actor, pad_token_id=tokenizer.eos_token_id).generate(
        prompts, num_samples=args.num_samples, max_new_tokens=args.max_new_tokens
    )
    prefixes, budgets = [], []
    for prompt, group in zip(prompts, primary, strict=True):
        for completion in group:
            episode = Episode(
                prompt, completion.token_ids, 0.0, completion.finish_reason
            )
            boundaries = step_boundaries_from_tokens(
                completion.token_ids,
                tokenizer,
                dataset=args.task,
                eos_token_ids=[tokenizer.eos_token_id],
                context_ids=prompt,
            )
            for prefix in _get_prefixes(episode, boundaries):
                prefixes.append(prefix)
                budgets.append(
                    continuation_budget(
                        len(prefix),
                        max_sequence_length=args.max_sequence_length,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
    return {
        "prefixes": prefixes,
        "budgets": budgets,
        "episodes": sum(map(len, primary)),
    }


def measure(name, sampler, prefixes, budgets, samples, extra):
    torch.cuda.synchronize()
    started = perf_counter()
    groups = sampler.generate_each(
        prefixes, budgets, num_samples=samples, temperature=1.0
    )
    torch.cuda.synchronize()
    seconds = perf_counter() - started
    lengths = sorted(len(c.token_ids) for group in groups for c in group)
    cuts = quantiles(lengths, n=100)
    return {
        "configuration": name,
        "prefixes": len(prefixes),
        "siblings": samples,
        "completions": len(lengths),
        "output_tokens": sum(lengths),
        "seconds": seconds,
        "output_tokens_per_second": sum(lengths) / seconds,
        "output_length": {
            "p50": cuts[49],
            "p90": cuts[89],
            "p99": cuts[98],
            "max": lengths[-1],
        },
        "truncated": sum(c.finish_reason == "length" for g in groups for c in g),
        "prefix_length": {
            "mean": sum(map(len, prefixes)) / len(prefixes),
            "max": max(map(len, prefixes)),
        },
    } | extra


def main():
    args = parse_args()
    assert args.backend == "vllm" and args.finetuning == "lora"
    learner_gpu = gpu_identity()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True
    )
    actor = apply_lora(
        AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            local_files_only=True,
            dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
        ).to("cuda:0"),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    client, workers = VLLMClient.from_directory(
        args.worker_dir,
        count=args.num_workers,
        api_key=os.environ["VLLM_API_KEY"],
        model_path=args.model_path,
        other_gpu_uuids=[learner_gpu["uuid"]],
    )
    args.output.mkdir(parents=True)
    replay = os.environ.get("REPLAY_PREFIXES")
    workload = (
        json.loads(Path(replay).read_text())
        if replay
        else record_prefixes(args, actor, tokenizer)
    )
    (args.output / "prefixes.json").write_text(json.dumps(workload) + "\n")
    prefixes, budgets = workload["prefixes"], workload["budgets"]
    layout = {
        "learner_gpu": learner_gpu["name"],
        "workers": len(workers),
        "worker_gpus": [worker["gpu"]["name"] for worker in workers],
        "allocated_gpus": len(workers) + 1,
        "replayed_from": replay,
    }
    remote = VLLMInference(
        actor,
        client,
        pad_token_id=tokenizer.eos_token_id,
        export_dir=args.output / "actors",
        seed=args.seed,
        manifest={},
    )
    remote.sync(0)
    rows = []
    with (args.output / "replay.jsonl").open("x") as stream:

        def emit(row):
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)

        # Discarded warm-up: engine compilation and allocator growth.
        remote.generate_each(
            prefixes[:8], budgets[:8], num_samples=args.mc_num_samples, temperature=1.0
        )
        for bound in map(
            int, os.environ.get("REPLAY_MAX_SEQUENCES", "32,128,256,512").split(",")
        ):
            client.max_sequences = bound
            before = [dict(stats) for stats in client.stats]
            row = measure(
                f"vllm-1+{len(workers)}",
                remote,
                prefixes,
                budgets,
                args.mc_num_samples,
                layout | {"max_in_flight_sequences": bound},
            )
            row["per_worker"] = [
                {key: after[key] - prior[key] for key in after}
                for after, prior in zip(client.stats, before, strict=True)
            ]
            emit(row)
        limit = int(os.environ.get("HF_PREFIX_LIMIT", len(prefixes)))
        local = HFInference(actor, pad_token_id=tokenizer.eos_token_id)
        local.generate_each(
            prefixes[:2], budgets[:2], num_samples=args.mc_num_samples, temperature=1.0
        )
        emit(
            measure(
                "hf-sequential",
                local,
                prefixes[:limit],
                budgets[:limit],
                args.mc_num_samples,
                layout | {"workers": 0, "allocated_gpus": 1},
            )
        )


if __name__ == "__main__":
    main()
