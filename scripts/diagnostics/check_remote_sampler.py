"""Sampler acceptance: do vLLM replicas sample from the learner's actor?

Takes the training runner's CLI (launch it through launch.sh with
TRAINING_ENTRY=scripts/diagnostics/check_remote_sampler.py --backend vllm) and
writes <output>/acceptance.json. On fixed prefixes it compares, per replica:

- greedy continuations of the zero-initialised adapter against the unadapted
  base on the same replica (same-backend control) and against the HF learner;
- teacher-forced log-probabilities of fixed token sequences against the
  learner, before and after a deliberate adapter change.

Differences and near-tied logits are recorded, not thresholded: BF16 replica
adapters versus FP32 learner adapters make small differences expected. Hard
failures are a replica whose scores ignore the adapter change, a replica that
still serves a retired actor, and a stale-version response.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.inference import HFInference, token_logprobs
from core.lora import apply_lora
from core.prompt_protocol import prompt_protocol
from core.vllm_inference import BASE_MODEL, VLLMClient, VLLMInference, gpu_identity
from experiments.training.run import load_examples, parse_args

GREEDY_TOKENS = 96


def replica_scores(client, model, sequences):
    """Teacher-forced log p(sequence[1:]) from one replica."""
    scores = []
    for sequence in sequences:
        response = client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": sequence,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 1,
                "add_special_tokens": False,
            },
        )
        assert response.status_code == 200, response.text
        entries = response.json()["choices"][0]["prompt_logprobs"]
        assert entries[0] is None and len(entries) == len(sequence)
        scores.append([e[str(t)]["logprob"] for e, t in zip(entries[1:], sequence[1:])])
    return scores


def learner_scores(actor, sequences):
    scores = []
    for sequence in sequences:
        ids = torch.tensor([sequence], device=actor.device)
        with torch.no_grad():
            scores.append(token_logprobs(actor, ids, torch.ones_like(ids))[0].tolist())
    return scores


def differences(left, right):
    flat = [
        abs(a - b)
        for x, y in zip(left, right, strict=True)
        for a, b in zip(x, y, strict=True)
    ]
    return {
        "max_abs": max(flat),
        "mean_abs": sum(flat) / len(flat),
        "tokens": len(flat),
    }


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
        other_gpu_uuids=[learner_gpu["uuid"]]
        if args.worker_placement == "dedicated"
        else [],
    )
    assert args.worker_placement != "colocated" or (
        workers[0]["gpu"]["uuid"] == learner_gpu["uuid"]
    ), "A colocated worker must share the learner GPU"
    args.output.mkdir(parents=True)
    examples, _ = load_examples(args.dataset_path, args.task)
    profile = prompt_protocol(args.model_protocol, args.task)
    prompts = profile.encode(
        tokenizer,
        [
            profile.format(tokenizer, e.question)
            for e in examples[: args.prompts_per_iteration]
        ],
    )
    remote = VLLMInference(
        actor,
        client,
        pad_token_id=tokenizer.eos_token_id,
        export_dir=args.output / "actors",
        seed=args.seed,
        manifest={},
    )
    report = {"learner_gpu": learner_gpu, "workers": workers, "prompts": len(prompts)}

    remote.sync(0)
    first = remote.adapter
    greedy = {
        "adapter": remote.generate(
            prompts, max_new_tokens=GREEDY_TOKENS, temperature=0.0
        )
    }
    remote.adapter = BASE_MODEL
    greedy["base"] = remote.generate(
        prompts, max_new_tokens=GREEDY_TOKENS, temperature=0.0
    )
    remote.adapter = first
    greedy["learner"] = [
        HFInference(actor, pad_token_id=tokenizer.eos_token_id).generate(
            [prompt], max_new_tokens=GREEDY_TOKENS, temperature=0.0
        )[0]
        for prompt in prompts
    ]
    tokens = {k: [g[0].token_ids for g in v] for k, v in greedy.items()}
    report["greedy"] = {
        other: {
            "identical_sequences": sum(
                a == b for a, b in zip(tokens["adapter"], tokens[other])
            ),
            "first_divergence": [
                next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                for a, b in zip(tokens["adapter"], tokens[other])
            ],
        }
        for other in ("base", "learner")
    }
    sequences = [p + t for p, t in zip(prompts, tokens["adapter"], strict=True)]
    # Near ties explain greedy divergences that are not weight mismatches.
    gaps = []
    for sequence in sequences:
        ids = torch.tensor([sequence], device=actor.device)
        with torch.no_grad():
            top = actor(input_ids=ids).logits[0].float().topk(2).values
        gaps.append((top[:, 0] - top[:, 1]).min().item())
    report["learner_min_top2_logit_gap"] = gaps

    before = learner_scores(actor, sequences)
    report["zero_adapter"] = [
        differences(replica_scores(c, first, sequences), before) for c in client.clients
    ]
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    remote.sync(1)
    after = learner_scores(actor, sequences)
    report["learner_change"] = differences(after, before)
    report["changed_adapter"] = []
    for index, worker in enumerate(client.clients):
        scores = replica_scores(worker, remote.adapter, sequences)
        moved = differences(scores, before)
        assert moved["max_abs"] > 10 * report["zero_adapter"][index]["max_abs"], (
            f"Worker {index} ignored the adapter change"
        )
        report["changed_adapter"].append(
            {"vs_learner": differences(scores, after), "vs_previous_version": moved}
        )
        stale = worker.post(
            "/v1/completions",
            json={"model": first, "prompt": prompts[0], "max_tokens": 1},
        )
        assert stale.status_code != 200, f"Worker {index} still serves a retired actor"
    report["retired_actor_rejected_by_every_worker"] = True
    report["published"] = [
        json.loads(line)
        for line in (args.output / "actors/published.jsonl").read_text().splitlines()
    ]
    (args.output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("greedy", "zero_adapter", "learner_change", "changed_adapter")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
