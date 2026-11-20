"""OpenRouter text proxy, NOT a test of RhoMath's original sampled-token IDs.

Run with OPENROUTER_API_KEY in the environment. Never stores credentials.
No model/provider fallback or retries; failed API calls abort visibly.
"""

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.diagnostics.sample_segmentation import (
    ROOT,
    AutoTokenizer,
    assess,
    load_from_disk,
    read_artifacts,
    select_prompts,
    write_json,
)

MODEL = "meta-llama/llama-3.2-1b-instruct"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LIMITATION = (
    "General instruction-tuned Llama 3.2 1B proxy, not math-tuned RhoMath. "
    "API text is re-encoded with the pinned RhoMath tokenizer. These are NOT "
    "the provider's or RhoMath's sampled token IDs. No sampled EOS is available "
    "or synthesized. Rejection frequencies do not estimate RhoMath training risk."
)


def complete(prompt: str, seed: int, key: str) -> tuple[dict, dict]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 1024,
        "seed": seed,
        "stream": False,
        "provider": {
            "only": ["cloudflare"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        result = json.load(response)
    assert "error" not in result, "OpenRouter returned an API error"
    assert result["model"] == MODEL, "Unexpected response model"
    assert len(result["choices"]) == 1, "Expected one completion"
    choice = result["choices"][0]
    assert "error" not in choice and choice["finish_reason"] in ("stop", "length")
    assert isinstance(choice["message"]["content"], str), "Missing completion text"
    return body, result


def main(output: Path, count: int = 16, seed: int = 42) -> None:
    assert 0 < count <= 32, "Bound this diagnostic to at most 32 questions per task"
    key = os.environ["OPENROUTER_API_KEY"]
    assert key, "Empty OpenRouter key"
    output.mkdir(parents=True, exist_ok=False)
    manifest = ROOT / ".cache/download_manifest.json"
    provenance = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "limitation": LIMITATION,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "provider": "cloudflare",
        "num_prompts_per_task": count,
        "selection_seed": seed,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "scripts/diagnostics/sample_segmentation.py",
                ROOT / "src/core/segmentation.py",
                ROOT / "src/core/vendor/vineppo/math_extract_steps_inplace.py",
            )
        },
        "versions": {
            name: version(name) for name in ("transformers", "tokenizers", "datasets")
        },
        "tasks": {},
    }
    metrics = {"complete": False, "proxy_only": True, "tasks": {}}
    write_json(output / "metrics.json", metrics)
    for task_index, task in enumerate(("gsm8k", "math")):
        model, dataset_entry = read_artifacts(manifest, task)
        dataset = load_from_disk(dataset_entry["path"])
        assert len(dataset["train"]) == dataset_entry["splits"]["train"]["rows"]
        prompts = select_prompts(dataset, task, count, seed)
        tokenizer = AutoTokenizer.from_pretrained(
            model["path"], local_files_only=True, trust_remote_code=False
        )
        provenance["tasks"][task] = {
            "tokenizer_model": model,
            "tokenizer_sha256": hashlib.sha256(
                (Path(model["path"]) / "tokenizer.json").read_bytes()
            ).hexdigest(),
            "dataset": dataset_entry,
            "split": "train",
            "selection": prompts,
        }
        write_json(output / "provenance.json", provenance)
        categories, finishes = Counter(), Counter()
        with (output / f"{task}.jsonl").open("x") as stream:
            for index, prompt in enumerate(prompts):
                request, response = complete(
                    prompt["prompt"], seed + task_index * count + index, key
                )
                choice = response["choices"][0]
                text = choice["message"]["content"]
                encoded = tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True
                )
                ids = encoded["input_ids"]
                assessment = assess(ids, tokenizer, task=task, eos_token_ids=[])
                record = prompt | {
                    "task": task,
                    "request": request,
                    "response": response,
                    "text": text,
                    "token_id_source": "locally_reencoded_proxy_text_NOT_sampled",
                    "reencoded_token_ids": ids,
                    "character_spans": encoded["offset_mapping"],
                    "decoded_reencoded_text": tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "segmentation": assessment,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                categories[assessment["category"]] += 1
                finishes[choice["finish_reason"]] += 1
                metrics["tasks"][task] = {
                    "count": index + 1,
                    "by_category": dict(categories),
                    "by_finish_reason": dict(finishes),
                }
                write_json(output / "metrics.json", metrics)
                print(task, index + 1, assessment["category"], flush=True)
    metrics["complete"] = True
    write_json(output / "metrics.json", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    main(args.output, args.count)
