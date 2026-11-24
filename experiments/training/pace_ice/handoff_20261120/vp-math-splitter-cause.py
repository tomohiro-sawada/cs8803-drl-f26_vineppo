"""Read-only: why the upstream MATH splitter rejects seven OLMo responses, and what each policy would do."""
import json, collections
from datasets import load_from_disk
from core.vendor.vineppo import math_extract_steps_inplace as up
from core.verifier import verify_response

S = "/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo"
rows = [json.loads(line) for line in open(f"{S}/training/pace_ice/olmo3-sft-math-segmentation-hf-20/records.jsonl")]
train = load_from_disk(f"{S}/cache/math/datasets/math")["train"]
EOS = "<|endoftext|>"


def outcome(text):
    try:
        return len(up.split_solution_inplace(text)) - 1
    except AssertionError:
        return None


def nested(text):
    """Placeholders that sit inside a later placeholder's value."""
    _, placeholders = up._disguise(text)
    keys = list(placeholders)
    return sum(any(key in placeholders[later] for later in keys[i + 1 :]) for i, key in enumerate(keys))


def recover_nested(text, placeholders):
    for key in reversed(list(placeholders)):
        text = text.replace(key, placeholders[key])
    return text


by_prompt = collections.defaultdict(list)
fail = []
for row in rows:
    text = row["text"].removesuffix(EOS).removesuffix("<|im_end|>")
    steps = outcome(text)
    by_prompt[row["id"]].append(steps)
    if steps is None:
        fail.append((row, text))
print(json.dumps({"samples": len(rows), "rejected": len(fail), "prompts": len(by_prompt), "prompts_with_a_rejection": sum(any(s is None for s in v) for v in by_prompt.values())}))
ok = [s for v in by_prompt.values() for s in v if s is not None]
print(json.dumps({"accepted_steps_mean": round(sum(ok) / len(ok), 1), "accepted_with_nested_placeholders": sum(nested(r["text"].removesuffix(EOS)) > 0 for r in rows if outcome(r["text"].removesuffix(EOS)) is not None)}))
for row, text in fail:
    disguised, placeholders = up._disguise(text)
    plain = up._recover(disguised, placeholders)
    fixed = recover_nested(disguised, placeholders)
    index = int(row["id"].split(":")[-1])
    example = train[index]
    reward = verify_response(text, example["solution"], task="math", question=example["problem"], reference_is_solution=True)
    siblings = by_prompt[row["id"]]
    print(json.dumps({
        "prompt": row["id"], "sample": row["sample_index"], "tokens": len(row["token_ids"]), "finish": row["finish_reason"],
        "nested_placeholders": nested(text), "upstream_recovery_restores_text": plain == text, "reverse_order_recovery_restores_text": fixed == text,
        "leftover_placeholder_in_upstream_recovery": "<|PLACEHOLDER_" in plain,
        "currency_dollar_signs": text.count("$"), "correct": reward.correct, "extracted": reward.extracted_answer,
        "siblings_rejected": sum(s is None for s in siblings), "siblings": len(siblings),
        "sibling_steps": [s for s in siblings if s is not None],
        "question": example["problem"][:110],
    }))
