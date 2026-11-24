"""Read-only what-if: recover nested placeholders in reverse insertion order (in memory only)."""
import json
from core.vendor.vineppo import math_extract_steps_inplace as up

S = "/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo"
rows = [json.loads(line) for line in open(f"{S}/training/pace_ice/olmo3-sft-math-segmentation-hf-20/records.jsonl")]
texts = [row["text"].removesuffix("<|endoftext|>") for row in rows]


def split(text):
    try:
        return up.split_solution_inplace(text)
    except AssertionError as error:
        return str(error) or "AssertionError"


before = [split(text) for text in texts]
original = up._recover


def reverse(text, placeholders):
    for key in reversed(list(placeholders)):
        text = text.replace(key, placeholders[key])
    return text


up._recover = reverse
after = [split(text) for text in texts]
up._recover = original
accepted = [i for i, b in enumerate(before) if isinstance(b, list)]
print(json.dumps({"accepted_before": len(accepted), "unchanged_after": sum(before[i] == after[i] for i in accepted)}))
for i, b in enumerate(before):
    if not isinstance(b, list):
        a = after[i]
        steps = [texts[i][x:y] for x, y in zip(a, a[1:])] if isinstance(a, list) else None
        print(json.dumps({"prompt": rows[i]["id"], "sample": rows[i]["sample_index"], "after": len(a) - 1 if isinstance(a, list) else a,
                          "longest_step_chars": max(map(len, steps)) if steps else None, "first_steps": [s[:60] for s in steps[:3]] if steps else None}))
