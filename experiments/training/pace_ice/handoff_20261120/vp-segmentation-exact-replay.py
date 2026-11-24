from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import traceback

from transformers import AutoTokenizer
from core.segmentation import step_boundaries_from_tokens

root = Path(sys.argv[1])
source = (root / "records.jsonl").read_bytes()
records = [json.loads(line) for line in source.splitlines()]
assert len(records) == 512
meta = json.loads((root / "provenance.json").read_text())
local_model = Path(meta["model"]["path"])
tok = AutoTokenizer.from_pretrained(local_model, local_files_only=True)
assert tok.eos_token_id == json.loads((local_model / "generation_config.json").read_text())["eos_token_id"] == 2
transitions, errors, changed = Counter(), Counter(), 0
for row in records:
    old = row["segmentation"]
    response = row["token_ids"]
    try:
        new = step_boundaries_from_tokens(
            response, tok, dataset="math", eos_token_ids=(2,), context_ids=row["prompt_ids"],
        )
    except (AssertionError, ValueError) as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        transitions[(old["category"], "failure")] += 1
        errors[(Path(frame.filename).name, frame.line, str(exc))] += 1
        continue
    assert new[0] == 0 and new[-1] == len(response)
    assert all(a < b for a,b in zip(new,new[1:]))
    rebuilt = [token for a,b in zip(new,new[1:]) for token in response[a:b]]
    assert rebuilt == response
    transitions[(old["category"], "success")] += 1
    changed += old["success"] and new != tuple(old["boundaries"])
print("source sha256", hashlib.sha256(source).hexdigest(), flush=True)
print("transitions", sorted(transitions.items()), flush=True)
print("changed prior successes", changed, flush=True)
print("remaining", sorted(errors.items()), flush=True)
