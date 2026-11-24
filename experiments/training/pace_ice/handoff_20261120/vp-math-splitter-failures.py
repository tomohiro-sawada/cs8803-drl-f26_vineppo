"""Read-only: the OLMo MATH responses the step splitter rejects, with the failing frame."""
import json, traceback
from transformers import AutoTokenizer
from core import segmentation as seg
from core.prompt_protocol import prompt_protocol

S = "/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo"
tok = AutoTokenizer.from_pretrained(f"{S}/cache/math/models/Olmo-3-7B-Instruct-SFT-e1452fc", local_files_only=True)


out = f"{S}/training/pace_ice/olmo3-sft-math-segmentation-hf-20"
eos = json.load(open(f"{out}/tokenizer.json"))["eos_token_ids"]
controls = prompt_protocol("olmo3-sft", "math").context_special_ids(tok)
rows = [json.loads(line) for line in open(f"{out}/records.jsonl")]
ok_steps, failures = [], []
for index, row in enumerate(rows):
    try:
        cuts = seg.step_boundaries_from_tokens(row["token_ids"], tok, dataset="math", eos_token_ids=eos, context_ids=row["prompt_ids"], context_special_ids=controls)
        ok_steps.append(len(cuts) - 1)
    except (AssertionError, ValueError) as error:
        frames = traceback.extract_tb(error.__traceback__)
        failures.append((index, row, error, [(f.filename.split("/")[-1], f.lineno, f.name, (f.line or "")[:110]) for f in frames[-3:]]))
print(json.dumps({"samples": len(rows), "failures": len(failures), "prompts_with_failures": len({r["id"] for _, r, _, _ in failures})}))
for index, row, error, frames in failures:
    text = row["text"]
    print("=" * 100)
    print(json.dumps({"index": index, "prompt": row["id"], "sample": row["sample_index"], "finish": row["finish_reason"], "tokens": len(row["token_ids"]), "chars": len(text),
                      "error": type(error).__name__, "message": str(error)[:200], "frames": frames,
                      "boxed": text.count("\\boxed"), "asy": text.count("[asy]"), "dollars": text.count("$"), "open_display": text.count("\\["), "close_display": text.count("\\]"),
                      "open_paren_math": text.count("\\("), "close_paren_math": text.count("\\)"), "newlines": text.count("\n")}))
    print("HEAD:", repr(text[:700]))
    print("TAIL:", repr(text[-500:]))
