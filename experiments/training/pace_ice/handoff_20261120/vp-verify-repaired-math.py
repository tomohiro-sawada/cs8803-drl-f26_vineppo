from collections import Counter
from pathlib import Path
import sys
from datasets import load_from_disk
from experiments.training.run import load_examples
from core.verifier import _math_references, verify_response

path = Path(sys.argv[1])
examples, provenance = load_examples(path, "math")
train = load_from_disk(str(path))["train"]
expected = {1740: "Saturday", 3512: "4", 4867: "9", 6075: "2", 7318: "0", 11385: "0"}
changed = []
for index, (example,row) in enumerate(zip(examples, train, strict=True)):
    assert example.id == f"math:train:{index}" and example.question == row["problem"].strip()
    refs = _math_references(example.question, example.reference_answer, True)
    if example.reference_answer == row["solution"]:
        continue
    assert refs == (expected[index],)
    assert verify_response(r"\boxed{"+expected[index]+"}", example.reference_answer,
                           task="math", question=example.question, reference_is_solution=True).correct
    changed.append(index)
assert set(changed) == expected.keys()
print("validated", len(examples), "train references; changed", changed, flush=True)
print("normalized training SHA256", provenance["normalized_sha256"], flush=True)
