from collections import Counter
import json
from pathlib import Path
import sys
from datasets import load_from_disk
from core.vendor.vineppo.math_answer_exctraction import extract_math_answer, split_math_answers, strip_string

data = load_from_disk(sys.argv[1])
for split, ds in data.items():
    counts = Counter()
    exceptional = []
    for i,row in enumerate(ds):
        full = tuple(extract_math_answer(row["problem"],row["solution"]))
        final = tuple(split_math_answers(row["problem"], [strip_string(row["answer"])]))
        counts["same" if full == final else "different"] += 1
        if not all(full) or not full or not all(final) or not final:
            exceptional.append((i, row["unique_id"], row["answer"], full, final, row["solution"][-340:]))
        if len(set(full)) > 1:
            counts["multi-full"] += 1
        if not full or not all(full):
            counts["bad-full"] += 1
        if not final or not all(final):
            counts["bad-final"] += 1
    print(split,dict(counts),flush=True)
    for row in exceptional:
        print(row,flush=True)
