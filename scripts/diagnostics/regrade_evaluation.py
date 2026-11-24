"""Re-grade stored evaluation records with the current verifier.

    python -m scripts.diagnostics.regrade_evaluation EVAL_DIR [BASELINE_EVAL_DIR]

Each directory holds round-*/records.jsonl from experiments/training/evaluate.py.
Prints how many stored grades the current verifier reproduces. For GSM8K it adds
a lenient reading that separates answer format from arithmetic: the last number
in the response equals the reference, ignoring "$", thousands separators and
trailing decimal zeros. The official grader compares the text after the last
"####" with the reference and strips separators from the response only, so a
reference that contains one never matches. With a baseline directory, also
prints paired differences in points with question-clustered 95% intervals.
"""

import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from core.verifier import verify_response

NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def number(text):
    return Decimal(text.replace("$", "").replace(",", ""))


def lenient(record):
    found = NUMBER.findall(record["response"])
    return bool(found) and number(found[-1]) == number(record["reference_answer"])


def regrade(record):
    return verify_response(
        record["response"],
        record["reference_answer"],
        task=record["task"],
        question=record["question"],
        reference_is_solution=record["task"] == "math",
    ).correct


def load(directory):
    files = sorted(Path(directory).glob("round-*/records.jsonl"))
    assert files, f"No evaluation records under {directory}"
    return [json.loads(line) for file in files for line in file.open()]


def report(records):
    with ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        for record, correct in zip(records, pool.map(regrade, records), strict=True):
            record["regraded"] = correct
    summary = {
        "records": len(records),
        "stored_correct": sum(r["correct"] for r in records),
        "regraded_correct": sum(r["regraded"] for r in records),
        "grades_changed": sum(r["correct"] != r["regraded"] for r in records),
        "truncated": sum(r["finish_reason"] == "length" for r in records),
    }
    if records[0]["task"] == "math":
        return summary
    assert all(lenient(r) for r in records if r["correct"]), "Lenient must admit strict"
    missing = [r for r in records if r["extracted_answer"] is None]
    comma = [r for r in records if "," in r["reference_answer"]]
    return summary | {
        "extracted": len(records) - len(missing),
        "lenient_correct": sum(map(lenient, records)),
        "unextracted_boxed": sum("\\boxed" in r["response"] for r in missing),
        "unextracted_lenient_correct": sum(map(lenient, missing)),
        "separator_reference": {
            "records": len(comma),
            "stored_correct": sum(r["correct"] for r in comma),
            "lenient_correct": sum(map(lenient, comma)),
        },
    }


def question_means(records, grade):
    scores = defaultdict(list)
    for record in records:
        scores[record["example_id"]].append(float(grade(record)))
    return {key: sum(values) / len(values) for key, values in scores.items()}


def interval(values):
    """Mean and 95% interval in points, one value per question."""
    mean = sum(values) / len(values)
    spread = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    error = sqrt(spread / len(values))
    return [round(100 * x, 2) for x in (mean, mean - 1.96 * error, mean + 1.96 * error)]


def paired(treated, baseline, grade):
    a, b = question_means(treated, grade), question_means(baseline, grade)
    assert a.keys() == b.keys(), "Evaluations cover different questions"
    return interval([a[key] - b[key] for key in a])


def main(directories):
    assert 1 <= len(directories) <= 2, __doc__
    loaded = [load(directory) for directory in directories]
    for directory, records in zip(directories, loaded, strict=True):
        print(json.dumps({"directory": directory} | report(records)), flush=True)
    if len(loaded) == 2:
        grades = {"stored": lambda r: r["correct"], "regraded": lambda r: r["regraded"]}
        if loaded[0][0]["task"] == "gsm8k":
            grades["lenient"] = lenient
        gains = {name: paired(*loaded, grade) for name, grade in grades.items()}
        print(json.dumps({"first_minus_second_points": gains}))


if __name__ == "__main__":
    main(sys.argv[1:])
