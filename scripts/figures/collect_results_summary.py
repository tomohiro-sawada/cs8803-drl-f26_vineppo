"""Collect the numbers behind the results-summary figures into one JSON file.

    python -m scripts.figures.collect_results_summary RUNS_DIR OUT.json

RUNS_DIR is .../training/pace_ice. Each run is read through the script that
produced the 2026-11-22 report tables, so the figures and those tables share
one source: the frozen 16-round test, the validation manifest and the training
metrics. Percentages are in points and intervals cluster by question. "Lenient"
is the GSM8K reading defined in scripts/diagnostics/regrade_evaluation.py.
A validation point's hours sum iteration_seconds over the logged iterations up
to it, so they include sampling, updates, checkpoints and earlier validations.
"""

import json
import runpy
import sys
from pathlib import Path

from scripts.diagnostics.regrade_evaluation import (
    interval,
    lenient,
    load,
    question_means,
)

ANALYSIS = "experiments/training/pace_ice/handoff_20261120/vp-training-analysis.py"
extract = runpy.run_path(str(Path(__file__).resolve().parents[2] / ANALYSIS))["extract"]


RUNS = {
    "rhomath-gsm8k": {
        "SFT": ["sft-baseline-test-rhomath-gsm8k-vllm-h100-01"],
        "RestEM matched": ["handoff-rho-restem-match-s42-03"],
        "RLOO 200": [f"handoff-rho-rloo-200-s{seed}-03" for seed in (42, 43, 44)],
        "GRPO 200": [f"handoff-rho-grpo-200-s{seed}-03" for seed in (42, 43, 44)],
        "PPO 200": [
            "handoff-rho-ppo-200-s42-03",
            "handoff-rho-ppo-200-s43-07",
            "handoff-rho-ppo-200-s44-07",
        ],
        "VinePPO 200": [f"handoff-rho-vineppo-200-s{seed}-03" for seed in (42, 43, 44)],
        "RLOO 650": ["handoff-rho-rloo-650-s42-07"],
        "GRPO 650": ["handoff-rho-grpo-650-s42-07"],
        "PPO 650": ["handoff-rho-ppo-650-s42-07"],
        "VinePPO 650": ["handoff-rho-vineppo-650-s42-11"],
    },
    "olmo-math": {
        "SFT": ["handoff-olmo3-math-sft-04"],
        "RestEM matched": ["handoff-olmo3-math-restem-match-s42-20"],
        "RLOO 50": ["handoff-olmo3-math-rloo-50-s42-20"],
        "GRPO 50": ["handoff-olmo3-math-grpo-50-s42-10"],
    },
    "olmo-gsm8k": {
        "SFT": ["handoff-olmo3-gsm8k-sft-10"],
        "RestEM matched": ["handoff-olmo3-gsm8k-restem-match-s42-20"],
        "RLOO 50": ["handoff-olmo3-gsm8k-rloo-50-s42-20"],
        "GRPO 50": ["handoff-olmo3-gsm8k-grpo-50-s42-10"],
    },
}


def logged(runs_dir, lineage):
    """Metric rows by iteration over a resume chain; a later attempt wins."""
    rows = {}
    for name in lineage:
        for path in sorted((runs_dir / name).glob("attempt-*/metrics.jsonl")):
            for line in path.open():
                row = json.loads(line)
                rows[row["iteration"]] = row
    return rows


def test_records(root):
    marker = root / "test-complete.json"
    pointer = json.loads(
        (marker if marker.is_file() else root / "COMPLETE").read_text()
    )
    return load(Path(pointer["result"]).parent)


def describe(runs_dir, task, name):
    run, scores = extract(runs_dir / name)
    rows = logged(runs_dir, run["lineage"])
    assert len(rows) == run["logged_iterations"], name

    def hours(iteration):
        seconds = (r["iteration_seconds"] for i, r in rows.items() if i <= iteration)
        return round(sum(seconds) / 3600, 3)

    record = {
        "run": name,
        "seed": run["seed"],
        "selected": run["selection_iteration"],
        "test": interval(list(scores.values())),
        "validation": [[i, round(100 * a, 2), hours(i)] for i, a in run["validation"]],
        "primary_tokens": run["primary_tokens"],
        "mc_tokens": run["mc_tokens"],
        "phase_seconds": run["phase_seconds"],
        "wandb": run["wandb"],
        "reward": [[i, rows[i]["reward_mean"]] for i in sorted(rows)],
    }
    if task == "olmo-gsm8k":
        grades = question_means(test_records(runs_dir / name), lenient)
        assert grades.keys() == scores.keys(), name
        record["test_lenient"] = interval(list(grades.values()))
    return record


def main(runs_dir, out):
    runs_dir = Path(runs_dir)
    data = {
        task: {
            arm: [describe(runs_dir, task, name) for name in names]
            for arm, names in arms.items()
        }
        for task, arms in RUNS.items()
    }
    Path(out).write_text(json.dumps(data, separators=(",", ":")) + "\n")
    for task, arms in data.items():
        for arm, runs in arms.items():
            for run in runs:
                print(task, arm, run["seed"], run["selected"], run["test"])


if __name__ == "__main__":
    main(*sys.argv[1:])
