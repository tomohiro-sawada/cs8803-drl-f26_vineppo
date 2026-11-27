"""Collect the numbers behind the results-summary figures into one JSON file.

    python -m scripts.figures.collect_results_summary RUNS_DIR OUT.json

RUNS_DIR is .../training/pace_ice. Each run is read through the script that
produced the 2026-11-22 report tables, so the figures and those tables share
one source: the frozen 16-round test, the validation manifest and the training
metrics. Test scores are unrounded points with intervals clustered by question. "Lenient"
is the GSM8K reading defined in scripts/diagnostics/regrade_evaluation.py.
A validation point's hours sum iteration_seconds over the logged iterations up
to it, so they include sampling, updates, checkpoints and earlier validations.
Its training accuracy is the mean stopped_correctness_mean of the iterations
since the previous validation point: correctness of the stopped training
responses, sampled at temperature 1 on training questions.

Besides the runs, the file holds: "paired", same-seed differences between
arms of one budget and against SFT, in points over the test questions;
"value_replay", value_estimate_analysis.py's summary of each replay; and
"sft_validation_protocol", the unadapted models scored with the validation
protocol on the validation split and on 373 random training questions.
"""

import json
import runpy
import sys
from itertools import pairwise
from pathlib import Path

from scripts.diagnostics.regrade_evaluation import lenient, load, question_means
from scripts.diagnostics.value_estimate_analysis import summarize

ANALYSIS = "experiments/training/pace_ice/handoff_20261120/vp-training-analysis.py"
analysis = runpy.run_path(str(Path(__file__).resolve().parents[2] / ANALYSIS))
extract, ci = analysis["extract"], analysis["ci"]


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
        "PPO rank-16 200": [
            f"handoff-rho-ppo-lora-critic-rank16-200-s{seed}-30"
            for seed in (42, 43, 44)
        ],
        "PPO rank-64 200": ["handoff-rho-ppo-lora-critic-rank64-200-s42-30"],
        "PPO rank-256 200": ["handoff-rho-ppo-lora-critic-rank256-200-s42-30"],
        "VinePPO 200": [f"handoff-rho-vineppo-200-s{seed}-03" for seed in (42, 43, 44)],
        "RLOO 650": ["handoff-rho-rloo-650-s42-07"],
        "GRPO 650": [
            "handoff-rho-grpo-650-s42-07",
            "handoff-rho-grpo-650-s43-30",
            "handoff-rho-grpo-650-s44-30",
        ],
        "PPO 650": ["handoff-rho-ppo-650-s42-07"],
        "VinePPO 650": [
            "handoff-rho-vineppo-650-s42-11",
            "handoff-rho-vineppo-650-s43-30",
            "handoff-rho-vineppo-650-s44-30",
        ],
    },
    "olmo-math": {
        "SFT": ["handoff-olmo3-math-sft-04"],
        "RestEM matched": ["handoff-olmo3-math-restem-match-s42-20"],
        "RLOO 50": ["handoff-olmo3-math-rloo-50-s42-20"],
        "GRPO 50": ["handoff-olmo3-math-grpo-50-s42-10"],
        "PPO rank-16 50": ["handoff-olmo3-math-ppo-rank16-50-s42-30"],
    },
    "olmo-gsm8k": {
        "SFT": ["handoff-olmo3-gsm8k-sft-10"],
        "RestEM matched": ["handoff-olmo3-gsm8k-restem-match-s42-20"],
        "RLOO 50": ["handoff-olmo3-gsm8k-rloo-50-s42-20"],
        "GRPO 50": ["handoff-olmo3-gsm8k-grpo-50-s42-10"],
        "PPO rank-16 50": ["handoff-olmo3-gsm8k-ppo-rank16-50-s42-30"],
        "VinePPO 50": ["handoff-olmo3-gsm8k-vineppo-50-s42-30"],
    },
}
# Arms added once their run is COMPLETE; until then the summary lists them as pending.
PENDING = {"olmo-math": {"VinePPO 50": ["handoff-olmo3-math-vineppo-50-s42-30"]}}
REPLAYS = {
    "Full critic, seed 43": "full-s43",
    "Rank 16, seed 43": "rank16-s43",
    "Rank 16, seed 42": "rank16-s42",
    "Rank 64, seed 42": "rank64-s42",
    "Rank 256, seed 42": "rank256-s42",
}
SFT_CHECK = {"RhoMath": "rho", "OLMo 3": "olmo3"}


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

    def train_accuracy(start, stop):
        scores = [
            r["stopped_correctness_mean"] for i, r in rows.items() if start < i <= stop
        ]
        return round(100 * sum(scores) / len(scores), 2)

    points = [i for i, _ in run["validation"]]
    record = {
        "run": name,
        "seed": run["seed"],
        "iterations": run["logged_iterations"],
        "health": run["training_health_mean"],
        "health_tail": run["training_health_tail_mean"],
        "selected": run["selection_iteration"],
        "test": [100 * x for x in run["test_pass_at_1"]],
        "validation": [[i, round(100 * a, 2), hours(i)] for i, a in run["validation"]],
        "primary_tokens": run["primary_tokens"],
        "mc_tokens": run["mc_tokens"],
        "phase_seconds": run["phase_seconds"],
        "wandb": run["wandb"],
        "reward": [[i, rows[i]["reward_mean"]] for i in sorted(rows)],
        "train_accuracy": [
            [stop, train_accuracy(start, stop)] for start, stop in pairwise(points)
        ],
    }
    if task == "olmo-gsm8k":
        grades = question_means(test_records(runs_dir / name), lenient)
        assert grades.keys() == scores.keys(), name
        record["test_lenient"] = [100 * x for x in ci(list(grades.values()))]
    return record, scores


def paired(arms, scores):
    """Same-seed differences in points: arms of one budget, and each arm against SFT."""
    result = {}
    for first in arms:
        for second in arms:
            if first == second or "SFT" in first:
                continue
            if "SFT" not in second and (
                first.split()[-1] != second.split()[-1] or first < second
            ):
                continue
            by_seed = {}
            for a in arms[first]:
                for b in arms[second]:
                    if "SFT" in second or a["seed"] == b["seed"]:
                        left, right = scores[a["run"]], scores[b["run"]]
                        assert left.keys() == right.keys(), (a["run"], b["run"])
                        by_seed[a["seed"]] = [
                            100 * x for x in ci([left[q] - right[q] for q in left])
                        ]
            if by_seed:
                result[f"{first} minus {second}"] = by_seed
    return result


def sft_check(runs_dir, model):
    """Validation-protocol Pass@1 of the unadapted model, intervals by question."""
    result = {}
    for which in ("validation", "train373"):
        root = runs_dir / f"sft-validation-protocol-{model}-gsm8k-{which}-30"
        records = load(root / "validation")
        strict = question_means(records, lambda record: record["correct"])
        result[which] = {"strict": [100 * x for x in ci(list(strict.values()))]}
        loose = question_means(records, lenient)
        result[which]["lenient"] = [100 * x for x in ci(list(loose.values()))]
        result[which]["rounds"] = len(records) // len(strict)
    return result


def main(runs_dir, out):
    runs_dir = Path(runs_dir)
    for task, arms in PENDING.items():
        for arm, names in arms.items():
            if all((runs_dir / name / "COMPLETE").is_file() for name in names):
                RUNS[task][arm] = names
    data, scores = {}, {}
    for task, arms in RUNS.items():
        data[task] = {}
        for arm, names in arms.items():
            described = [describe(runs_dir, task, name) for name in names]
            data[task][arm] = [record for record, _ in described]
            scores |= {record["run"]: values for record, values in described}
    data["paired"] = {task: paired(data[task], scores) for task in RUNS}
    data["value_replay"] = {
        label: summarize(
            [
                json.loads(line)
                for line in (
                    runs_dir / f"value-replay-rho-ppo-{name}-i200-30/value-replay.jsonl"
                ).open()
            ]
        )
        for label, name in REPLAYS.items()
    }
    data["sft_validation_protocol"] = {
        label: sft_check(runs_dir, model) for label, model in SFT_CHECK.items()
    }
    Path(out).write_text(json.dumps(data, separators=(",", ":")) + "\n")
    for task in RUNS:
        for arm, runs in data[task].items():
            for run in runs:
                print(task, arm, run["seed"], run["selected"], f"{run['test'][0]:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
