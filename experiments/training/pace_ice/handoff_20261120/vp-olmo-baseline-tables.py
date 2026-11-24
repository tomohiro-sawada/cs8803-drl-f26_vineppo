"""Print the OLMo baseline tables for the 2026-11-24 report.

    python vp-olmo-baseline-tables.py RUNS_DIR [TASK:ARM:RUN_NAME ...]

RUNS_DIR is .../training/pace_ice. Reads the eight OLMo runs' frozen test
records and training metrics. Every percentage averages 16 rounds per held-out
question; intervals cluster by question. "Lenient" is the reading defined in
scripts/diagnostics/regrade_evaluation.py and is printed for GSM8K only.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.diagnostics.regrade_evaluation import (
    interval,
    lenient,
    load,
    question_means,
)

RUNS = {
    "math": {
        "SFT": "handoff-olmo3-math-sft-04",
        "GRPO": "handoff-olmo3-math-grpo-50-s42-10",
        "RLOO": "handoff-olmo3-math-rloo-50-s42-20",
        "RestEM": "handoff-olmo3-math-restem-match-s42-20",
    },
    "gsm8k": {
        "SFT": "handoff-olmo3-gsm8k-sft-10",
        "GRPO": "handoff-olmo3-gsm8k-grpo-50-s42-10",
        "RLOO": "handoff-olmo3-gsm8k-rloo-50-s42-20",
        "RestEM": "handoff-olmo3-gsm8k-restem-match-s42-20",
    },
}
# The baseline evaluations record no Slurm job; these are from sacct by job name.
BASELINE_JOBS = {
    "handoff-olmo3-math-sft-04": "7228515",
    "handoff-olmo3-gsm8k-sft-10": "7236066",
}


def read(path):
    return json.loads(Path(path).read_text())


def strict(record):
    return record["correct"]


def gain(run, baseline):
    assert run.keys() == baseline.keys(), "Evaluations cover different questions"
    return interval([run[key] - baseline[key] for key in run])


def clock(seconds):
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def describe(root):
    """Frozen test result, its records, and what training consumed."""
    if (root / "test-complete.json").is_file():
        pointer = read(root / "test-complete.json")
        for key in ("result", "selection"):
            digest = hashlib.sha256(Path(pointer[key]).read_bytes()).hexdigest()
            assert digest == pointer[f"{key}_sha256"], f"{key} changed: {root}"
        selection = read(pointer["selection"])
        validation = selection["best"]["accuracy"]
    else:
        pointer, validation = read(root / "COMPLETE"), None
    result = read(pointer["result"])
    assert result["complete"] and result["rounds_completed"] == 16
    assert result["split"] == "test"
    records = load(Path(pointer["result"]).parent)
    assert len(records) == result["total_count"] == 16 * result["num_examples"]
    assert sum(map(strict, records)) == result["total_correct"]
    metrics = [
        json.loads(line)
        for path in sorted(root.glob("attempt-*/metrics.jsonl"))
        for line in path.open()
    ]
    # Every allocation of the run: training attempts plus the final-test link.
    jobs = {read(path)["slurm_job_id"] for path in root.glob("attempt-*/metadata.json")}
    jobs |= (
        {read(root / "status.json")["job"]} if metrics else {BASELINE_JOBS[root.name]}
    )
    rows = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(sorted(jobs))]
        + ["-o", "JobIDRaw,State,ElapsedRaw"],
        text=True,
    ).splitlines()
    assert {row.split("|")[0] for row in rows} == jobs, f"Missing accounting: {jobs}"
    assert all(row.split("|")[1] == "COMPLETED" for row in rows), rows
    return {
        "result": result,
        "records": records,
        "validation": validation,
        "iterations": len(metrics),
        "samples": sum(row["primary_continuations"] for row in metrics),
        "tokens": sum(row["sampled_tokens"] for row in metrics),
        "seconds": sum(int(row.split("|")[2]) for row in rows),
        "jobs": sorted(jobs),
        "metrics": metrics,
        # One W&B run per allocation: the training curves, or the baseline test.
        "wandb": [
            read(path)["url"] for path in sorted(root.glob("attempt-*/wandb.json"))
        ]
        or [result["evaluation_run"]["wandb"]["url"]],
    }


HEALTH = {
    "Reward": ("reward_mean", 1),
    "Clip (%)": ("clip_frac", 100),
    "Grad norm": ("grad_norm", 1),
    "Truncated (%)": ("truncation_fraction", 100),
    "Length": ("response_length_mean", 1),
    "Accepted per round": ("restem/accepted_episodes", 1),
}


def health(metrics):
    """Mean of each logged quantity over the run; a dash where an arm lacks it."""
    cells = []
    for key, scale in HEALTH.values():
        values = [row[key] for row in metrics if key in row]
        cells.append(f"{scale * sum(values) / len(values):.3f}" if values else "—")
    return cells


def main(runs_dir, *additional_runs):
    runs_dir = Path(runs_dir)
    runs = {task: arms.copy() for task, arms in RUNS.items()}
    for argument in additional_runs:
        task, arm, name = argument.split(":")
        assert arm not in runs[task] and Path(name).name == name, argument
        runs[task][arm] = name
    for task, arms in runs.items():
        data = {arm: describe(runs_dir / name) for arm, name in arms.items()}
        readings = {"strict": strict} | (
            {"lenient": lenient} if task == "gsm8k" else {}
        )
        scores = {
            (arm, name): question_means(data[arm]["records"], grade)
            for arm in data
            for name, grade in readings.items()
        }
        print(
            f"\n### {task.upper()}: {data['SFT']['result']['num_examples']} questions"
        )
        header = ["Arm", "Selected", "Validation"]
        for name in readings:
            header += [f"Pass@1, {name} [95% CI]", f"Gain over SFT, {name}"]
        header += ["Samples", "Sampled tokens", "GPU wall", "W&B"]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for arm, run in data.items():
            trained = arm != "SFT"
            row = [
                arm,
                str(run["result"]["checkpoint"]["iteration"]) if trained else "SFT",
                f"{100 * run['validation']:.2f}%" if trained else "—",
            ]
            for name in readings:
                row.append(
                    "{:.2f}% [{:.2f}, {:.2f}]".format(
                        *interval(list(scores[arm, name].values()))
                    )
                )
                row.append(
                    "{:+.2f} [{:+.2f}, {:+.2f}]".format(
                        *gain(scores[arm, name], scores["SFT", name])
                    )
                    if trained
                    else "—"
                )
            row += [
                f"{run['samples']:,}" if trained else "—",
                f"{run['tokens']:,}" if trained else "—",
                clock(run["seconds"]),
                ", ".join(f"[Run]({url})" for url in run["wandb"]),
            ]
            print("| " + " | ".join(row) + " |")
        print(f"\nPaired differences between trained arms, {task.upper()}:\n")
        print("| Comparison | " + " | ".join(readings) + " |")
        print("|" + "---|" * (1 + len(readings)))
        trained = [arm for arm in data if arm != "SFT"]
        for index, first in enumerate(trained):
            for second in trained[index + 1 :]:
                cells = [
                    "{:+.2f} [{:+.2f}, {:+.2f}]".format(
                        *gain(scores[first, name], scores[second, name])
                    )
                    for name in readings
                ]
                print(f"| {first} minus {second} | " + " | ".join(cells) + " |")
        print(f"\nTraining health, {task.upper()} (mean over all iterations):\n")
        print(
            "| Arm | Iterations | "
            + " | ".join(HEALTH)
            + " | Validation by iteration |"
        )
        print("|" + "---|" * (3 + len(HEALTH)))
        for arm in trained:
            metrics = data[arm]["metrics"]
            curve = ", ".join(
                f"{row['iteration']} → {100 * row['eval/validation/pass_at_1']:.2f}"
                for row in metrics
                if "eval/validation/pass_at_1" in row
            )
            cells = [arm, str(len(metrics)), *health(metrics), curve]
            print("| " + " | ".join(cells) + " |")
        if task == "gsm8k":
            print("\nAnswer format, GSM8K:\n")
            print(
                "| Arm | Responses | With a `####` answer | Truncated | Right final number, marked wrong |"
            )
            print("|---|---:|---:|---:|---:|")
            for arm, run in data.items():
                records = run["records"]
                found = sum(
                    record["extracted_answer"] is not None for record in records
                )
                cut = sum(record["finish_reason"] == "length" for record in records)
                missed = sum(lenient(r) and not r["correct"] for r in records)
                print(
                    f"| {arm} | {len(records):,} | {found:,} ({100 * found / len(records):.2f}%) | {cut} | {missed:,} |"
                )


if __name__ == "__main__":
    main(*sys.argv[1:])
