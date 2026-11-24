from collections import defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, stdev
import sys

def read(path):
    return json.loads(path.read_text())

def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)

def ci(samples):
    avg = mean(samples)
    half = 1.96 * stdev(samples) / len(samples) ** 0.5
    return [avg, avg - half, avg + half]

def extract(root):
    if (root / "test-complete.json").is_file():
        pointer = read(root / "test-complete.json")
        for key in ("result", "selection"):
            file = Path(pointer[key])
            assert hashlib.sha256(file.read_bytes()).hexdigest() == pointer[f"{key}_sha256"]
    elif (root / "final-evaluation.json").is_file():
        pointer = read(root / "final-evaluation.json")
    elif root.name.startswith("handoff-olmo3-") and "-sft-" in root.name:
        pointer = {"result": read(root / "COMPLETE")["result"]}
    else:
        pointer = {"result": str(root / "result.json")}
    result = Path(pointer["result"])
    test = read(result)
    assert test["complete"] and test["rounds_completed"] == 16 and test["split"] == "test"
    validation = []
    if "selection" in pointer:
        manifest = read(Path(pointer["selection"]))
        assert manifest["best"]["checkpoint"] == test["checkpoint"]
        validation = [(row["checkpoint"]["iteration"], row["accuracy"]) for row in manifest["candidates"]]
        if manifest["baseline"]:
            validation.insert(0, (0, manifest["baseline"]["accuracy"]))
    by_question = defaultdict(list)
    for round_dir in sorted(result.parent.glob("round-*")):
        for row in lines(round_dir / "records.jsonl"):
            by_question[row["example_id"]].append(int(row["correct"]))
    assert len(by_question) == test["num_examples"] and all(len(q) == 16 for q in by_question.values())
    scores = {k: mean(v) for k, v in by_question.items()}
    assert abs(mean(scores.values()) - test["accuracy"]) < 1e-12
    lineage = [root]
    while (lineage[0] / "chain-plan.json").is_file():
        options = read(lineage[0] / "chain-plan.json")["options"]
        args = dict(zip(options[::2], options[1::2], strict=True))
        if "--resume" not in args:
            break
        earlier = Path(args["--resume"]).resolve().parents[2]
        assert earlier.parent == root.parent and earlier not in lineage
        lineage.insert(0, earlier)
    training = [read(p) for r in lineage for p in sorted(r.glob("attempt-*/metadata.json"))]
    if not training and (root / "metadata.json").is_file():
        training = [read(root / "metadata.json")]
    rows = {}
    retrained = 0
    for p in [p for r in lineage for p in sorted(r.glob("attempt-*/metrics.jsonl")) + list(r.glob("metrics.jsonl"))]:
        if not p.stat().st_size:
            continue
        start = read(p.with_name("metadata.json")).get("start_iteration", 0)
        for i in [i for i in rows if i > start]:
            retrained += rows.pop(i)["sampled_tokens"]
        for row in lines(p):
            i = row["iteration"]
            assert i not in rows and i > start
            rows[i] = row
    targets = next((m for m in reversed(training) if "trainer_config" in m), {})
    finish = targets.get("cli", {}).get("num_iterations", 0)
    tracking = [read(p)["url"] for r in lineage for p in sorted(r.glob("attempt-*/wandb.json")) + list(r.glob("wandb.json"))]
    if not tracking and (result.parent / "wandb.json").is_file():
        tracking = [read(result.parent / "wandb.json")["url"]]
    recent = [rows[i] for i in sorted(rows)[-25:]]
    quality_keys = (
        "reward_mean", "reference_kl", "clip_frac", "grad_norm", "truncation_fraction",
        "response_length_mean", "raw_advantage_token_zero_fraction",
        "raw_advantage_token_std",
        "group_zero_reward_variance_fraction", "critic/value_loss",
        "vineppo/mc_prefix_value_mean", "vineppo/mc_prefix_value_std",
        "restem/accepted_episodes",
    )
    def summarize(rows, keys):
        return {k: mean(values) for k in keys if (values := [r[k] for r in rows if k in r])}
    resume_points = sorted(m["start_iteration"] for m in training if m.get("start_iteration"))
    resume_changes = {}
    for i in resume_points:
        if i in rows and i + 1 in rows:
            resume_changes[i] = {k: rows[i + 1][k] - rows[i][k] for k in quality_keys if k in rows[i] and k in rows[i + 1]}
    slurm = {m["slurm_job_id"] for m in training if m.get("slurm_job_id")}
    if (root / "status.json").is_file():
        slurm.add(read(root / "status.json")["job"])
    rec = {
        "task": test["task"], "test_pass_at_1": ci(scores.values()),
        "selection_iteration": test["checkpoint"]["iteration"] if "selection" in pointer else None,
        "lineage": [r.name for r in lineage],
        "slurm": sorted(slurm), "resume_iterations": resume_points,
        "changes_across_resumes": resume_changes, "validation": validation,
        "training_health_mean": summarize(rows.values(), quality_keys),
        "training_health_tail_mean": summarize(recent, quality_keys),
        "wandb": tracking, "logged_iterations": len(rows), "expected_iterations": finish,
        "primary_tokens": sum(r["primary_tokens"] for r in rows.values()),
        "primary_continuations": sum(r["primary_continuations"] for r in rows.values()),
        "mc_tokens": sum(r["mc_tokens"] for r in rows.values()),
        "retried_recorded_tokens": retrained,
        "phase_seconds": {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()), {}) if k.endswith("_seconds")},
        "kl_coef": targets.get("trainer_config", {}).get("kl_coef"),
    }
    match = re.search(r"(?:seed|-s)(\d+)", root.name)
    rec["seed"] = int(match.group(1)) if match else 42
    return rec, scores

def main(root):
    patterns = ["long-*", "sft-baseline-test-rhomath-*", "handoff-rho-*", "handoff-olmo3-*-sft-*", "handoff-olmo3-*-grpo-50-*"]
    folders = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_dir()})
    folders = [p for p in folders if any((p / name).is_file() for name in ("COMPLETE", "final-evaluation.json", "result.json"))]
    runs = {p.name: extract(p) for p in folders}
    pairs = {}
    for (an, (a, sa)), (bn, (b, sb)) in combinations(runs.items(), 2):
        if a["task"] == b["task"] and a["seed"] == b["seed"]:
            assert sa.keys() == sb.keys()
            pairs[f"{an} minus {bn}"] = ci([sa[q] - sb[q] for q in sa])
    print(json.dumps({"runs": {key: result for key, (result, _) in runs.items()}, "same_seed_differences": pairs}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
