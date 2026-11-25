from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import runpy
from statistics import mean, stdev
import subprocess
import sys

log = Path(sys.argv[1]).read_text()
data, _ = json.JSONDecoder().raw_decode(log[log.index("\n{") + 1:])
runs = data["runs"]
archive_outputs = subprocess.check_output([
    "rg", "--no-heading", "--with-filename", "-g", "*.out",
    r"^verified training/pace_ice/[^/:]+: \d+ files, \d+ bytes$",
    "artifacts/training/pace_ice/slurm",
], text=True)
full_archives = {}
for row in archive_outputs.splitlines():
    m = re.fullmatch(r".*-(\d+)\.out:verified training/pace_ice/([^/:]+): \d+ files, \d+ bytes", row)
    assert m, row
    job, name = m.groups()
    full_archives[name] = job
partial = {
    "handoff-rho-ppo-200-s43-07": "Winner and test",
    "handoff-rho-ppo-200-s44-07": "Winner and test",
    "handoff-olmo3-math-grpo-50-s42-10": "Winner and test",
    "handoff-rho-vineppo-650-s42-11": "Winner",
}
extra_jobs = runpy.run_path(
    str(Path(__file__).with_name("vp-training-analysis.py"))
)["EXTRA_SLURM_JOBS"]
ids = set(i for name, row in runs.items() for i in row["slurm"] + list(extra_jobs.get(name, ())))
assert ids
account = subprocess.check_output([
    "sacct", "-X", "-n", "-P", "-D", "-S", "2026-11-16", "-j", ",".join(sorted(ids)),
    "-o", "JobIDRaw,State,ElapsedRaw,Submit,Eligible,Start,End",
], text=True)
accounting = {}
for line in account.splitlines():
    job, state, elapsed, submit, eligible, start, end = line.split("|")
    accounting[job] = (state, int(elapsed), submit, eligible, start, end)
assert set(accounting) == ids, (ids - accounting.keys(), accounting.keys() - ids)

def time_str(seconds):
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"

def duration(first, last):
    return round((datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds())

def times(name, row):
    rows = [accounting[i] for i in set(row["slurm"]) | set(extra_jobs.get(name, ())) if accounting[i][4] not in ("Unknown", "None")]
    assert rows, name
    allocated = sum(r[1] for r in rows)
    first = min(rows, key=lambda r: r[4])
    return allocated, duration(first[4], max(r[5] for r in rows)), duration(first[2], first[4])

def arm(name):
    if name.startswith("sft-baseline-"):
        return "RhoMath SFT"
    if name.startswith("long-"):
        _, method, *_ = name.split("-")
        count = re.search(r"-(\d+)iter-", name).group(1)
        main = not ("seed42" in name and name.endswith("-200iter-01"))
        desc = f"{'VinePPO' if method == 'vineppo' else method.upper()} {count}"
        desc += " (1e-4 ablation)" if method == "grpo" else " (no dedup)"
        return desc if main else f"{desc}, earlier replicate"
    if "-olmo3-" in name:
        _, _, task, method, *_ = name.split("-")
        return f"OLMo {task.upper()} {method.upper()}" + (" 50" if method == "grpo" else "")
    if "-restem-" in name:
        return "RestEM matched"
    _, _, method, count, *_ = name.split("-")
    return f"{'VinePPO' if method == 'vineppo' else method.upper()} {count}" + (" (dedup)" if method == "vineppo" else "")

arms = defaultdict(list)
ordering = [
    "RhoMath SFT", "GRPO 200 (1e-4 ablation)", "VinePPO 200 (no dedup)",
    "GRPO 500 (1e-4 ablation)", "GRPO 200 (1e-4 ablation), earlier replicate",
    "VinePPO 200 (no dedup), earlier replicate", "RLOO 200", "GRPO 200",
    "PPO 200", "VinePPO 200 (dedup)", "RestEM matched", "RLOO 650",
    "GRPO 650", "PPO 650", "VinePPO 650 (dedup)", "OLMo MATH SFT",
    "OLMo MATH GRPO 50", "OLMo GSM8K SFT", "OLMo GSM8K GRPO 50",
]
for name, row in runs.items():
    arms[arm(name)].append((name, row))
assert not (arms.keys() - set(ordering))
print("| Run | Seed | Winner | Pass@1 [95% CI] | Primary / MC training tokens | GPU wall | Start→end | Initial queue | W&B | Verified bucket copy |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
for label in ordering:
    for name, row in sorted(arms[label], key=lambda p: p[1]["seed"]):
        elapsed, wall, queue = map(time_str, times(name, row))
        accuracy, low, high = [f"{v * 100:.2f}" for v in row["test_pass_at_1"]]
        mark = "≥" if row["logged_iterations"] != row["expected_iterations"] else ""
        mc_mark = mark if row["mc_tokens"] else ""
        sampled = f"{mark}{row['primary_tokens']:,} / {mc_mark}{row['mc_tokens']:,}" if row["expected_iterations"] else "—"
        winner = row["selection_iteration"] or "SFT"
        tracking = row["wandb"][-1]
        complete = [full_archives[source] for source in row["lineage"] if source in full_archives]
        archive = f"Full ({', '.join(complete)})" if len(complete) == len(row["lineage"]) else partial.get(name, "Pending")
        print(f"| {label} | {row['seed']} | {winner} | {accuracy}% [{low}, {high}] | {sampled} | {elapsed} | {wall} | {queue} | [Run]({tracking}) | {archive} |")
print()
for label in ordering:
    group = arms[label]
    if len(group) > 1:
        scores = [r["test_pass_at_1"][0] * 100 for _, r in group]
        average = mean(scores)
        print(f"{label}: n={len(group)}, mean {average:.2f}%, sample SD {stdev(scores):.2f} pp.")
print()
print("| Arm | Minutes in recorded iterations / mean per run | Main phases (% of measured iteration time) |")
print("|---|---:|---|")
for label in ordering:
    group = arms[label]
    if not group or not group[0][1]["phase_seconds"]:
        continue
    phases = defaultdict(float)
    total = 0
    for _, row in group:
        total += row["phase_seconds"]["iteration_seconds"]
        for k,v in row["phase_seconds"].items():
            if k not in ("iteration_seconds", "mc_generation_seconds"):
                phases[k.removesuffix("_seconds")] += v
    breakdown = ", ".join(f"{k.replace('_', ' ')} {100*v/total:.1f}%" for k,v in phases.items() if v/total >= .001)
    print(f"| {label} | {total/60/len(group):.1f} | {breakdown} |")
print()
print("| Seed-42 validation checkpoints | Iteration → Pass@1 (%) |")
print("|---|---|")
for label in ordering:
    selected = [r for _, r in arms[label] if r["seed"] == 42 and r["validation"]]
    if selected:
        row, = selected
        values = ", ".join(f"{i} → {100*val:.2f}" for i,val in row["validation"])
        print(f"| {label} | {values} |")
print()
print("| Arm | Reward | Ref. KL | Clip (%) | Grad norm | Truncated (%) | Length | Zero adv. (%) | Zero-variance groups (%) | Value loss | MC value mean / SD |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for label in ordering:
    group = [r["training_health_tail_mean"] for _, r in arms[label] if r["training_health_tail_mean"]]
    if not group:
        continue
    def col(key, factor=1):
        vals = [g[key] for g in group if key in g]
        return f"{mean(vals)*factor:.3f}" if vals else "—"
    v = [
        col("reward_mean"), col("reference_kl"), col("clip_frac",100), col("grad_norm"),
        col("truncation_fraction",100), col("response_length_mean"),
        col("raw_advantage_token_zero_fraction",100), col("group_zero_reward_variance_fraction",100),
        col("critic/value_loss"), f"{col('vineppo/mc_prefix_value_mean')} / {col('vineppo/mc_prefix_value_std')}",
    ]
    print(f"| {label} | {' | '.join(v)} |")
print()
pairs = data["same_seed_differences"]
def paired(a, b):
    assert a != b and runs[a]["seed"] == runs[b]["seed"]
    if a < b:
        return pairs[f"{a} minus {b}"]
    avg, lo, hi = pairs[f"{b} minus {a}"]
    return [-avg,-hi,-lo]
comparisons = (
    ("VinePPO 200 (dedup)", "GRPO 200"), ("VinePPO 200 (dedup)", "RLOO 200"),
    ("VinePPO 200 (dedup)", "PPO 200"), ("VinePPO 200 (dedup)", "VinePPO 200 (no dedup)"),
    ("GRPO 200", "GRPO 200 (1e-4 ablation)"), ("VinePPO 200 (no dedup)", "GRPO 500 (1e-4 ablation)"),
    ("VinePPO 200 (dedup)", "GRPO 500 (1e-4 ablation)"), ("RestEM matched", "RhoMath SFT"),
    ("VinePPO 650 (dedup)", "GRPO 650"), ("VinePPO 650 (dedup)", "RLOO 650"),
    ("VinePPO 650 (dedup)", "PPO 650"), ("OLMo MATH GRPO 50", "OLMo MATH SFT"),
    ("OLMo GSM8K GRPO 50", "OLMo GSM8K SFT"),
)
print("| Comparison | Seed 42 | Seed 43 | Seed 44 | Mean | 3-seed t interval |")
print("|---|---:|---:|---:|---:|---:|")
for la, lb in comparisons:
    lookup = {v["seed"]: n for n,v in arms[lb]}
    matched = {r["seed"]: paired(n, lookup[r["seed"]]) for n,r in arms[la] if r["seed"] in lookup}
    if not matched:
        continue
    values = []
    for seed in (42,43,44):
        if seed in matched:
            avg, lo, hi = [x*100 for x in matched[seed]]
            values.append(f"{avg:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        else:
            values.append("—")
    changes = [x[0]*100 for x in matched.values()]
    center = mean(changes)
    if len(changes) == 3:
        margin = 4.3026527299 * stdev(changes) / len(changes)**0.5
        t_result = f"[{center-margin:+.2f}, {center+margin:+.2f}]"
    else:
        t_result = "—"
    print(f"| {la} minus {lb} | {' | '.join(values)} | {center:+.2f} | {t_result} |")
