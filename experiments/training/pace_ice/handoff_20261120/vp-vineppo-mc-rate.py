"""Print a completed VinePPO run's Monte Carlo sampling cost from its metrics.

    python vp-vineppo-mc-rate.py RUN_ROOT
"""

import json
import sys
from pathlib import Path

SUMS = (
    "primary_continuations", "primary_tokens", "mc_continuations", "mc_tokens",
    "vineppo/step_count", "primary_generation_seconds", "mc_generation_seconds",
    "mc_estimation_seconds", "iteration_seconds",
)

root = Path(sys.argv[1])
assert (root / "COMPLETE").is_file(), root
rows = [
    json.loads(line)
    for path in sorted(root.glob("attempt-*/metrics.jsonl"))
    for line in path.open()
]
iterations = [row["iteration"] for row in rows]
assert iterations == sorted(set(iterations)), iterations
summary = {"run": root.name, "logged_iterations": len(rows)}
summary |= {key: sum(row[key] for row in rows) for key in SUMS}
summary["mc_tokens_per_primary_token"] = summary["mc_tokens"] / summary["primary_tokens"]
summary["mc_tokens_per_generation_second"] = (
    summary["mc_tokens"] / summary["mc_generation_seconds"]
)
print(json.dumps(summary))
