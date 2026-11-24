"""Assign every object in the project bucket a triage tier before thinning it.

    python vp-bucket-triage.py BACKUP_RECEIPT.json PLAN.json

Read-only: lists the bucket and reads the local selection manifests. Tiers 1-4
are deletion candidates, ordered by how little is lost:

1. byte-identical copies of a kept file (the kept copy stays in the bucket);
2. weights of smoke, dry-run, benchmark, pilot and probe runs;
3. full-format non-selected checkpoints of the two earliest seed-42 runs;
4. intermediate PPO critic snapshots.

Tiers 2-4 leave the off-site backup as the only copy, so each such path must be
in BACKUP_RECEIPT.json with the same size and content hash. Everything else is
kept. scripts/remote_sync/delete_planned.py executes the plan.
"""

import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.remote_sync.hf_bucket_sync import BUCKET, RUNS, hf

SCRATCH = str(RUNS.resolve(strict=True))
TRAINING = f"{SCRATCH}/training/pace_ice"
WEIGHTS = (".pt", ".safetensors", ".tmp")
REPORTED = re.compile(
    r"long-(grpo|vineppo)-rhomath-gsm8k-seed\d+-vllm-h100-(200|500)iter-\d+"
    r"|handoff-rho-((rloo|grpo|vineppo|ppo)-(200|650)|restem-match)-s\d+-\d+"
    r"|handoff-olmo3-(math|gsm8k)-(grpo-50|rloo-50|restem-match)-s\d+-\d+"
)


def run_of(path):
    parts = path.split("/")
    parts = parts[1:] if parts[0] == "checkpoint-history" else parts
    assert parts[:2] == ["training", "pace_ice"], path
    return parts[2]


def iteration(path):
    return int(re.search(r"-(\d+)\.pt$", path).group(1))


def lineage(run):
    return re.sub(r"-\d+$", "", run)


def selected_checkpoints():
    markers = glob.glob(f"{TRAINING}/*/test-complete.json")
    markers += glob.glob(f"{TRAINING}/long-*/final-evaluation.json")
    selected = set()
    for marker in markers:
        root = os.path.dirname(marker)
        if not REPORTED.fullmatch(os.path.basename(root)):
            continue
        record = json.loads(Path(marker).read_text())
        selection = record.get("selection", f"{root}/selection-frozen.json")
        checkpoint = json.loads(Path(selection).read_text())["best"]["checkpoint"]
        path = os.path.realpath(checkpoint["path"])
        assert path.startswith(SCRATCH + "/"), path
        selected.add(path[len(SCRATCH) + 1 :])
    return selected


def main(backup_receipt, output):
    receipt = json.loads(Path(backup_receipt).read_text())
    assert receipt["bucket"] == BUCKET
    backup = {e["path"]: (e["size"], e["xet_hash"]) for e in receipt["files"]}
    listing = hf("buckets", "list", BUCKET, "--recursive", "--json", online=True)
    files = {e["path"]: e for e in json.loads(listing) if e["type"] == "file"}
    selected = selected_checkpoints()
    paired = {
        f"{os.path.dirname(path)}/critic-{iteration(path)}.pt" for path in selected
    }
    final = collections.defaultdict(int)
    for path in files:
        name = os.path.basename(path)
        if re.fullmatch(r"(iteration|critic)-\d+\.pt", name) and REPORTED.fullmatch(
            run_of(path)
        ):
            key = lineage(run_of(path)), name.split("-")[0]
            final[key] = max(final[key], iteration(path))

    def role(path, entry):
        if "/" not in path:
            return "keep: bucket-level record"
        run, name = run_of(path), os.path.basename(path)
        if not path.endswith(WEIGHTS):
            return "keep: results, records, metrics, logs"
        if not REPORTED.fullmatch(run):
            return "2 smoke, dry-run, benchmark, pilot or probe weights"
        mirror = path.removeprefix("checkpoint-history/")
        if mirror in selected:
            return "keep: validation-selected checkpoint"
        if name == "frozen-lora.pt":
            return "keep: frozen base weights"
        if name == "adapter_model.safetensors":
            return "keep: served adapter of an evaluated checkpoint"
        kind = name.split("-")[0]
        if kind == "resume":
            return "keep: resume slot of a finished run"
        if kind == "critic":
            if mirror in paired:
                return "keep: PPO critic of the validation-selected checkpoint"
            if iteration(path) == final[lineage(run), kind]:
                return "keep: final PPO critic"
            return "4 intermediate PPO critic snapshot"
        assert kind == "iteration", path
        if entry["size"] > 3.9e9:
            return "3 full-format non-selected checkpoint"
        if iteration(path) == final[lineage(run), kind]:
            return "keep: last-iteration checkpoint"
        return "keep: other validation checkpoint"

    roles = {path: role(path, entry) for path, entry in files.items()}

    def content(path):
        return files[path]["size"], files[path]["xet_hash"]

    # One kept path per distinct weight content: a run mirror before
    # checkpoint history, and the long VinePPO run before the others.
    keepers = {}
    for path in sorted(
        (p for p, r in roles.items() if r.startswith("keep") and p.endswith(WEIGHTS)),
        key=lambda p: (
            p.startswith("checkpoint-history/"),
            "handoff-rho-vineppo-650-s42-03" not in p,
            p,
        ),
    ):
        keepers.setdefault(content(path), path)
    kept = set(keepers.values())
    plan = {
        path: "1 byte-identical copy of a kept file"
        if path.endswith(WEIGHTS) and content(path) in keepers and path not in kept
        else roles[path]
        for path in files
    }
    assert all(
        any(
            plan.get(prefix + path, "").startswith("keep")
            for prefix in ("", "checkpoint-history/")
        )
        for path in selected
    ), "A selected checkpoint would lose its last copy"
    assert all(
        backup.get(path) == content(path)
        for path, tier in plan.items()
        if tier[0] in "234"
    ), "A tier 2-4 path is not in the backup receipt"

    summary = collections.defaultdict(lambda: [0, 0, 0, 0])
    for path, tier in plan.items():
        row, size = summary[tier], files[path]["size"]
        covered = backup.get(path) == content(path)
        row[0] += 1
        row[1] += size
        row[2] += covered
        row[3] += size * covered
    print(f"{len(files)} files, {sum(e['size'] for e in files.values())} bytes")
    print("files, bytes, files and bytes in the backup receipt, tier")
    for tier, (count, size, covered, covered_size) in sorted(summary.items()):
        print(f"{count:6d} {size:13d} {covered:5d} {covered_size:13d}  {tier}")
    Path(output).write_text(
        json.dumps(
            {
                "bucket": BUCKET,
                "backup_receipt_verified_at": receipt["verified_at"],
                "kept_copy_of": {
                    path: keepers[content(path)]
                    for path, tier in plan.items()
                    if tier.startswith("1 ")
                },
                "files": {
                    path: {
                        "plan": plan[path],
                        "size": files[path]["size"],
                        "xet_hash": files[path]["xet_hash"],
                    }
                    for path in sorted(files)
                },
            },
            indent=0,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
