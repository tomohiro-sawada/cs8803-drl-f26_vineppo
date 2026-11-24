"""Delete from the project bucket exactly the paths a reviewed plan names.

    python -m scripts.remote_sync.delete_planned PLAN.json DELETED.jsonl TIER [TIER ...] [--apply]

PLAN.json maps every bucket path to a tier, size and content hash; see
experiments/training/pace_ice/handoff_20261120/vp-bucket-triage.py. Without
--apply this runs every check and deletes nothing.

Each path must still have its planned size and content hash in a fresh bucket
listing. A tier-1 path is a duplicate, so its kept copy must be in the bucket
with identical content. A tier 2-4 path must appear with the same size and hash
in SYNC_RECEIPT.json at the bucket root, the manifest of the off-site backup.
Each deletion is appended to DELETED.jsonl, so an interrupted run can be
repeated with the same arguments.

AGENT INSTRUCTIONS: delete single named paths only. Do not add prefix,
recursive or pattern deletion, or a way to skip a check. A bucket has no
versions; a deleted object cannot be recovered from it.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.remote_sync.hf_bucket_sync import BUCKET, hf

TIERS = ("1", "2", "3", "4")


def identities(entries):
    return {entry["path"]: (entry["size"], entry["xet_hash"]) for entry in entries}


def bucket_listing():
    listing = hf("buckets", "list", BUCKET, "--recursive", "--json", online=True)
    return identities(e for e in json.loads(listing) if e["type"] == "file")


def backup_manifest():
    with TemporaryDirectory() as tmp:
        copy = Path(tmp) / "receipt.json"
        source = f"hf://buckets/{BUCKET}/SYNC_RECEIPT.json"
        hf("buckets", "cp", source, str(copy), online=True)
        receipt = json.loads(copy.read_text())
    assert receipt["bucket"] == BUCKET
    assert len(receipt["files"]) == receipt["num_files"]
    return identities(receipt["files"])


def checked(plan, paths, listing, backup):
    """Return paths after checking every planned deletion against live state."""
    files, kept = plan["files"], plan["kept_copy_of"]
    for path in paths:
        tier = files[path]["plan"][0]
        identity = files[path]["size"], files[path]["xet_hash"]
        assert tier in TIERS, f"Not a deletion tier: {path}"
        assert listing.get(path) == identity, f"Bucket differs from the plan: {path}"
        if tier == "1":
            copy = kept[path]
            assert files[copy]["plan"].startswith("keep"), (
                f"Kept copy is planned for deletion: {copy}"
            )
            assert listing.get(copy) == identity, (
                f"No identical kept copy in the bucket: {path}"
            )
        else:
            assert backup.get(path) == identity, f"Not in the backup receipt: {path}"
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("deleted", type=Path)
    parser.add_argument("tiers", nargs="+", choices=TIERS)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["bucket"] == BUCKET
    files = plan["files"]
    done = set()
    if args.deleted.exists():
        rows = args.deleted.read_text().splitlines()
        done = {json.loads(row)["path"] for row in rows}
    before = bucket_listing()
    assert not done & before.keys(), "A path recorded as deleted is in the bucket"
    paths = checked(
        plan,
        [p for p, f in files.items() if f["plan"][0] in args.tiers and p not in done],
        before,
        backup_manifest(),
    )
    for tier in sorted(set(args.tiers)):
        sizes = [files[p]["size"] for p in paths if files[p]["plan"][0] == tier]
        print(f"tier {tier}: {len(sizes)} files, {sum(sizes)} bytes", flush=True)
    if not args.apply:
        print("checks passed; nothing deleted (pass --apply to delete)")
        return
    with args.deleted.open("a") as stream:
        for path in paths:
            hf("buckets", "remove", f"{BUCKET}/{path}", "--yes", online=True)
            stream.write(json.dumps({"path": path} | files[path]) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    after = bucket_listing()
    gone = set(paths)
    assert not gone & after.keys(), "A deleted path is still listed"
    assert all(
        after.get(path) == identity
        for path, identity in before.items()
        if path not in gone
    ), "A path outside the plan changed"
    print(
        f"deleted {len(paths)} files, {sum(files[p]['size'] for p in paths)} bytes; "
        f"bucket now has {len(after)} files, "
        f"{sum(size for size, _ in after.values())} bytes"
    )


if __name__ == "__main__":
    main()
