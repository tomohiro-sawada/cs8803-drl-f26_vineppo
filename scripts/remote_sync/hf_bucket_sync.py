"""Archive training run directories to the project's private Hugging Face bucket.

PACE-ICE scratch has a 300 GB quota and is purged storage, not an archive. This
mirrors run directories (checkpoints, selections, evaluation records, metrics)
under hf://buckets/<bucket>/<path relative to --root>. W&B's local run folders
are skipped: their content already lives on the W&B server.

Nothing is ever deleted, locally or remotely. Free local space separately, and
only after `verify` has passed for that run:

    python scripts/remote_sync/hf_bucket_sync.py push RUN_DIR [RUN_DIR ...]
    python scripts/remote_sync/hf_bucket_sync.py verify RUN_DIR [RUN_DIR ...]

`verify` re-downloads every file into --scratch and compares SHA256 with the
local copy, so it needs free space for the largest single file. HF_TOKEN is read
from the environment or the repository's .env. Login and compute nodes both
reach huggingface.co; long transfers belong in an ice-cpu job, not on a login
node.

AGENT INSTRUCTIONS: keep this a mirror. Do not add deletion, pruning or
"upload then remove" here; a wrong path would destroy the only copy of a run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "tsawada/cs8803-drl-f26_vineppo"
RUNS = Path("/home/hice1/tsawada7/scratch/cs8803-drl-f26_vineppo")
EXCLUDE = ("wandb/**", "evaluations/*/wandb/**")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf(*args: str) -> str:
    """Run the hf CLI of this environment; fail loudly, never echo the token."""
    env = dict(os.environ)
    if "HF_TOKEN" not in env:
        env["HF_TOKEN"] = dotenv_values(ROOT / ".env", interpolate=False)["HF_TOKEN"]
    assert env["HF_TOKEN"].strip(), "HF_TOKEN is empty"
    result = subprocess.run(
        [str(Path(sys.executable).with_name("hf")), *args],
        env=env | {"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        check=False,  # Asserted below, with the token redacted from the message.
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).replace(env["HF_TOKEN"], "<redacted>")
    assert result.returncode == 0, f"hf {' '.join(args)} failed:\n{output[-2000:]}"
    return result.stdout


def local_files(run: Path) -> dict[str, int]:
    """Relative path -> size for everything `push` uploads from this run."""
    files = {}
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run)
        if path.is_file() and "wandb" not in relative.parts[:-1]:
            files[relative.as_posix()] = path.stat().st_size
    assert files, f"Nothing to archive under {run}"
    return files


def remote_files(prefix: str) -> dict[str, int]:
    listing = hf("buckets", "list", f"{BUCKET}/{prefix}", "--recursive", "--json")
    return {
        entry["path"].removeprefix(prefix + "/"): entry["size"]
        for entry in json.loads(listing)
        if entry["type"] == "file"
    }


def targets(args) -> list[tuple[Path, str]]:
    root = args.root.resolve(strict=True)
    result = []
    for run in args.runs:
        run = run.resolve(strict=True)
        assert run.is_dir() and root in run.parents, f"{run} is not under {root}"
        result.append((run, run.relative_to(root).as_posix()))
    assert len({prefix for _, prefix in result}) == len(result), "Duplicate run"
    return result


def push(args) -> None:
    for run, prefix in targets(args):
        expected = local_files(run)
        hf(
            "buckets",
            "sync",
            str(run),
            f"hf://buckets/{BUCKET}/{prefix}",
            *(part for pattern in EXCLUDE for part in ("--exclude", pattern)),
        )
        assert remote_files(prefix) == expected, f"Bucket listing differs: {prefix}"
        print(f"pushed {prefix}: {len(expected)} files, {sum(expected.values())} bytes")


def verify(args) -> None:
    args.scratch.mkdir(parents=True, exist_ok=False)
    for run, prefix in targets(args):
        expected = local_files(run)
        assert remote_files(prefix) == expected, f"Bucket listing differs: {prefix}"
        for relative in expected:
            copy = args.scratch / "download"
            hf("buckets", "cp", f"hf://buckets/{BUCKET}/{prefix}/{relative}", str(copy))
            assert file_sha256(copy) == file_sha256(run / relative), (
                f"Content mismatch: {prefix}/{relative}"
            )
            copy.unlink()
        print(
            f"verified {prefix}: {len(expected)} files, {sum(expected.values())} bytes"
        )
    args.scratch.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=("push", "verify"))
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories")
    parser.add_argument("--root", type=Path, default=RUNS, help="Bucket path origin")
    parser.add_argument(
        "--scratch", type=Path, default=RUNS / "bucket-verify", help="New, verify only"
    )
    args = parser.parse_args()
    {"push": push, "verify": verify}[args.action](args)


if __name__ == "__main__":
    main()
