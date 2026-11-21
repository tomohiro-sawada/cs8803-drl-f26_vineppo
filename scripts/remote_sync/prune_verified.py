"""Prune checkpoint payloads after successful verification of an exact run."""

import argparse
import json
import os
import re
from pathlib import Path

from experiments.training.resume import atomic_record
from scripts.remote_sync.hf_bucket_sync import RUNS, local_files


def prune(run, log, receipt):
    run = run.resolve(strict=True)
    prefix = run.relative_to(RUNS.resolve(strict=True)).as_posix()
    assert prefix.startswith("training/pace_ice/") and (run / "COMPLETE").is_file()
    matches = re.findall(
        rf"^verified {re.escape(prefix)}: (\d+) files, (\d+) bytes$",
        log.read_text(),
        re.MULTILINE,
    )
    assert len(matches) == 1, f"No unique verification for {prefix}"
    count, size = map(int, matches[0])
    removed = (
        [json.loads(line) for line in receipt.read_text().splitlines()]
        if receipt.exists()
        else []
    )
    files = local_files(run)
    assert len(files) + len(removed) == count
    assert sum(files.values()) + sum(row["bytes"] for row in removed) == size
    assert all(row["run"] == prefix and row["path"] not in files for row in removed)
    with receipt.open("a") as stream:
        for relative, length in files.items():
            path = run / relative
            if (
                path.suffix != ".pt"
                or "checkpoints" not in path.relative_to(run).parts[:-1]
            ):
                continue
            assert path.resolve(strict=True) == path
            path.unlink()
            stream.write(
                json.dumps({"run": prefix, "path": relative, "bytes": length}) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    atomic_record(
        receipt.with_suffix(".complete.json"),
        {
            "run": prefix,
            "files": count,
            "bytes": size,
            "verification_log": str(log),
            "deleted_paths": str(receipt),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "log", "receipt"):
        parser.add_argument(name, type=Path)
    prune(**vars(parser.parse_args()))
