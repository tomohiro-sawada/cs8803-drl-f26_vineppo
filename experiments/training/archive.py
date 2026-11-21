"""Copy immutable training artifacts to the bucket and verify every byte."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.training.resume import atomic_record
from runtime.checkpoints import file_identity, file_sha256
from scripts.remote_sync.hf_bucket_sync import BUCKET, RUNS, hf, node_local_tmpdir


def archive_checkpoint(path):
    relative = path.resolve(strict=True).relative_to(RUNS.resolve(strict=True))
    assert relative.parts[:2] == ("training", "pace_ice")
    destination = f"hf://buckets/{BUCKET}/checkpoint-history/{relative.as_posix()}"
    receipt = path.with_suffix(".archive.json")
    identity = file_identity(path)
    digest = file_sha256(path)
    record = {"bucket": destination, "sha256": digest, "bytes": identity[2]}
    if receipt.exists():
        assert json.loads(receipt.read_text()) == record, (
            f"Invalid archive receipt: {receipt}"
        )
        return
    hf("buckets", "cp", str(path), destination, online=True)
    with TemporaryDirectory(
        dir=node_local_tmpdir(), prefix="checkpoint-verify-"
    ) as scratch:
        downloaded = Path(scratch) / "download"
        hf("buckets", "cp", destination, str(downloaded), online=True)
        assert file_sha256(downloaded) == digest, (
            f"Bucket content mismatch: {destination}"
        )
    assert file_identity(path) == identity, (
        f"Checkpoint changed while archiving: {path}"
    )
    atomic_record(receipt, record)
    print(
        f"verified {relative.as_posix()}: {record['bytes']} bytes (checkpoint history)"
    )
