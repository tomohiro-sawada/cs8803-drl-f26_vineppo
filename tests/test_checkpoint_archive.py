"""Ensure downloaded bytes are verified before the archive receipt exists."""

import json
from pathlib import Path

import pytest

from experiments.training import archive


@pytest.mark.parametrize("corrupt", [False, True])
def test_archive_checkpoint_uses_verified_download(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(archive, "RUNS", tmp_path)
    path = tmp_path / "training/pace_ice/run/checkpoints/iteration-1.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"candidate")
    candidate = path.with_suffix(".archive.json")
    transfers = []

    def copy(*argv, online):
        assert online and argv[:2] == ("buckets", "cp")
        src, dst = argv[2:]
        transfers.append((src, dst))
        if src.startswith("hf://"):
            Path(dst).write_bytes(b"bad" if corrupt else path.read_bytes())

    monkeypatch.setattr(archive, "hf", copy)
    if corrupt:
        with pytest.raises(AssertionError, match="Bucket content mismatch"):
            archive.archive_checkpoint(path)
        assert not candidate.exists()
    else:
        archive.archive_checkpoint(path)
        archive.archive_checkpoint(path)
        assert json.loads(candidate.read_text())["bytes"] == len(path.read_bytes())
    assert len(transfers) == 2
    assert transfers[0][1] == transfers[1][0]
