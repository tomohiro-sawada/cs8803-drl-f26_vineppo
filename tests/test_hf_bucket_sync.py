import os

import pytest

from scripts.remote_sync.hf_bucket_sync import FROZEN_BLOB, local_files


def test_only_verified_base_links_can_be_omitted(tmp_path):
    base = tmp_path / FROZEN_BLOB
    base.write_bytes(b"base")
    run = tmp_path / "run"
    checkpoints = run / "attempt-0000" / "checkpoints"
    checkpoints.mkdir(parents=True)
    blob = checkpoints / FROZEN_BLOB
    blob.write_bytes(base.read_bytes())
    checkpoint = checkpoints / "iteration-2.pt"
    checkpoint.write_bytes(b"state")
    assert blob.relative_to(run).as_posix() in local_files(run)
    with pytest.raises(AssertionError, match="Unlinked frozen base"):
        local_files(run, linked_frozen_base=base)
    blob.unlink()
    os.link(base, blob)
    assert local_files(run, linked_frozen_base=base) == {
        checkpoint.relative_to(run).as_posix(): 5
    }
