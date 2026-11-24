"""Hard-link identical frozen blobs of a completed run to its verified base."""

import argparse
import os
from pathlib import Path

from scripts.remote_sync.hf_bucket_sync import FROZEN_BLOB, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("sha256")
    args = parser.parse_args()
    assert (args.run / "COMPLETE").is_file(), f"Run incomplete: {args.run}"
    assert file_sha256(args.base) == args.sha256, f"Canonical base differs: {args.base}"
    blobs = sorted(args.run.glob(f"attempt-*/checkpoints/{FROZEN_BLOB}"))
    assert blobs, f"No frozen base under {args.run}"
    for blob in blobs:
        if not blob.samefile(args.base):
            assert file_sha256(blob) == args.sha256, f"Frozen blob differs: {blob}"
            temporary = blob.with_name(f".{FROZEN_BLOB}.{os.getpid()}.link")
            os.link(args.base, temporary)
            os.replace(temporary, blob)
        assert blob.samefile(args.base)
        print(f"linked {blob}", flush=True)


if __name__ == "__main__":
    main()
