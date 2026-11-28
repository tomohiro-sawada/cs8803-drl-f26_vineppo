"""Restore one full-critic PPO checkpoint from the bucket to its recorded paths.

    PYTHONPATH=src:. python vp-restore-ppo-checkpoint.py RUN ATTEMPT ITERATION BASE

RUN is the run directory on scratch, BASE the canonical frozen base of its
model. Downloads ATTEMPT/checkpoints/iteration-N.pt and critic-N.pt, hard-links
BASE at the frozen path the checkpoint records, and checks the base and the
critic against the SHA256 values in the checkpoint. Refuses to overwrite.
"""

import os
import sys
from pathlib import Path

import torch

from scripts.remote_sync.hf_bucket_sync import BUCKET, RUNS, file_sha256, hf


def main(run, attempt, iteration, base):
    run, base = Path(run).resolve(strict=True), Path(base).resolve(strict=True)
    directory = run / attempt / "checkpoints"
    prefix = f"hf://buckets/{BUCKET}/{directory.relative_to(RUNS.resolve())}"
    for name in (f"iteration-{iteration}.pt", f"critic-{iteration}.pt"):
        assert not (directory / name).exists(), directory / name
        hf("buckets", "cp", f"{prefix}/{name}", str(directory / name))
    checkpoint = directory / f"iteration-{iteration}.pt"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    frozen, critic = state["frozen"], state["critic_blob"]
    assert Path(critic["path"]) == directory / f"critic-{iteration}.pt", critic["path"]
    assert file_sha256(base) == frozen["sha256"], "Canonical base differs"
    assert not Path(frozen["path"]).exists(), frozen["path"]
    os.link(base, frozen["path"])
    digest = file_sha256(Path(critic["path"]))
    assert digest == critic["sha256"], "Critic differs from the checkpoint's record"
    print(
        f"restored {checkpoint}\ncritic sha256 {digest}\nfrozen base {frozen['path']}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
