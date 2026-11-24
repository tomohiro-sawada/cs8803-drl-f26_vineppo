"""Download the kept frozen bases from the bucket to node-local disk and print their SHA256."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.remote_sync.hf_bucket_sync import BUCKET, hf, node_local_tmpdir

EXPECTED = {
    "training/pace_ice/handoff-rho-vineppo-650-s42-03/attempt-0000/checkpoints/frozen-lora.pt": "086cc6d318d3aeadf44ab916dd960f5dceb56c14dcd0d1e90f5bdfa45d0b9fbf",
    "training/pace_ice/handoff-olmo3-math-grpo-50-s42-10/attempt-0000/checkpoints/frozen-lora.pt": "7a574b6997d1acb19ec12204c10522f106b94ea5ee303417a78c0faa3c1c646a",
}
with TemporaryDirectory(dir=node_local_tmpdir(), prefix="kept-base-") as tmp:
    copy = Path(tmp) / "download"
    for path, expected in EXPECTED.items():
        hf("buckets", "cp", f"hf://buckets/{BUCKET}/{path}", str(copy), online=True)
        digest = hashlib.sha256()
        with copy.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 24), b""):
                digest.update(chunk)
        size = copy.stat().st_size
        copy.unlink()
        print(f"{digest.hexdigest()} {size} {path}", flush=True)
        assert digest.hexdigest() == expected, f"Kept base differs: {path}"
print("kept bases verified", flush=True)
