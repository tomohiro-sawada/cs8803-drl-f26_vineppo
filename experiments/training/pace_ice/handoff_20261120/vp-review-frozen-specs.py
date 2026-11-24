"""Read-only: which frozen-base file (path, SHA256) each compact checkpoint expects."""
import glob, hashlib, os, sys
import torch

T = "/storage/ice1/2/9/tsawada7/cs8803-drl-f26_vineppo/training/pace_ice"
specs = {}
for run in sorted(os.listdir(T)):
    found = sorted(glob.glob(f"{T}/{run}/attempt-*/checkpoints/iteration-*.pt")) + sorted(glob.glob(f"{T}/{run}/checkpoints/iteration-*.pt"))
    found = [f for f in found if os.path.getsize(f) < 1e9]
    if not found or not run.startswith(("handoff-rho-", "handoff-olmo3-", "long-")):
        continue
    state = torch.load(found[-1], weights_only=True, map_location="cpu", mmap=True)
    spec = state["frozen"]
    own = os.path.dirname(spec["path"]) == os.path.dirname(os.path.realpath(found[-1]))
    print(f"{run} | {os.path.basename(found[-1])} | format {state['format_version']} | frozen sha256 {spec['sha256'][:16]} | frozen path in its own checkpoint dir: {own} | {os.path.relpath(spec['path'], T)}", flush=True)
    specs.setdefault(spec["sha256"], set()).add(run.split("-")[1])
print({k[:16]: sorted(v) for k, v in specs.items()})
for path in sys.argv[1:]:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    print("sha256", digest.hexdigest(), os.path.relpath(path, T), os.stat(path).st_nlink, flush=True)
