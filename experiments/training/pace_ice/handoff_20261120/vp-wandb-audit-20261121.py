import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

from dotenv import dotenv_values
from wandb import Api

def read(path):
    return json.loads(path.read_text())

root = Path(sys.argv[1])
os.environ["WANDB_API_KEY"] = dotenv_values(Path.cwd() / ".env", interpolate=False)["WANDB_API_KEY"]
os.environ["WANDB_BASE_URL"] = "https://api.forge.coreweave.com"
api = Api(timeout=30)
checked = 0
for path in sorted(root.glob("handoff-*/COMPLETE")):
    directory = path.parent
    name = directory.name
    if not (name.startswith("handoff-rho-") or name.startswith("handoff-olmo3-")) or "-smoke-" in name:
        continue
    if (pointer := directory / "test-complete.json").is_file():
        result_file = Path(read(pointer)["result"])
        candidates = sorted(directory.glob("attempt-*/wandb.json"))
        assert candidates
        tracking = read(candidates[-1])
    else:
        assert "-sft-" in name
        result_file = Path(read(path)["result"])
        tracking = read(result_file.with_name("wandb.json"))
    result = read(result_file)
    url = urlparse(tracking["url"])
    assert url.netloc == "forge.coreweave.com", name
    run = api.run(f"tsawada/vineppo/{tracking['id']}")
    assert run.id == tracking["id"], name
    assert run.summary["eval/test/rounds_completed"] == result["rounds_completed"] == 16, name
    assert abs(run.summary["eval/test/pass_at_1"] - result["accuracy"]) < 1e-10, name
    print(name, run.id, run.state, result["accuracy"], tracking["url"], flush=True)
    checked += 1
assert checked >= 14, checked
print(f"Verified online tests for {checked} required runs.", flush=True)
