import json
from pathlib import Path
import sys

a, b = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
jobs = {j["name"]: j for j in a["jobs"]}
assert "vp26-olmo3-math-smoke-gate-audit" in jobs
assert not any("olmo3-gsm8k-smoke-" in n for n in jobs)
gate = jobs["vp26-olmo3-gsm8k-sft-gate-audit"]["gates"][-1]
assert gate.endswith("/handoff-olmo3-math-grpo-50-s42-gate-audit/COMPLETE")
gate = jobs["vp26-olmo3-gsm8k-grpo-50-s42-gate-audit"]["gates"][-1]
assert gate.endswith("/handoff-olmo3-gsm8k-sft-gate-audit/COMPLETE")
orig = {j["name"]:j for j in b["jobs"]}
for arm in ("olmo3-gsm8k-sft", "olmo3-gsm8k-grpo-50-s42"):
    after = jobs[f"vp26-{arm}-gate-audit"]
    before = orig[f"vp26-{arm}-10"]
    assert [v.replace("gate-audit","10") for v in after["submit"]] == before["submit"]
    assert after["space_gib"] == before["space_gib"]
print("Validated OLMo MATH and GSM8K admission order with identical production options and reservations.")
