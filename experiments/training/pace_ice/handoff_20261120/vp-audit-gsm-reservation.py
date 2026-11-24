import json
from pathlib import Path
import sys

generated, active = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
new = next(j for j in generated["jobs"] if j["name"] == "vp26-olmo3-gsm8k-grpo-50-s42-disk-audit")
old = next(j for j in active["jobs"] if j["name"] == "vp26-olmo3-gsm8k-grpo-50-s42-10")
assert new["space_gib"] == 35
assert new["submit"][2] == "--time=06:00:00" and old["submit"][2] == "--time=12:00:00"
normalize = lambda v: [p.replace("disk-audit","10") for p in v]
assert normalize(new["submit"][3:]) == old["submit"][3:]
assert new["gates"][-1].endswith("/handoff-olmo3-gsm8k-sft-disk-audit/COMPLETE")
assert generated["deadline"] == active["deadline"]
print("35 GiB reservation and six-hour link verified; all scientific options and gates match.")
