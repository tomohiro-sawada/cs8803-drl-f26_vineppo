import json
from pathlib import Path
import sys

parent=Path(sys.argv[1])
for name, marker in (
    ("handoff-olmo3-gsm8k-sft-10", "COMPLETE"),
    ("handoff-olmo3-gsm8k-grpo-50-s42-10", "test-complete.json"),
):
    root=parent/name
    result=json.loads(Path(json.loads((root/marker).read_text())["result"]).read_text())
    result_dir=Path(result["evaluation_run"]["path"])
    records=[json.loads(line) for p in sorted(result_dir.glob("round-*/records.jsonl")) for line in p.open()]
    parsed=sum(bool(r["extracted_answer"]) for r in records)
    format_lines=sum(any(l.startswith("#### ") for l in r["response"].splitlines()) for r in records)
    correct=sum(bool(r["correct"]) for r in records)
    assert correct == result["total_correct"] and len(records) == result["total_count"]
    assert all(r["extracted_answer"] or not r["correct"] for r in records)
    print(name, "records", len(records), "parsed", parsed, "format-lines", format_lines,
          "correct", correct, "parsed accuracy", correct/parsed,
          "truncated", sum(r["finish_reason"]=="length" for r in records), flush=True)
