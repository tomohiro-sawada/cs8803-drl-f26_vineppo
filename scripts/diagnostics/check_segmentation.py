"""Offline fixed-span replay of explicit OpenRouter proxy .json/.jsonl inputs.

JSON accepts the version-1 fixture envelope; JSONL accepts producer records.
Replays the historical strict cut policy captured in those snapshots.
No tokenizer, model, network, sampled-token validation, or repairs are involved.
"""

import argparse
import hashlib
import json
import re
from functools import partial
from pathlib import Path

from core.segmentation import gsm8k_step_boundaries, math_step_boundaries

TOKEN_SOURCE = "locally_reencoded_proxy_text_NOT_sampled"
SEGMENT = {
    "gsm8k": gsm8k_step_boundaries,
    "math": partial(math_step_boundaries, round_forward=False),
}
LIMITATIONS = (
    "Only saved proxy text, token IDs and character spans are replayed; their "
    "tokenizer correspondence is not validated. Tokenizer roundtrip and special-"
    "token assertions cannot be reproduced. These are NOT sampled token IDs. "
    "RhoMath training risk is not estimated. No cuts or tokens are repaired."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_record(record: dict) -> None:
    """Validate saved data outside the production rejection-catching boundary."""
    require(isinstance(record, dict), "Record must be an object")
    require(record.get("task") in ("gsm8k", "math"), "Invalid task")
    require(isinstance(record.get("id"), str) and bool(record["id"]), "Invalid id")
    require(isinstance(record.get("text"), str), "Invalid text")
    require(record.get("token_id_source") == TOKEN_SOURCE, "Invalid token_id_source")
    text = record["text"]
    ids, spans = record.get("reencoded_token_ids"), record.get("character_spans")
    require(
        isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids),
        "Token IDs must be nonnegative integers",
    )
    require(isinstance(spans, list) and len(spans) == len(ids), "One span per token")
    previous_start = covered = 0
    for span in spans:
        require(
            isinstance(span, list)
            and len(span) == 2
            and all(type(i) is int for i in span),
            "Span must contain two integers",
        )
        start, end = span
        require(0 <= start <= end <= len(text), "Invalid span range")
        require(previous_start <= start <= covered <= end, "Nonmonotone/gapped spans")
        require(start < end or start == len(text), "Nonterminal zero-width span")
        previous_start, covered = start, end
    require(covered == len(text), "Spans must cover text")
    if "decoded_reencoded_text" in record:
        require(
            isinstance(record["decoded_reencoded_text"], str), "Invalid decoded text"
        )
    expected = record.get("segmentation")
    require(isinstance(expected, dict), "Missing segmentation snapshot")
    require(type(expected.get("success")) is bool, "Invalid snapshot success")
    if expected["success"]:
        require(
            set(expected) == {"success", "category", "boundaries"},
            "Invalid snapshot keys",
        )
        require(expected["category"] == "success", "Invalid success category")
        boundaries = expected["boundaries"]
        require(
            isinstance(boundaries, list)
            and len(boundaries) >= 2
            and all(type(i) is int and 0 <= i <= len(ids) for i in boundaries),
            "Invalid snapshot boundaries",
        )
    else:
        require(
            set(expected)
            == {"success", "category", "exception_type", "exception_message"},
            "Invalid rejection snapshot keys",
        )
        require(
            expected["category"]
            in ("inside_token", "empty", "upstream_or_span", "roundtrip", "special"),
            "Invalid rejection category",
        )
        require(
            expected["exception_type"] in ("AssertionError", "ValueError"),
            "Invalid exception type",
        )
        require(
            isinstance(expected["exception_message"], str), "Invalid exception message"
        )


def check_record(record: dict) -> dict:
    """Return observed outcome and exact snapshot comparison without mutation."""
    validate_record(record)
    text, ids, spans = (
        record["text"],
        record["reencoded_token_ids"],
        record["character_spans"],
    )
    expected = record["segmentation"]
    result = {"task": record["task"], "id": record["id"], "expected": expected}
    if "decoded_reencoded_text" in record and record["decoded_reencoded_text"] != text:
        return result | {
            "replayable": False,
            "observed": None,
            "comparison": "non_comparable",
            "reason": "decoded_reencoded_text differs from original text",
        }
    diagnostics = None
    try:
        boundaries = SEGMENT[record["task"]](text, ids, spans)
    except (AssertionError, ValueError) as error:
        message = str(error)
        cut = re.fullmatch(
            r"Step boundary at character (\d+) falls inside a sampled token", message
        )
        category = "upstream_or_span"
        if cut:
            category = "inside_token"
            position = int(cut[1])
            context_start, context_end = (
                max(0, position - 40),
                min(len(text), position + 40),
            )
            diagnostics = {
                "character_cut": position,
                "overlapping_tokens": [
                    {
                        "index": i,
                        "id": token,
                        "span": span,
                        "text": text[span[0] : span[1]],
                    }
                    for i, (token, span) in enumerate(zip(ids, spans, strict=True))
                    if span[0] < position < span[1]
                ],
                "context_span": [context_start, context_end],
                "context": text[context_start:context_end],
            }
        elif message == "Text and sampled response must be nonempty":
            category = "empty"
        observed = {
            "success": False,
            "category": category,
            "exception_type": type(error).__name__,
            "exception_message": message,
        }
    else:
        observed = {
            "success": True,
            "category": "success",
            "boundaries": list(boundaries),
        }
    result.update(replayable=True, observed=observed)
    if expected["category"] in ("roundtrip", "special"):
        result.update(
            comparison="non_comparable",
            reason="Expected tokenizer assertion is outside fixed-span replay scope",
        )
    else:
        result["comparison"] = "match" if expected == observed else "changed"
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    return result


def check_inputs(inputs: list[Path]) -> dict:
    """Read, validate and hash all inputs before replaying any records."""
    require(bool(inputs), "At least one explicit input is required")
    records, provenance, seen = [], [], set()
    for path in inputs:
        data = path.read_bytes()
        if path.suffix == ".json":
            envelope = json.loads(data)
            require(isinstance(envelope, dict), "Expected fixture envelope")
            require(
                type(envelope.get("schema_version")) is int
                and envelope["schema_version"] == 1,
                "Expected schema_version 1",
            )
            rows = envelope.get("cases")
            require(isinstance(rows, list), "Expected cases list")
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        else:
            raise ValueError(
                f"Expected .json fixture or .jsonl producer records: {path}"
            )
        require(bool(rows), f"Empty input: {path}")
        for row in rows:
            validate_record(row)
            key = (row["task"], row["id"])
            require(key not in seen, f"Duplicate task/id: {key}")
            seen.add(key)
        records.extend(rows)
        provenance.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "records": len(rows),
            }
        )
    findings = [check_record(record) for record in records]
    tasks = {}
    for finding in findings:
        counts = tasks.setdefault(
            finding["task"],
            dict.fromkeys(
                (
                    "total",
                    "success",
                    "rejected",
                    "not_replayable",
                    "match",
                    "changed",
                    "non_comparable",
                ),
                0,
            ),
        )
        counts["total"] += 1
        counts[finding["comparison"]] += 1
        observed = finding["observed"]
        outcome = (
            "not_replayable"
            if observed is None
            else "success"
            if observed["success"]
            else "rejected"
        )
        counts[outcome] += 1
    return {
        "limitations": LIMITATIONS,
        "inputs": provenance,
        "tasks": tasks,
        "records": findings,
        "changed_expectations": [
            {"task": f["task"], "id": f["id"]}
            for f in findings
            if f["comparison"] == "changed"
        ],
    }


def exit_status(report: dict, *, allow_known_rejections: bool = False) -> int:
    return int(
        any(
            record["comparison"] != "match"
            or (not allow_known_rejections and not record["observed"]["success"])
            for record in report["records"]
        )
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, help="Exclusively create a JSON report instead of stdout"
    )
    parser.add_argument(
        "--allow-known-rejections",
        action="store_true",
        help="Exit 0 only if every replay matches its snapshot, including rejections",
    )
    args = parser.parse_args(argv)
    report = check_inputs(args.input)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    return exit_status(report, allow_known_rejections=args.allow_known_rejections)


if __name__ == "__main__":
    raise SystemExit(main())
