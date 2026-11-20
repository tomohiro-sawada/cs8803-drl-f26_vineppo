"""Offline CLI and schema coverage using all six saved OpenRouter fixtures."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.diagnostics import check_segmentation as checker

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/segmentation/openrouter_proxy.json"


@pytest.fixture
def cases():
    return json.loads(FIXTURE.read_text())["cases"]


def write_jsonl(path, cases):
    path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    return path


def test_all_six_fixtures_and_diagnostics(cases):
    original = copy.deepcopy(cases)
    report = checker.check_inputs([FIXTURE])
    assert len(report["records"]) == 6
    assert report["changed_expectations"] == []
    for task, total, success, rejected in (("gsm8k", 2, 2, 0), ("math", 4, 2, 2)):
        assert report["tasks"][task] == {
            "total": total,
            "success": success,
            "rejected": rejected,
            "not_replayable": 0,
            "match": total,
            "changed": 0,
            "non_comparable": 0,
        }
    assert report["inputs"] == [
        {
            "path": str(FIXTURE),
            "sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "records": 6,
        }
    ]
    for case, finding in zip(cases, report["records"], strict=True):
        assert checker.check_record(case) == finding
        assert finding["observed"] == case["segmentation"]
        assert finding["comparison"] == "match"
        if not finding["observed"]["success"]:
            diagnostic = finding["diagnostics"]
            cut = diagnostic["character_cut"]
            assert (
                cut == {"math:train:1535": 2244, "math:train:11087": 1516}[case["id"]]
            )
            assert str(cut) in finding["observed"]["exception_message"]
            assert diagnostic["overlapping_tokens"]
            for token in diagnostic["overlapping_tokens"]:
                start, end = token["span"]
                assert start < cut < end
                assert case["reencoded_token_ids"][token["index"]] == token["id"]
                assert case["character_spans"][token["index"]] == token["span"]
                assert token["text"] == case["text"][start:end]
            start, end = diagnostic["context_span"]
            assert diagnostic["context"] == case["text"][start:end]
    assert cases == original
    assert "not validated" in report["limitations"]
    assert "RhoMath training risk is not estimated" in report["limitations"]
    assert checker.exit_status(report) == 1
    assert checker.exit_status(report, allow_known_rejections=True) == 0


@pytest.mark.parametrize("allow, status", [(False, 1), (True, 0)])
def test_real_module_cli(allow, status):
    command = [
        sys.executable,
        "-m",
        "scripts.diagnostics.check_segmentation",
        "--input",
        str(FIXTURE),
    ]
    if allow:
        command.append("--allow-known-rejections")
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=30, check=False
    )
    assert result.returncode == status, result.stderr
    assert len(json.loads(result.stdout)["records"]) == 6
    assert not result.stderr


def test_jsonl_producer_shape_and_multiple_inputs(cases, tmp_path, capsys):
    for case in cases:
        case.update(
            decoded_reencoded_text=case["text"],
            index=1,
            prompt="saved prompt",
            request={"model": "proxy"},
            response={"choices": []},
        )
    paths = [
        write_jsonl(tmp_path / f"{task}.jsonl", [c for c in cases if c["task"] == task])
        for task in ("gsm8k", "math")
    ]
    assert checker.main(["--input", *map(str, paths), "--allow-known-rejections"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["inputs"]) == 2
    assert len(report["records"]) == 6


def test_changed_boundaries_and_failure_messages(cases, tmp_path):
    cases[0]["segmentation"]["boundaries"][1] += 1
    cases[3]["segmentation"]["exception_message"] += " corrupted"
    path = write_jsonl(tmp_path / "drift.jsonl", cases)
    report = checker.check_inputs([path])
    assert report["changed_expectations"] == [
        {"task": c["task"], "id": c["id"]} for c in (cases[0], cases[3])
    ]
    assert checker.exit_status(report) == 1
    assert checker.exit_status(report, allow_known_rejections=True) == 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("task", "other"),
        ("task", []),
        ("id", 1),
        ("id", ""),
        ("text", None),
        ("token_id_source", "sampled"),
        ("reencoded_token_ids", [True]),
        ("reencoded_token_ids", [-1]),
        ("reencoded_token_ids", [1.5]),
        ("character_spans", []),
        ("decoded_reencoded_text", 123),
        ("segmentation", {}),
        ("segmentation", {"success": 1}),
    ],
)
def test_invalid_record_schema_is_fatal(cases, field, value, monkeypatch):
    def must_not_run(*args):
        pytest.fail("Malformed schema reached production")

    monkeypatch.setitem(checker.SEGMENT, "gsm8k", must_not_run)
    cases[0][field] = value
    with pytest.raises(ValueError):
        checker.check_record(cases[0])


@pytest.mark.parametrize(
    "span",
    [[0], [0, 1, 2], [False, 1], [0, 1.0], [-1, 1], [0, 999999], [1, 1], [0, 0], "bad"],
)
def test_invalid_spans_are_fatal(cases, span):
    cases[0]["character_spans"][0] = span
    with pytest.raises(ValueError):
        checker.check_record(cases[0])


def test_missing_fields_and_non_objects(cases):
    for field in (
        "task",
        "id",
        "text",
        "token_id_source",
        "reencoded_token_ids",
        "character_spans",
        "segmentation",
    ):
        case = cases[0].copy()
        del case[field]
        with pytest.raises(ValueError):
            checker.check_record(case)
    with pytest.raises(ValueError):
        checker.check_record([])


def test_duplicate_and_all_inputs_validated_before_replay(cases, tmp_path, monkeypatch):
    first = write_jsonl(tmp_path / "first.jsonl", cases)
    second = write_jsonl(tmp_path / "second.jsonl", [cases[0]])

    def must_not_run(*args):
        pytest.fail("Replay ran before validating all inputs")

    monkeypatch.setattr(checker, "check_record", must_not_run)
    with pytest.raises(ValueError, match="Duplicate task/id"):
        checker.check_inputs([first, second])
    with pytest.raises(ValueError, match="Duplicate task/id"):
        checker.check_inputs([write_jsonl(second, [cases[0], cases[0]])])
    cases[0]["text"] = None
    write_jsonl(second, [cases[0]])
    with pytest.raises(ValueError, match="Invalid text"):
        checker.check_inputs([first, second])


@pytest.mark.parametrize("category", ["roundtrip", "special"])
def test_unsupported_expectation_is_not_false_agreement(cases, tmp_path, category):
    case = cases[0]
    case["segmentation"] = {
        "success": False,
        "category": category,
        "exception_type": "AssertionError",
        "exception_message": "tokenizer-only assertion",
    }
    report = checker.check_inputs([write_jsonl(tmp_path / "unsupported.jsonl", [case])])
    finding = report["records"][0]
    assert finding["observed"]["success"] is True
    assert finding["comparison"] == "non_comparable"
    assert report["changed_expectations"] == []
    assert checker.exit_status(report, allow_known_rejections=True) == 1


def test_decoded_text_mismatch_not_replayed(cases, tmp_path, monkeypatch):
    cases[0]["decoded_reencoded_text"] = cases[0]["text"] + "different"

    def must_not_run(*args):
        pytest.fail("Mismatched decoded text was replayed")

    monkeypatch.setitem(checker.SEGMENT, "gsm8k", must_not_run)
    report = checker.check_inputs(
        [write_jsonl(tmp_path / "mismatch.jsonl", [cases[0]])]
    )
    finding = report["records"][0]
    assert finding["observed"] is None
    assert finding["replayable"] is False
    assert finding["comparison"] == "non_comparable"
    assert report["tasks"]["gsm8k"]["not_replayable"] == 1
    assert (
        checker.exit_status(report)
        == checker.exit_status(report, allow_known_rejections=True)
        == 1
    )


def test_empty_text_and_production_exceptions(cases, monkeypatch):
    case = cases[0]
    case.update(text="", reencoded_token_ids=[], character_spans=[])
    finding = checker.check_record(case | {"segmentation": cases[3]["segmentation"]})
    assert finding["observed"]["category"] == "empty"
    for exception in (
        AssertionError("upstream"),
        ValueError("upstream"),
        RuntimeError("unexpected"),
    ):

        def fail(*args, exception=exception):
            raise exception

        monkeypatch.setitem(checker.SEGMENT, "gsm8k", fail)
        if isinstance(exception, RuntimeError):
            with pytest.raises(RuntimeError, match="unexpected"):
                checker.check_record(case | {"segmentation": cases[3]["segmentation"]})
        else:
            observed = checker.check_record(
                case | {"segmentation": cases[3]["segmentation"]}
            )["observed"]
            assert observed["exception_type"] == type(exception).__name__
            assert observed["category"] == "upstream_or_span"


def test_explicit_inputs_missing_files_and_exclusive_output(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        checker.main([])
    assert error.value.code == 2
    with pytest.raises(FileNotFoundError):
        checker.check_inputs([tmp_path / "missing.jsonl"])
    with pytest.raises(ValueError):
        checker.check_inputs([])
    output = tmp_path / "report.json"
    capsys.readouterr()
    assert checker.main(["--input", str(FIXTURE), "--output", str(output)]) == 1
    assert not capsys.readouterr().out
    saved = output.read_bytes()
    assert len(json.loads(saved)["records"]) == 6
    with pytest.raises(FileExistsError):
        checker.main(["--input", str(FIXTURE), "--output", str(output)])
    assert output.read_bytes() == saved


@pytest.mark.parametrize(
    "name, content",
    [
        ("bad.json", "[]"),
        ("bad.json", '{"schema_version": 2, "cases": []}'),
        ("bad.json", '{"schema_version": true, "cases": []}'),
        ("bad.json", '{"schema_version": 1, "cases": []}'),
        ("bad.jsonl", ""),
        ("bad.jsonl", "\n"),
        ("bad.jsonl", "not json"),
        ("bad.txt", "{}"),
    ],
)
def test_malformed_files(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(ValueError):
        checker.check_inputs([path])


def test_import_does_not_load_model_stack():
    code = (
        "import sys; from scripts.diagnostics import check_segmentation; "
        "assert not {'torch', 'transformers', 'datasets', 'tokenizers'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=30)
