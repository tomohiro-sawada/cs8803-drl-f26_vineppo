import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from core import verifier
from core._safe_symbolic import parse_expr
from core.vendor.vineppo.math_answer_exctraction import extract_answer, strip_string
from core.verifier import VerificationResult, verify_response


@pytest.fixture(autouse=True)
def fresh_grading_cache():
    verifier._math_correct.cache_clear()
    yield
    verifier._math_correct.cache_clear()


def test_exact_math_cache_preserves_order_and_multiplicity(monkeypatch):
    import json

    observed = []

    def passed(args, *, input, **kwargs):
        observed.append(json.loads(input))
        return subprocess.CompletedProcess(args, 0, "true", "")

    monkeypatch.setattr(subprocess, "run", passed)
    for predictions in (("1", "2"), ("1", "2"), ("2", "1"), ("1", "1", "2")):
        assert verifier._math_correct(predictions, ("1", "2"))
    assert observed == [
        [["1", "2"], ["1", "2"]],
        [["2", "1"], ["1", "2"]],
        [["1", "1", "2"], ["1", "2"]],
    ]
    assert verifier._math_correct.cache_info().hits == 1
    assert verifier._math_correct.cache_info().maxsize == 8192


def test_batched_grading_matches_individual_math_results():
    responses = [r"\boxed{\sqrt{8}}", r"\boxed{1}", r"\boxed{2\sqrt{2}}", ""]
    kwargs = {"question": "Find the value.", "reference_is_solution": True}
    reference = r"Solution: \boxed{2\sqrt{2}}"
    expected = [
        verify_response(text, reference, task="math", **kwargs) for text in responses
    ]
    verifier._math_correct.cache_clear()
    assert (
        verifier.verify_math_responses(responses, reference, workers=2, **kwargs)
        == expected
    )


def test_batched_grading_deduplicates_work_with_bounded_overlap(monkeypatch):
    import json

    overlap = Barrier(2)
    calls = []

    def run(args, *, input, **kwargs):
        predictions, references = json.loads(input)
        calls.append(predictions)
        overlap.wait(timeout=5)
        return subprocess.CompletedProcess(
            args, 0, json.dumps(predictions == references), ""
        )

    monkeypatch.setattr(subprocess, "run", run)
    results = verifier.verify_math_responses(
        [r"\boxed{1}", r"\boxed{2}", r"\boxed{1}"],
        "1",
        question="",
        reference_is_solution=False,
        workers=2,
    )
    assert [item.correct for item in results] == [True, False, True]
    assert sorted(calls) == [["1"], ["2"]]


def test_batched_grading_propagates_worker_failures_without_caching(monkeypatch):
    def interrupted(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", interrupted)
    for _ in range(2):
        with pytest.raises(subprocess.TimeoutExpired) as failure:
            verifier.verify_math_responses(
                [r"\boxed{1}"] * 2,
                "1",
                question="",
                reference_is_solution=False,
                workers=2,
            )
        assert "MATH sample 0" in failure.value.__notes__[0]
    cache = verifier._math_correct.cache_info()
    assert (cache.hits, cache.misses, cache.currsize) == (0, 2, 0)


@pytest.mark.parametrize(
    "response,reference,answer,correct",
    [
        ("Work\n#### 42", "42", "42", True),
        ("#### -1,234", "-1234", "-1,234", True),
        ("#### 1\n#### 42", "42", "42", True),
        ("#### 42.0", "42", "42.0", False),
        ("#### 042", "42", "042", False),
        ("#### 42 dollars", "42", "42 dollars", False),
        ("#### 42\nmore reasoning", "42", "42\nmore reasoning", False),
        ("#### 42.", "42", "42.", False),
        ("The answer is 42", "42", None, False),
        (r"\boxed{42}", "42", None, False),
        ("Work 42\n###", "42", None, False),
        ("####   ", "42", None, False),
        ("", "42", None, False),
    ],
)
def test_gsm8k_original_format(response, reference, answer, correct):
    assert verify_response(response, reference, task="gsm8k") == VerificationResult(
        answer, correct, (answer,) if answer else ()
    )


@pytest.mark.parametrize(
    "response,reference,answer,correct",
    [
        (r"Thus \boxed{42}.", "42", "42", True),
        (r"\boxed{\dfrac12}", r"\frac{1}{2}", r"\frac{1}{2}", True),
        (r"\boxed{1/2}", "0.5", r"\frac{1}{2}", True),
        (r"\boxed{\sqrt{8}}", r"2\sqrt{2}", r"\sqrt{8}", True),
        (r"\boxed{(x+1)^2}", "x^2+2*x+1", "(x+1)^2", True),
        (r"\boxed{2*x}", "x+x", "2*x", True),
        (r"\boxed{\sin(0)}", "0", r"\sin(0)", True),
        (r"\boxed{x=2}", "2", "x=2", True),
        (r"\boxed{50%}", "0.5", "50%", True),
        (r"\boxed{50}", "0.5", "50", True),
        (r"\boxed{1.0005}", "1", "1.0005", True),
        (r"\boxed{1.01}", "1", "1.01", False),
        (r"\boxed{3}", "4", "3", False),
        (r"\boxed{(1,2)}", "[1,2]", "(1,2)", True),
        (r"\boxed{1}\boxed{2}", "2", "2", True),
        # Upstream deduplicates before selecting the last answer (not simply last box).
        (r"\boxed{1}\boxed{2}\boxed{1}", "2", "2", True),
        ("The answer is 42.\nMore work 7", "42", "42", True),
        ("The final answer is $42$. I hope it is correct. 7", "42", "42", True),
        ("```output\n42\n```", "42", "42", True),
        ("Unfinished calculation: 6 * 7 = 42", "42", "42", True),
        (r"\boxed{42", "42", None, False),
        (r"\boxed{}", "42", None, False),
        ("No numeric answer", "42", None, False),
        ("", "42", None, False),
    ],
)
def test_math_official_policies(response, reference, answer, correct):
    assert verify_response(response, reference, task="math") == VerificationResult(
        answer, correct, (answer,) if answer else ()
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        (r"\left( 2 \right)", "(2)"),
        (r"\frac12", r"\frac{1}{2}"),
        (r"\sqrt2", r"\sqrt{2}"),
        (r"12\text{ cm}", "12"),
        (r"90^{\circ}", "90"),
        (r"\$1,234.", "1,234"),
        (r"\text{yes}", "yes"),
    ],
)
def test_official_normalization(source, expected):
    assert strip_string(source) == expected


@pytest.mark.parametrize(
    "response,correct",
    [(r"\boxed{2}", False), (r"\boxed{1}\boxed{2}", True)],
)
def test_full_solution_preserves_multiple_reference_answers(response, correct):
    solution = r"First \boxed{1}, then \boxed{2}."
    result = verify_response(
        response, solution, task="math", reference_is_solution=True
    )
    assert result.correct is correct
    # Reducing the reference to its last answer incorrectly accepts a partial answer.
    assert verify_response(r"\boxed{2}", "2", task="math").correct


@pytest.mark.parametrize("task,response", [("math", r"\boxed{2}"), ("gsm8k", "#### 2")])
def test_explicit_final_answer_flag_preserves_default(task, response):
    assert verify_response(response, "2", task=task, reference_is_solution=False) == (
        verify_response(response, "2", task=task)
    )


def test_solution_reference_validation():
    with pytest.raises(ValueError, match="only for MATH"):
        verify_response("#### 2", "#### 2", task="gsm8k", reference_is_solution=True)
    with pytest.raises(ValueError, match="nonempty reference"):
        verify_response(
            r"\boxed{2}", r"\boxed{}", task="math", reference_is_solution=True
        )


def test_malformed_box_suppresses_number_fallback():
    assert extract_answer(r"42 then \boxed{broken") == ""
    assert extract_answer(r"\boxed{42} then \boxed{broken") == "42"


@pytest.mark.parametrize(
    "source", ["x.real", "x[0]", "lambda x: x", "'x'", "x_1", "x if x else 0"]
)
def test_symbolic_parser_rejects_non_arithmetic_syntax(source):
    with pytest.raises((ValueError, SyntaxError)):
        parse_expr(source, implicit=True)


def test_symbolic_parser_never_uses_eval(monkeypatch):
    from sympy.parsing import sympy_parser

    def forbidden(*args, **kwargs):
        pytest.fail("eval-based parsing must not be used")

    monkeypatch.setattr(sympy_parser, "eval_expr", forbidden)
    monkeypatch.setattr(sympy_parser, "parse_expr", forbidden)
    assert str(parse_expr("2x+x", implicit=True)) == "3*x"


def test_invalid_inputs():
    with pytest.raises(ValueError, match="Unsupported task"):
        verify_response("42", "42", task="other")
    with pytest.raises(TypeError):
        verify_response(None, "42", task="math")
    with pytest.raises(ValueError, match="nonempty"):
        verify_response("", " ", task="gsm8k")
    with pytest.raises(ValueError, match="exceed"):
        verify_response("x" * (verifier.MAX_INPUT_CHARS + 1), "1", task="math")


def test_worker_timeout_raises_and_next_call_works(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(verifier, "GRADING_TIMEOUT_SECONDS", 0.001)
        with pytest.raises(subprocess.TimeoutExpired):
            verify_response(r"\boxed{\sqrt{8}}", r"2\sqrt{2}", task="math")
    assert verify_response(r"\boxed{\sqrt{8}}", r"2\sqrt{2}", task="math").correct


def test_infrastructure_errors_are_not_scored(monkeypatch):
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "missing dependency")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        verify_response(r"\boxed{2}", "2", task="math")


def test_worker_bootstrap_is_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert verify_response(r"\boxed{2*x}", "x+x", task="math").correct


def test_threaded_caller():
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(verify_response, r"\boxed{2*x}", "x+x", task="math")
            for _ in range(2)
        ]
        assert all(f.result().correct for f in futures)


def test_provenance():
    import hashlib
    import json
    from pathlib import Path

    from core.vendor.vineppo import math_answer_exctraction

    assert verifier.UPSTREAM_COMMIT == "9f716e68bb934e71da86c226c1699c36b5553faf"
    assert verifier.UPSTREAM_COMMIT in verifier.VERIFIER_PROVENANCE
    source = Path(math_answer_exctraction.__file__)
    manifest = json.loads(source.with_name("sources.json").read_text())
    # Reconstruct upstream after undoing the documented mechanical adaptations.
    original = source.read_text().split("\n", 3)[3]
    original = (
        original.replace(
            "    return split_math_answers(question, extract_answer(reasoning, exhaust=True))\n\n\n"
            "def split_math_answers(question, answers):\n"
            "    # Factored unchanged from extract_math_answer to support final-answer references.\n"
            "    answer = []\n    for ans in answers:",
            "    answer = []\n    for ans in extract_answer(reasoning, exhaust=True):",
        )
        .replace("except AssertionError:", "except:")
        .replace("except (ValueError, AssertionError):", "except:")
    )
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == manifest["sha256"]["src/treetune/tasks/math_answer_exctraction.py"]
    )


@pytest.mark.parametrize(
    "answer,reference",
    [
        (
            r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
            r"\begin{bmatrix}1&2\\3&4\end{bmatrix}",
        ),
        (r"1\cup2", r"2\cup1"),
        ("x=2", "2=x"),
    ],
)
def test_official_structured_equality(answer, reference):
    assert verify_response("The answer is " + answer, reference, task="math").correct


def test_both_symbolic_inputs_are_restricted():
    assert not verify_response(r"\boxed{x.real}", "x", task="math").correct
    assert not verify_response(r"\boxed{x}", "x.real", task="math").correct


@pytest.mark.parametrize(
    "response,reference,question,gold,selected,correct",
    [
        (
            r"\boxed{2,1}",
            "1,2",
            "Give roots separated by commas",
            r"\boxed{1,2}",
            ("2", "1"),
            True,
        ),
        (
            r"\boxed{1}\boxed{2}",
            "1,2",
            "Give roots separated by commas",
            r"\boxed{1}\boxed{2}",
            ("1", "2"),
            True,
        ),
        (
            r"\boxed{9}\boxed{2}\boxed{1}",
            "1,2",
            "Give roots separated by commas",
            r"\boxed{1,2}",
            ("2", "1"),
            True,
        ),
        (
            r"\boxed{1}",
            "1,2",
            "Give roots separated by commas",
            r"\boxed{1,2}",
            ("1",),
            False,
        ),
        (
            r"\boxed{1,3}",
            "1,2",
            "Give roots separated by commas",
            r"\boxed{1,2}",
            ("1", "3"),
            False,
        ),
        (
            r"\boxed{1}\boxed{2}\boxed{1}",
            "1,1,2",
            "Give roots separated by commas",
            r"\boxed{1,1,2}",
            ("1", "2"),
            True,
        ),
        (
            r"\boxed{2}\boxed{1}",
            r"1\text{ and }2",
            "Give both roots",
            r"\boxed{1\text{ and }2}",
            ("2", "1"),
            True,
        ),
        (
            r"\boxed{2}\boxed{1}",
            r"1\text{and}2",
            "separated by commas",
            r"\boxed{1\text{and}2}",
            ("1",),
            False,
        ),
        (
            r"\boxed{(1,2)}",
            "(1,2)",
            "Give coordinates separated by commas",
            r"\boxed{(1,2)}",
            ("(1,2)",),
            True,
        ),
        (
            r"\boxed{[1,2]}",
            "[1,2]",
            "separated by commas",
            r"\boxed{[1,2]}",
            ("[1,2]",),
            True,
        ),
        (r"\boxed{2,1}", "1,2", "Separated by commas", r"\boxed{1,2}", ("2,1",), False),
        (r"\boxed{2,3,4}", "2,3,4", "", r"\boxed{2,3,4}", ("2,3,4",), True),
    ],
)
def test_question_aware_official_list_path(
    response, reference, question, gold, selected, correct
):
    from core.vendor.vineppo.math_answer_exctraction import (
        extract_math_answer,
        split_math_answers,
    )
    from core.vendor.vineppo.math_grader_minerva import eval_math

    # Upstream grade_answer_minerva_format ignores ground_truth and extracts gold
    # from the full solution. Show that this normalized scalar preserves that list.
    gold_answers = extract_math_answer(question, "Worked solution. " + gold)
    assert split_math_answers(question, [strip_string(reference)]) == gold_answers
    item = {
        "prediction": extract_math_answer(question, response),
        "answer": gold_answers,
    }
    assert bool(eval_math(item)) is correct
    assert tuple(item["prediction"]) == selected
    result = verify_response(response, reference, task="math", question=question)
    assert result == VerificationResult(selected[-1], correct, selected)


def test_question_default_and_gsm8k_compatibility():
    response = r"\boxed{2,1}"
    assert not verify_response(response, "1,2", task="math").correct
    assert verify_response(
        response, "1,2", task="math", question="separated by commas"
    ).correct
    assert verify_response(
        "#### 42", "42", task="gsm8k", question="separated by commas"
    ).correct
    with pytest.raises(TypeError, match="question"):
        verify_response(response, "1,2", task="math", question=None)
    with pytest.raises(ValueError, match="exceed"):
        verify_response(
            response, "1,2", task="math", question="x" * (verifier.MAX_INPUT_CHARS + 1)
        )


@pytest.mark.parametrize("returncode", [-9, -24, 1])
def test_worker_termination_is_not_incorrect(monkeypatch, returncode):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a[0], returncode, "", "worker terminated"
        ),
    )
    with pytest.raises(subprocess.CalledProcessError):
        verify_response(r"\boxed{2}", "2", task="math")


@pytest.mark.parametrize(
    "fault", [RuntimeError, MemoryError, RecursionError, KeyboardInterrupt]
)
def test_symbolic_faults_propagate(monkeypatch, fault):
    from core.vendor.vineppo import math_grader_minerva as grader

    def broken(*args, **kwargs):
        raise fault("not a mathematical nonmatch")

    monkeypatch.setattr(grader, "_sympy_parse", broken)
    with pytest.raises(fault):
        grader.eval_math({"prediction": ["2*x"], "answer": ["x+x"]})


def test_worker_bad_protocol_is_not_incorrect(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, '"false"', ""),
    )
    with pytest.raises(AssertionError, match="Invalid verifier worker result"):
        verify_response(r"\boxed{2}", "2", task="math")
