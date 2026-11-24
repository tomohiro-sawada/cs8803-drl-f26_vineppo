"""CPU-only checks of the lenient GSM8K reading and the paired interval."""

import pytest

from scripts.diagnostics import regrade_evaluation as regrade


@pytest.mark.parametrize(
    "response,reference,expected",
    [
        ("so she makes \\boxed{18} dollars every day.", "18", True),
        ("#### $70,000", "70000", True),
        ("= 6.00\n\n#### ####\n6.00", "6", True),
        ("#### 2125", "2,125", True),
        ("#### -5", "-5", True),
        ("#### 19", "18", False),
        ("#### 6.50", "6", False),
        ("#### " + "9" * 400, "18", False),
        ("no number here", "18", False),
    ],
)
def test_lenient_reads_the_last_number(response, reference, expected):
    record = {"response": response, "reference_answer": reference}
    assert regrade.lenient(record) is expected


def test_paired_gap_clusters_rounds_by_question():
    def records(grades):
        return [
            {"example_id": question, "correct": correct}
            for question, rounds in enumerate(grades)
            for correct in rounds
        ]

    treated = records([(True, True), (True, False), (False, False)])
    baseline = records([(True, False), (False, False), (False, False)])
    mean, low, high = regrade.paired(treated, baseline, lambda r: r["correct"])
    assert mean == pytest.approx(100 / 3, abs=0.01)
    assert low == pytest.approx(0.67, abs=0.01)
    assert high == pytest.approx(66.0, abs=0.01)

    with pytest.raises(AssertionError, match="different questions"):
        regrade.paired(treated, baseline[:2], lambda r: r["correct"])


def test_interval_is_the_question_mean_with_its_standard_error():
    assert regrade.interval([1.0, 0.5, 0.0]) == [50.0, -6.58, 106.58]
