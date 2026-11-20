"""VinePPO final-answer verification (not the training truncation penalty).

MATH uses the official no-answer-prefix/Minerva extraction and grading path;
GSM8K uses the official original-format task (last ####, string equality).
References default to final answers; MATH can opt into full solutions. The question controls
upstream comma splitting on both predictions and the reference. Multiple answers
retain upstream deduplication, last-N selection and list matching. See
vendor/vineppo/NOTICE.md for the precise relationship to upstream's full-solution API.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .vendor.vineppo.gsm8k import GSM8K
from .vendor.vineppo.math_answer_exctraction import (
    extract_math_answer,
    split_math_answers,
    strip_string,
)

UPSTREAM_REPOSITORY = "https://github.com/McGill-NLP/VinePPO"
UPSTREAM_COMMIT = "9f716e68bb934e71da86c226c1699c36b5553faf"
VERIFIER_PROVENANCE = f"{UPSTREAM_REPOSITORY}/tree/{UPSTREAM_COMMIT}/src/treetune/tasks"
VERIFIER_POLICY = "vineppo-question-final-answer-safe-v2"
MAX_INPUT_CHARS = 65536
GRADING_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class VerificationResult:
    extracted_answer: str | None
    correct: bool
    extracted_answers: tuple[str, ...] = ()


def _math_correct(predictions: list[str], references: list[str]) -> bool:
    worker = Path(__file__).with_name("_verifier_worker.py")
    # run kills and reaps the worker on TimeoutExpired. Do not conflate resource
    # limits, startup delays or external termination with a mathematical nonmatch.
    result = subprocess.run(
        [sys.executable, "-I", str(worker)],
        input=json.dumps([predictions, references]),
        capture_output=True,
        check=False,
        text=True,
        timeout=GRADING_TIMEOUT_SECONDS,
        env={},
    )
    result.check_returncode()  # Missing dependencies / broken workers are not wrong answers.
    correct = json.loads(result.stdout)
    assert type(correct) is bool, "Invalid verifier worker result"
    return correct


def verify_response(
    response: str,
    reference_answer: str,
    *,
    task: Literal["math", "gsm8k"],
    question: str = "",
    reference_is_solution: bool = False,
) -> VerificationResult:
    """Score extracted text regardless of stop/length finish reason.

    Supply the original question for official question-dependent MATH splitting;
    the empty default preserves older callers, not full question-aware parity.
    extracted_answers contains all deduplicated, last-N selected predictions;
    extracted_answer is its final member (or None for missing/empty extraction).
    reference_is_solution extracts all reference answers from a MATH solution;
    the default retains the final-answer API. GSM8K does not support this option.
    Timeout/resource/infrastructure failures raise rather than assign a score.
    """
    if task not in ("math", "gsm8k"):
        raise ValueError(f"Unsupported task: {task!r}")
    if reference_is_solution and task != "math":
        raise ValueError("reference_is_solution is supported only for MATH")
    if not all(
        isinstance(value, str) for value in (response, reference_answer, question)
    ):
        raise TypeError("response, reference_answer and question must be strings")
    if not reference_answer.strip():
        raise ValueError("reference_answer must be a nonempty normalized final answer")
    if max(len(response), len(reference_answer), len(question)) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Verifier inputs must not exceed {MAX_INPUT_CHARS} characters"
        )
    if task == "gsm8k":
        grader = GSM8K()
        answer = grader.extract_predicted_answer_from_text(response)
        return VerificationResult(
            answer or None,
            bool(answer)
            and grader.grade_answer(given_answer=answer, ground_truth=reference_answer),
            (answer,) if answer else (),
        )
    if reference_is_solution:
        references = extract_math_answer(question, reference_answer)
        if not references or not all(references):
            raise ValueError("MATH solution must contain nonempty reference answers")
    else:
        reference = strip_string(reference_answer)
        if not reference:
            raise ValueError("MATH reference normalizes to an empty answer")
        references = split_math_answers(question, [reference])
    predictions = extract_math_answer(question, response)
    # Mirror eval_math's selection for reporting; send original lists for grading.
    selected = tuple(dict.fromkeys(predictions))[-len(set(references)) :]
    if not selected or selected == ("",):
        return VerificationResult(None, False)
    return VerificationResult(
        selected[-1] or None, _math_correct(predictions, references), selected
    )
