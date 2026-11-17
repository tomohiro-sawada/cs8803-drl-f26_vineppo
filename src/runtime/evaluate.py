"""Text evaluation with explicit inputs and sample-level accuracy.

The default bare-question prompt is a smoke test, not the paper's prompting
recipe. Malformed answers count as incorrect; truncated answers are scored from
available text, without a training truncation penalty. Backend/verifier failures
abort evaluation. This module neither loads models/datasets nor selects splits.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol

from core.data import Example
from core.inference import TextGenerator
from core.verifier import VERIFIER_POLICY, VERIFIER_PROVENANCE, verify_response


@dataclass(frozen=True)
class EvaluationRecord:
    example_id: str
    sample_index: int
    task: Literal["math", "gsm8k"]
    question: str
    prompt: str
    response: str
    finish_reason: Literal["stop", "length"]
    extracted_answer: str | None
    extracted_answers: tuple[str, ...]
    reference_answer: str
    correct: bool


@dataclass(frozen=True)
class EvaluatorConfig:
    """Application-owned evaluation settings, not backend/loading configuration."""

    task: Literal["math", "gsm8k"]
    batch_size: int = 1
    num_samples: int = 1
    max_new_tokens: int = 100
    temperature: float = 0.0


def evaluate(
    examples: Sequence[Example],
    generator: TextGenerator,
    *,
    task: Literal["math", "gsm8k"],
    batch_size: int,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    formatter: Callable[[str], str] | None = None,
) -> tuple[list[EvaluationRecord], dict[str, int | float]]:
    """Borrow a text generator; never close it, even on failure.

    Records preserve input order then zero-based backend sample order. Accuracy
    is total_correct / total_count across *all* samples, not pass@k or voting.
    Given identical ordered completions, batching cannot change the result.
    A formatter sees only the question; absent one, the question is the prompt.
    """
    assert examples, "Evaluation requires at least one example"
    assert all(isinstance(e.id, str) and e.id.strip() for e in examples), (
        "Example IDs must be nonempty strings"
    )
    assert len({e.id for e in examples}) == len(examples), "Example IDs must be unique"
    assert task in ("math", "gsm8k"), "Unsupported task"
    for name, value in (
        ("batch_size", batch_size),
        ("num_samples", num_samples),
        ("max_new_tokens", max_new_tokens),
    ):
        assert type(value) is int and value > 0, f"{name} must be a positive integer"
    assert isinstance(temperature, (int, float)) and isfinite(temperature), (
        "temperature must be finite"
    )
    assert temperature >= 0, "temperature must be nonnegative"
    assert temperature > 0 or num_samples == 1, "Greedy decoding needs one sample"

    records = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [
            e.question if formatter is None else formatter(e.question) for e in batch
        ]
        assert all(isinstance(prompt, str) for prompt in prompts), (
            "Formatter must return strings"
        )
        completions = generator.generate(
            prompts,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        assert len(completions) == len(batch), "Outer completion count mismatch"
        assert all(len(group) == num_samples for group in completions), (
            "Inner completion count mismatch"
        )
        for example, prompt, group in zip(batch, prompts, completions, strict=True):
            for sample_index, completion in enumerate(group):
                assert completion.finish_reason in ("stop", "length"), (
                    f"Unsupported finish reason: {completion.finish_reason!r}"
                )
                result = verify_response(
                    completion.text,
                    example.reference_answer,
                    task=task,
                    question=example.question,
                )
                records.append(
                    EvaluationRecord(
                        example_id=example.id,
                        sample_index=sample_index,
                        task=task,
                        question=example.question,
                        prompt=prompt,
                        response=completion.text,
                        finish_reason=completion.finish_reason,
                        extracted_answer=result.extracted_answer,
                        extracted_answers=result.extracted_answers,
                        reference_answer=example.reference_answer,
                        correct=result.correct,
                    )
                )

    total_correct = sum(record.correct for record in records)
    total_count = len(records)
    return records, {
        "total_correct": total_correct,
        "total_count": total_count,
        "num_examples": len(examples),
        "accuracy": total_correct / total_count,
    }


class ClosableTextGenerator(TextGenerator, Protocol):
    """An owned text backend, not the token-level HFInference interface."""

    def close(self) -> None: ...


@contextmanager
def inference_context(
    generator_factory: Callable[[], ClosableTextGenerator],
) -> Iterator[ClosableTextGenerator]:
    """Own exactly the factory-created generator and close it on every exit."""
    generator = generator_factory()
    try:
        yield generator
    finally:
        generator.close()


def main(
    config: EvaluatorConfig,
    *,
    examples: Sequence[Example],
    generator_factory: Callable[[], ClosableTextGenerator],
    output_dir: str | Path,
    formatter: Callable[[str], str] | None = None,
) -> tuple[list[EvaluationRecord], dict[str, int | float]]:
    """Evaluate ready inputs and write artifacts to a new run directory.

    The caller owns dataset/model selection and the factory's backend settings.
    Config captures only evaluation settings; actual prompts are in records.
    Verifier provenance is not model/dataset provenance or a paper-reproduction
    claim. Existing output directories are rejected to avoid mixing runs.
    Write errors propagate and may leave an incomplete directory, never a
    fabricated successful result. Generator cleanup precedes persistence.
    """
    with inference_context(generator_factory) as generator:
        records, metrics = evaluate(
            examples, generator, **asdict(config), formatter=formatter
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    for name, data in (
        ("config", asdict(config)),
        ("metrics", metrics),
        (
            "provenance",
            {
                "verifier_policy": VERIFIER_POLICY,
                "verifier_provenance": VERIFIER_PROVENANCE,
                "prompt_policy": "bare-question-smoke-test"
                if formatter is None
                else "caller-supplied; see records for exact prompts",
                "accuracy_policy": "sample accuracy; no truncation penalty",
            },
        ),
    ):
        (output_dir / f"{name}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return records, metrics
