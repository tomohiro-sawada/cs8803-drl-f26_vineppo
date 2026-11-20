"""Offline evaluator contracts using deterministic text backends and real grading."""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import pytest

from core.data import Example
from core.inference import TextCompletion
from core.verifier import MAX_INPUT_CHARS
from runtime.evaluate import evaluate, inference_context


class FakeGenerator:
    def __init__(self, responses, *, error=None):
        self.responses = responses
        self.error = error
        self.calls = []
        self.close_count = 0

    def generate(self, prompts, *, num_samples, max_new_tokens, temperature):
        assert isinstance(prompts, Sequence)
        self.calls.append((list(prompts), num_samples, max_new_tokens, temperature))
        if self.error is not None:
            raise self.error
        return [list(self.responses[prompt]) for prompt in prompts]

    def close(self):
        self.close_count += 1


@pytest.mark.parametrize("full_solution", [False, True])
def test_math_solution_reference_opt_in(full_solution):
    reference = r"First \boxed{1}, then \boxed{2}." if full_solution else "2"
    examples = [Example("row", "Find both answers.", reference)]
    generator = FakeGenerator(
        {examples[0].question: [TextCompletion(r"\boxed{2}", "stop")]}
    )
    records, metrics = evaluate(
        examples,
        generator,
        task="math",
        batch_size=1,
        num_samples=1,
        max_new_tokens=10,
        temperature=0,
        reference_is_solution=full_solution,
    )
    assert records[0].reference_answer == reference
    assert records[0].correct is (not full_solution)
    assert metrics["accuracy"] == (0.0 if full_solution else 1.0)


@pytest.fixture
def examples():
    return tuple(Example(f"row-{i}", f"Question {i}?", str(i)) for i in range(5))


@pytest.fixture
def settings():
    return {
        "task": "gsm8k",
        "batch_size": 2,
        "num_samples": 2,
        "max_new_tokens": 37,
        "temperature": 0.6,
    }


def responses_for(examples):
    return {
        e.question: [
            TextCompletion("#### wrong", "stop"),
            TextCompletion(f"#### {e.reference_answer}", "length"),
        ]
        for e in examples
    }


@pytest.mark.parametrize("batch_size, sizes", [(2, [2, 2, 1]), (3, [3, 2])])
def test_alignment_sampling_and_uneven_batches(examples, settings, batch_size, sizes):
    generator = FakeGenerator(responses_for(examples))
    settings["batch_size"] = batch_size
    records, metrics = evaluate(examples, generator, **settings)
    assert [len(call[0]) for call in generator.calls] == sizes
    assert all(call[1:] == (2, 37, 0.6) for call in generator.calls)
    assert metrics == {
        "total_correct": 5,
        "total_count": 10,
        "num_examples": 5,
        "accuracy": 0.5,
    }
    for index, record in enumerate(records):
        example = examples[index // 2]
        sample = index % 2
        assert asdict(record) == {
            "example_id": example.id,
            "sample_index": sample,
            "task": "gsm8k",
            "question": example.question,
            "prompt": example.question,
            "response": "#### wrong"
            if sample == 0
            else f"#### {example.reference_answer}",
            "finish_reason": "stop" if sample == 0 else "length",
            "extracted_answer": "wrong" if sample == 0 else example.reference_answer,
            "extracted_answers": (
                "wrong" if sample == 0 else example.reference_answer,
            ),
            "reference_answer": example.reference_answer,
            "correct": sample == 1,
        }
    with pytest.raises(FrozenInstanceError):
        records[0].correct = True
    assert generator.close_count == 0


def test_uneven_batches_use_sample_accuracy(examples, settings):
    examples = examples[:3]
    settings.update(batch_size=2, num_samples=1, temperature=0)
    generator = FakeGenerator(
        {
            e.question: [
                TextCompletion(
                    f"#### {e.reference_answer if i < 2 else 'wrong'}", "stop"
                )
            ]
            for i, e in enumerate(examples)
        }
    )
    records, metrics = evaluate(examples, generator, **settings)
    assert [len(call[0]) for call in generator.calls] == [2, 1]
    assert [r.correct for r in records] == [True, True, False]
    assert metrics == {
        "total_correct": 2,
        "total_count": 3,
        "num_examples": 3,
        "accuracy": 2 / 3,
    }


def test_math_question_aware_multi_answer(settings):
    example = Example("multi", "Give values separated by commas", "1,2")
    generator = FakeGenerator(
        {"Solve the problem.": [TextCompletion(r"\boxed{1,2}", "stop")]}
    )
    settings.update(task="math", num_samples=1, temperature=0)
    records, metrics = evaluate(
        [example],
        generator,
        formatter=lambda question: "Solve the problem.",
        **settings,
    )
    assert records[0].question == example.question
    assert records[0].extracted_answer == "2"
    assert records[0].extracted_answers == ("1", "2")
    assert records[0].correct is True
    assert metrics["accuracy"] == 1.0


def test_repeat_and_batch_invariance(examples, settings):
    generator = FakeGenerator(responses_for(examples))
    expected = evaluate(examples, generator, **settings)
    for batch_size in (1, 2, 3, 8):
        settings["batch_size"] = batch_size
        assert evaluate(examples, generator, **settings) == expected
    assert examples == tuple(
        Example(f"row-{i}", f"Question {i}?", str(i)) for i in range(5)
    )


def test_formatter_receives_question_only_and_preserves_duplicate_prompts(settings):
    examples = [Example("a", "same?", "1"), Example("b", "same?", "2")]
    seen = []

    def formatter(question):
        seen.append(question)
        return "Solve: " + question

    generator = FakeGenerator({"Solve: same?": [TextCompletion("#### 1", "stop")]})
    settings.update(num_samples=1, temperature=0)
    records, metrics = evaluate(examples, generator, formatter=formatter, **settings)
    assert seen == ["same?", "same?"]
    assert generator.calls == [(["Solve: same?", "Solve: same?"], 1, 37, 0)]
    assert [r.example_id for r in records] == ["a", "b"]
    assert [r.correct for r in records] == [True, False]
    assert metrics["accuracy"] == 0.5


@pytest.mark.parametrize(
    "task, correct_text, malformed",
    [
        ("gsm8k", "work\n#### 7", "no final answer"),
        ("math", r"work \boxed{7}", ""),
    ],
)
def test_real_verifier_malformed_and_truncated(settings, task, correct_text, malformed):
    example = Example("id", "Compute 3 + 4.", "7")
    completions = [
        TextCompletion(malformed, "stop"),
        TextCompletion(malformed, "length"),
        TextCompletion(correct_text, "stop"),
        TextCompletion(correct_text, "length"),
    ]
    settings.update(task=task, num_samples=4)
    records, metrics = evaluate(
        [example], FakeGenerator({example.question: completions}), **settings
    )
    assert [r.correct for r in records] == [False, False, True, True]
    assert [r.extracted_answer for r in records] == [None, None, "7", "7"]
    assert [r.extracted_answers for r in records] == [(), (), ("7",), ("7",)]
    assert metrics == {
        "total_correct": 2,
        "total_count": 4,
        "num_examples": 1,
        "accuracy": 0.5,
    }


@pytest.mark.parametrize(
    "examples, message",
    [
        ([], "at least one"),
        ([Example("", "q", "1")], "nonempty"),
        ([Example("  ", "q", "1")], "nonempty"),
        ([Example("a", "q", "1"), Example("a", "other", "2")], "unique"),
    ],
)
def test_invalid_selection(examples, message, settings):
    generator = FakeGenerator({})
    with pytest.raises(AssertionError, match=message):
        evaluate(examples, generator, **settings)
    assert not generator.calls


@pytest.mark.parametrize(
    "field, value",
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", 1.5),
        ("batch_size", True),
        ("num_samples", 0),
        ("num_samples", 1.5),
        ("num_samples", True),
        ("max_new_tokens", 0),
        ("max_new_tokens", -1),
        ("max_new_tokens", 1.5),
        ("temperature", -1),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", "0"),
        ("temperature", 0),
        ("task", "unknown"),
    ],
)
def test_invalid_settings(examples, settings, field, value):
    settings[field] = value
    generator = FakeGenerator({})
    with pytest.raises(AssertionError):
        evaluate(examples, generator, **settings)
    assert not generator.calls


@pytest.mark.parametrize(
    "outer, inner, message",
    [
        (1, 2, "Outer"),
        (3, 2, "Outer"),
        (2, 1, "Inner"),
        (2, 3, "Inner"),
        (0, 2, "Outer"),
        (2, 0, "Inner"),
    ],
)
def test_exact_cardinality(examples, settings, outer, inner, message):
    class WrongCounts(FakeGenerator):
        def generate(self, prompts, **kwargs):
            return [[TextCompletion("#### 0", "stop")] * inner for _ in range(outer)]

    generator = WrongCounts({})
    with pytest.raises(AssertionError, match=message):
        evaluate(examples, generator, **settings)
    assert generator.close_count == 0


@pytest.mark.parametrize("reason", ["error", "content_filter", "", None])
def test_structural_completion_invalid_finish_reason_before_verification(
    settings, reason
):
    example = Example("id", "Question?", "1")
    # Oversized text would fail real verification if the boundary check ran too late.
    completion = SimpleNamespace(text="x" * (MAX_INPUT_CHARS + 1), finish_reason=reason)
    generator = FakeGenerator({example.question: [completion]})
    settings.update(num_samples=1, temperature=0)
    with pytest.raises(AssertionError, match="Unsupported finish reason"):
        evaluate([example], generator, **settings)
    assert generator.close_count == 0


def test_formatter_errors_propagate(examples, settings):
    generator = FakeGenerator({})

    def broken(question):
        raise RuntimeError("format failure")

    with pytest.raises(RuntimeError, match="format failure"):
        evaluate(examples, generator, formatter=broken, **settings)
    with pytest.raises(AssertionError, match="Formatter"):
        evaluate(examples, generator, formatter=lambda q: None, **settings)
    assert not generator.calls
    assert generator.close_count == 0


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("failure", ["backend", "verifier", "cardinality"])
def test_failures_and_ownership(examples, settings, owned, failure):
    responses = responses_for(examples)
    error = RuntimeError("backend failure") if failure == "backend" else None
    if failure == "verifier":
        responses[examples[0].question][0] = TextCompletion(
            "x" * (MAX_INPUT_CHARS + 1), "stop"
        )
    if failure == "cardinality":
        responses[examples[0].question] = []
    generator = FakeGenerator(responses, error=error)
    expected_error = {
        "backend": RuntimeError,
        "verifier": ValueError,
        "cardinality": AssertionError,
    }[failure]
    with pytest.raises(expected_error) as caught:
        if owned:
            with inference_context(lambda: generator) as opened:
                evaluate(examples, opened, **settings)
        else:
            evaluate(examples, generator, **settings)
    if error is not None:
        assert caught.value is error
    assert generator.close_count == int(owned)


def test_factory_and_close_failures_propagate():
    def broken_factory():
        raise RuntimeError("factory failure")

    with (
        pytest.raises(RuntimeError, match="factory failure"),
        inference_context(broken_factory),
    ):
        pytest.fail("Factory must fail before yielding")

    class BrokenClose(FakeGenerator):
        def close(self):
            super().close()
            raise RuntimeError("close failure")

    generator = BrokenClose({})
    with (
        pytest.raises(RuntimeError, match="close failure"),
        inference_context(lambda: generator) as borrowed,
    ):
        assert borrowed is generator
    assert generator.close_count == 1
