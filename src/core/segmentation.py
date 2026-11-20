"""VinePPO MATH/GSM8K segmentation with a shared exact-token adapter.

The low-level MATH/GSM8K adapters accept spans; ``step_boundaries_from_tokens``
constructs verified spans using a fast tokenizer. For the low-level adapters,
``token_spans[i]`` must describe the actual sampled ``response_ids[i]`` in
``text``, using Python character offsets, NOT UTF-8 byte offsets. The caller
must establish this correspondence; lengths alone cannot prove it. Offsets
from re-encoding text are valid only after checking the entire encoded ID
sequence against the sampled IDs (with explicit special-token handling).
Independently decoded token lengths are not a valid source of offsets.

Spans must cover text without gaps in monotone order. Overlap is allowed
(e.g. byte-fallback tokens sharing a Unicode character), but a step cannot
split overlapping tokens. Zero-width tokens are allowed only as a trailing
suffix at len(text), e.g. sampled EOS, and belong to the final step. No prompt,
padding, normalization, tokenization or decoding is performed by those adapters.

Character boundaries are preserved exactly: a boundary inside a token raises,
rather than snapping, merging steps, or changing token IDs. Empty/degenerate
partitions and upstream segmentation failures also raise.

GSM8K logic condensed from McGill-NLP/VinePPO, revision
9f716e68bb934e71da86c226c1699c36b5553faf, src/treetune/tasks/gsm8k.py,
GSM8K.split_solution_into_intermediate_steps. Copyright (c) 2024 McGill NLP,
MIT license: core/vendor/vineppo/LICENSE.VinePPO.
"""

from bisect import bisect_left
from collections.abc import Callable, Sequence
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from core.vendor.vineppo.math_extract_steps_inplace import split_solution_inplace


def step_boundaries_from_tokens(
    response_ids: Sequence[int],
    tokenizer: "PreTrainedTokenizerBase",
    *,
    dataset: Literal["math", "gsm8k"],
    eos_token_ids: Sequence[int],
    context_ids: Sequence[int] = (),
) -> tuple[int, ...]:
    """Segment sampled IDs using verified fast-tokenizer character offsets.

    context_ids are the prompt tokens the response was sampled after. They are
    decoded and re-encoded together with the response, because tokenizers such
    as Llama's encode standalone text differently from a continuation (a dummy
    prefix token). One leading BOS is dropped; offsets are response-relative.

    Supply EOS IDs from the generation configuration, not tokenizer defaults.
    One terminal EOS is excluded from text and assigned to the final step;
    other special tokens, empty text, non-roundtripping sampled IDs, uncovered
    whitespace offsets, and character cuts inside tokens fail loudly. This
    intentionally does not retokenize steps or approximate sampled prefixes.
    Returns response-relative token boundaries including 0 and len(response_ids).
    Adjacent boundaries delimit nonempty slices of the original sampled IDs.
    The lower-level adapters accept externally established spans when needed.

    Compatibility caveats (AGENT INSTRUCTIONS):
        Preserve the pinned upstream splitter's behavior, not just Appendix B's
        prose description. MATH inputs such as "x" and "Short." raise
        AssertionError upstream; repeated multiline math placeholders can also
        fail. GSM8K newline-only inputs can produce degenerate partitions,
        which this adapter rejects. These are not repaired or skipped.
        A valid character split may fall inside one sampled token. That also
        raises AssertionError: rounding the cut, merging steps, or re-encoding
        each step would change the trajectory/advantage assignment. Likewise,
        identical decoded text does not establish identical sampled token IDs.
        Do not add silent recovery for these cases; changing this policy requires
        an explicit compatibility decision and updated regression tests in
        tests/core/test_segmentation.py.
    """
    assert dataset in ("math", "gsm8k"), "Unsupported segmentation dataset"
    assert tokenizer.is_fast, "Exact offsets require a fast tokenizer"
    ids = list(response_ids)
    assert ids and all(type(token) is int and token >= 0 for token in ids)
    assert all(type(token) is int and token >= 0 for token in eos_token_ids)
    has_eos = ids[-1] in eos_token_ids
    content_ids = ids[:-1] if has_eos else ids
    assert not set(content_ids).intersection(
        set(tokenizer.all_special_ids) | set(eos_token_ids)
    ), "Only one terminal EOS is supported; response contains other special tokens"
    context = list(context_ids)
    if context and context[0] == tokenizer.bos_token_id:
        context = context[1:]
    assert not set(context).intersection(tokenizer.all_special_ids), (
        "Context contains special tokens other than one leading BOS"
    )
    prefix, full = (
        tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for ids in (context, context + content_ids)
    )
    assert full.startswith(prefix), "Context does not decode to a prefix"
    text = full[len(prefix) :]
    encoded = tokenizer(
        full,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    assert encoded["input_ids"] == context + content_ids, (
        "Decoded text does not roundtrip to the exact sampled token IDs"
    )
    spans = [
        (start - len(prefix), end - len(prefix))
        for start, end in encoded["offset_mapping"][len(context) :]
    ]
    if has_eos:
        spans.append((len(text), len(text)))
    segment = math_step_boundaries if dataset == "math" else gsm8k_step_boundaries
    return segment(text, ids, spans)


def math_step_boundaries(
    text: str,
    response_ids: Sequence[int],
    token_spans: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Return exact MATH token offsets for ``estimate_vineppo``.

    Implements MATH's answer_prefix=None path. See the module's token-span
    contract; the same validation and exact-token mapping apply to GSM8K.
    Upstream failures (including short inputs such as "x" and "Short.")
    propagate as AssertionError. No single-step fallback is substituted.
    See step_boundaries_from_tokens for the compatibility policy.
    """
    return _token_step_boundaries(
        text, response_ids, token_spans, split_solution_inplace
    )


def gsm8k_step_boundaries(
    text: str,
    response_ids: Sequence[int],
    token_spans: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Return exact GSM8K token offsets using upstream original-format defaults.

    Split only on literal LF (not periods, math, CR, or Unicode line separators).
    Newlines belong to the following step; leading whitespace-only lines merge
    into the first nonblank step. Interior/trailing blank lines remain steps.
    The exact ``\\n#### `` answer suffix, when present, merges with the last
    nonblank reasoning step, including intervening blank lines. Multiple answer
    prefixes raise ValueError, as upstream does. Without that exact prefix, no
    answer-specific merging occurs. See the module's shared token-span contract.
    """
    return _token_step_boundaries(
        text, response_ids, token_spans, _gsm8k_character_boundaries
    )


def _gsm8k_character_boundaries(text: str) -> list[int]:
    prefix = "\n#### "
    pieces = text.split(prefix)
    reasoning, answer = (text, None) if len(pieces) == 1 else pieces
    steps = reasoning.split("\n")
    first = next((i for i, step in enumerate(steps) if step.strip()), 0)
    steps[: first + 1] = ["\n".join(steps[: first + 1])]
    if answer is not None:
        last = next((i for i in range(len(steps) - 1, -1, -1) if steps[i].strip()), 0)
        steps[last:] = ["\n".join(steps[last:]) + prefix + answer]
    assert "\n".join(steps) == text
    return list(
        accumulate((len(step) + (i > 0) for i, step in enumerate(steps)), initial=0)
    )


def _token_step_boundaries(
    text: str,
    response_ids: Sequence[int],
    token_spans: Sequence[tuple[int, int]],
    split_characters: Callable[[str], Sequence[int]],
) -> tuple[int, ...]:
    assert text and response_ids, "Text and sampled response must be nonempty"
    assert len(response_ids) == len(token_spans), "One span per sampled token required"
    assert all(type(token) is int and token >= 0 for token in response_ids)
    previous_start = covered = 0
    starts = []
    for start, end in token_spans:
        assert type(start) is int and type(end) is int
        assert 0 <= start <= end <= len(text), "Invalid character span"
        assert previous_start <= start <= covered <= end, "Nonmonotone or gapped spans"
        assert start < end or start == len(text), (
            "Only trailing zero-width tokens allowed"
        )
        starts.append(start)
        previous_start, covered = start, end
    assert covered == len(text), "Token spans must cover all response text"

    chars = split_characters(text)
    assert len(chars) >= 2 and chars[0] == 0 and chars[-1] == len(text), (
        "Upstream did not produce a complete nonempty partition"
    )
    assert all(a < b for a, b in pairwise(chars)), "Degenerate upstream partition"
    boundaries = [0]
    for char in chars[1:-1]:
        token = bisect_left(starts, char)
        assert (
            0 < token < len(token_spans)
            and token_spans[token - 1][1] == char == starts[token]
        ), f"Step boundary at character {char} falls inside a sampled token"
        boundaries.append(token)
    boundaries.append(len(response_ids))
    assert all(a < b for a, b in pairwise(boundaries))
    return tuple(boundaries)
