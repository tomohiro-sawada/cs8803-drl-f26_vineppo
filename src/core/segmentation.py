"""VinePPO MATH/GSM8K segmentation with a shared exact-token adapter.

The low-level MATH/GSM8K adapters accept spans. ``step_boundaries_from_tokens``
constructs them by incrementally decoding the sampled IDs. For both adapters,
``token_spans[i]`` must describe the actual sampled ``response_ids[i]`` in
``text``, using Python character offsets, NOT UTF-8 byte offsets. The caller
must establish this correspondence. Decoding tokens in isolation loses context
needed for whitespace and multi-token Unicode characters.

Spans must cover text without gaps in monotone order. Overlapping tokens
(e.g. byte-fallback tokens sharing a Unicode character) stay together.
Zero-width tokens are allowed only as a trailing
suffix at len(text), e.g. sampled EOS, and belong to the final step. No prompt,
padding, normalization, tokenization or decoding is performed by those adapters.

MATH assigns a token spanning a character cut to the earlier step and advances
the cut to the next complete sampled-token boundary. Cuts that coincide are
merged. GSM8K requires exact character cuts. Both preserve the sampled IDs.
Empty/degenerate partitions and upstream segmentation failures raise.

GSM8K logic condensed from McGill-NLP/VinePPO, revision
9f716e68bb934e71da86c226c1699c36b5553faf, src/treetune/tasks/gsm8k.py,
GSM8K.split_solution_into_intermediate_steps. Copyright (c) 2024 McGill NLP,
MIT license: core/vendor/vineppo/LICENSE.VinePPO.
"""

from bisect import bisect_left
from collections.abc import Callable, Sequence
from functools import partial
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from core.vendor.vineppo.math_extract_steps_inplace import split_solution_inplace

MATH_SEGMENTATION_POLICY = "first_character_sampled_tokens_v1"


def step_boundaries_from_tokens(
    response_ids: Sequence[int],
    tokenizer: "PreTrainedTokenizerBase",
    *,
    dataset: Literal["math", "gsm8k"],
    eos_token_ids: Sequence[int],
    context_ids: Sequence[int] = (),
) -> tuple[int, ...]:
    """Segment sampled IDs using character offsets from incremental decoding.

    context_ids are the prompt tokens the response was sampled after. Decoding
    response prefixes in that context preserves continuation whitespace.
    One leading BOS is dropped, and offsets are response-relative.

    Supply EOS IDs from the generation configuration, not tokenizer defaults.
    One terminal EOS is excluded from text and assigned to the final step;
    other special tokens and empty text raise.
    Returns response-relative token boundaries including 0 and len(response_ids).
    Adjacent boundaries delimit nonempty slices of the original sampled IDs.
    The lower-level adapters accept externally established spans when needed.

    Compatibility caveats (AGENT INSTRUCTIONS):
        Keep the original sampled IDs even when the tokenizer would encode the
        decoded text differently. Derive offsets from decoded sampled prefixes.
        Preserve the pinned upstream splitters' behavior except that GSM8K
        treats everything from the first answer marker onward as one suffix,
        including repeated markers. MATH splitter errors and degenerate GSM8K
        whitespace-only partitions propagate.
        For MATH, advance cuts inside tokens and merge coincident cuts. The same
        resulting sampled-token boundaries define MC states and PPO credit.
        For GSM8K, a cut inside a token still raises. Alignment and upstream
        partition failures raise for both tasks.
    """
    assert dataset in ("math", "gsm8k"), "Unsupported segmentation dataset"
    assert tokenizer.is_fast, "Segmentation requires a fast tokenizer"
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
    decode = partial(
        tokenizer.decode, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    sequence = context + content_ids
    prefix, full = decode(context), decode(sequence)
    assert full.startswith(prefix), "Context does not decode to a prefix"
    offset = len(prefix)
    text = full[offset:]
    # Incomplete byte runs can rewrite earlier decoded characters or yield
    # replacement characters. Work backward to retain the last exact prefix
    # at each character offset and discard transient cuts that later retract.
    cuts = [(len(content_ids), len(text))]
    for i in range(len(content_ids) - 1, 0, -1):
        decoded = decode(sequence[: len(context) + i])
        end = len(decoded) - offset
        if 0 < end < cuts[-1][1] and full.startswith(decoded):
            cuts.append((i, end))
    cuts.append((0, 0))
    spans = [
        (start, end)
        for (a, start), (b, end) in pairwise(reversed(cuts))
        for _ in range(b - a)
    ]
    if has_eos:
        spans.append((len(text), len(text)))
    segment = math_step_boundaries if dataset == "math" else gsm8k_step_boundaries
    try:
        return segment(text, ids, spans)
    except AssertionError as error:
        error.add_note(f"Sampled response token IDs: {ids}")
        raise


def math_step_boundaries(
    text: str,
    response_ids: Sequence[int],
    token_spans: Sequence[tuple[int, int]],
    *,
    round_forward: bool = True,
) -> tuple[int, ...]:
    """Return exact MATH token offsets for ``estimate_vineppo_batch``.

    Implements MATH's answer_prefix=None path. See the module's token-span
    contract; the same validation and exact-token mapping apply to GSM8K.
    Upstream failures (including short inputs such as "x" and "Short.")
    propagate as AssertionError. No single-step fallback is substituted.
    Pass round_forward=False only to replay historical strict diagnostics.
    """
    return _token_step_boundaries(
        text,
        response_ids,
        token_spans,
        split_solution_inplace,
        round_forward=round_forward,
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
    The suffix starting at the first exact ``\\n#### `` answer marker merges with
    the last nonblank reasoning step, including intervening blank lines and
    any repeated answer markers. See the module's shared token-span contract.
    """
    return _token_step_boundaries(
        text, response_ids, token_spans, _gsm8k_character_boundaries
    )


def _gsm8k_character_boundaries(text: str) -> list[int]:
    prefix = "\n#### "
    reasoning, marker, answer = text.partition(prefix)
    steps = reasoning.split("\n")
    first = next((i for i, step in enumerate(steps) if step.strip()), 0)
    steps[: first + 1] = ["\n".join(steps[: first + 1])]
    if marker:
        last = next((i for i in range(len(steps) - 1, -1, -1) if steps[i].strip()), 0)
        steps[last:] = ["\n".join(steps[last:]) + marker + answer]
    assert "\n".join(steps) == text
    return list(
        accumulate((len(step) + (i > 0) for i, step in enumerate(steps)), initial=0)
    )


def _token_step_boundaries(
    text: str,
    response_ids: Sequence[int],
    token_spans: Sequence[tuple[int, int]],
    split_characters: Callable[[str], Sequence[int]],
    *,
    round_forward: bool = False,
) -> tuple[int, ...]:
    assert text and response_ids, "Text and sampled response must be nonempty"
    assert len(response_ids) == len(token_spans), "One span per sampled token required"
    assert all(type(token) is int and token >= 0 for token in response_ids)
    previous_start = covered = 0
    for start, end in token_spans:
        assert type(start) is int and type(end) is int
        assert 0 <= start <= end <= len(text), "Invalid character span"
        assert previous_start <= start <= covered <= end, "Nonmonotone or gapped spans"
        assert start < end or start == len(text), (
            "Only trailing zero-width tokens allowed"
        )
        previous_start, covered = start, end
    assert covered == len(text), "Token spans must cover all response text"

    chars = split_characters(text)
    assert len(chars) >= 2 and chars[0] == 0 and chars[-1] == len(text), (
        "Upstream did not produce a complete nonempty partition"
    )
    assert all(a < b for a, b in pairwise(chars)), "Degenerate upstream partition"
    cuts = {
        end: i
        for i, ((_, end), (start, _)) in enumerate(pairwise(token_spans), 1)
        if end == start < len(text)
    }
    cuts[len(text)] = len(response_ids)
    positions = list(cuts)
    boundaries = [0]
    for char in chars[1:-1]:
        position = positions[bisect_left(positions, char)]
        assert round_forward or position == char, (
            f"Step boundary at character {char} falls inside a sampled token"
        )
        token = cuts[position]
        if token == boundaries[-1] or token == len(response_ids):
            continue
        boundaries.append(token)
    boundaries.append(len(response_ids))
    assert all(a < b for a, b in pairwise(boundaries))
    return tuple(boundaries)
