"""Offline regressions for pinned upstream semantics and exact sampled-token cuts."""

import hashlib
import json
import random
from itertools import pairwise

import pytest

from core.segmentation import (
    gsm8k_step_boundaries,
    math_step_boundaries,
    step_boundaries_from_tokens,
)
from core.vendor.vineppo.math_extract_steps_inplace import split_solution_inplace


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", [0]),
        ("x" * 20, [0, 20]),
        ("x" * 101, [0, 101]),  # 100 is a target, not a hard length cap.
        ("x" * 20 + "\n", [0, 21]),
        ("First sentence is long enough. Second sentence is long enough.", [0, 30, 62]),
        (
            "First sentence is long enough.\n\nSecond sentence is long enough.",
            [0, 30, 63],
        ),
        ("First sentence is long enough. Next.", [0, 36]),
        ("$x=1.$ Then we get the result of our calculation.", [0, 6, 49]),
        (r"\begin{align}x&=1\\y&=2\end{align}", [0, 34]),
        ("Start with sufficient explanation.\n$\na=1\nb=2\n$", [0, 36, 40, 46]),
        (
            "Leading explanation of sufficient length.\n[asy]\nx=1;\ny=2;\n[/asy]",
            [0, 41, 52, 57, 64],
        ),
        ("α" * 25 + "\n" + "β" * 25, [0, 25, 51]),
        *[
            ("a" * 60 + sep + "b" * 60, [0, cut, 120 + len(sep)])
            for sep, cut in [(", ", 61), (" and ", 64), (": ", 61), (" to ", 63)]
        ],
        *[
            ("$" + "a" * 60 + sep + "b" * 60 + "$", [0, cut, 122 + len(sep)])
            for sep, cut in [("=", 62), (r"\equiv", 67), ("+", 62), (r"\cancelto", 70)]
        ],
    ],
)
def test_upstream_character_boundaries(text, expected):
    assert split_solution_inplace(text) == expected
    assert "".join(text[a:b] for a, b in pairwise(expected)) == text


@pytest.mark.parametrize("text", ["x", "Short.", "x" * 19, "$a\nb$ and $a\nb$"])
def test_upstream_failures_are_not_repaired(text):
    with pytest.raises(AssertionError):
        split_solution_inplace(text)


def _parity_corpus():
    rng = random.Random(83)
    fragments = [
        "abc",
        "A. ",
        "First sentence has enough characters.",
        "\n",
        "\n\n",
        " ",
        "$x=1.$",
        "$a\nb$",
        r"\[x+y=z\]",
        r"\begin{align}x=1\end{align}",
        "[asy]\nx=1;\n[/asy]",
        ",",
        " and ",
        ":",
        " to ",
        "=",
        "+",
        r"\equiv",
        r"\cancelto",
        "αβ😀",
        "z" * 105,
    ]
    return [
        "".join(rng.choices(fragments, k=rng.randrange(1, 45))) for _ in range(5000)
    ]


def _fingerprint(splitter):
    outcomes = []
    for text in _parity_corpus():
        try:
            outcomes.append(["ok", splitter(text)])
        except AssertionError:
            outcomes.append(["error", "AssertionError"])
    return hashlib.sha256(json.dumps(outcomes).encode()).hexdigest()


def test_pinned_upstream_corpus_fingerprint():
    # Computed from the original Git blob 2c37633a..., not from this refactor.
    assert _fingerprint(split_solution_inplace) == (
        "bf8b192ce89466a4f4c50894705ddae8721572eb8405a1254bc787e8853e4251"
    )


def test_exact_sampled_ids_and_eos_are_not_reencoded():
    text = "First sentence is long enough. Second sentence is long enough."
    ids = [900, 7, 3, 51, 2]
    spans = [(0, 8), (8, 30), (30, 45), (45, 62), (62, 62)]
    assert math_step_boundaries(text, ids, spans) == (0, 2, 5)
    assert ids == [900, 7, 3, 51, 2]
    assert spans == [(0, 8), (8, 30), (30, 45), (45, 62), (62, 62)]


@pytest.mark.parametrize("segment", [math_step_boundaries, gsm8k_step_boundaries])
def test_unicode_overlaps_are_valid_away_from_step_cuts(segment):
    text = "😀" + "a" * 24 + "\n" + "β" * 25
    spans = [(0, 1), (0, 1), (1, 25), (25, 26), (26, 51), (51, 51)]
    assert segment(text, [4, 5, 6, 7, 8, 2], spans) == (0, 3, 6)


@pytest.mark.parametrize("spans", [[(0, 31), (31, 62)], [(0, 30), (29, 62)]])
def test_straddling_or_overlapping_step_cut_is_rejected(spans):
    with pytest.raises(AssertionError, match="inside a sampled token"):
        math_step_boundaries(
            "First sentence is long enough. Second sentence is long enough.",
            [17, 18],
            spans,
        )


@pytest.mark.parametrize(
    ("text", "ids", "spans"),
    [
        ("", [2], [(0, 0)]),
        ("x" * 20, [], []),
        ("x" * 20, [1, 2], [(0, 20)]),
        ("x" * 20, [-1], [(0, 20)]),
        ("x" * 20, [True], [(0, 20)]),
        ("x" * 20, [1], [(0, 19)]),
        ("x" * 20, [1], [(1, 20)]),
        ("x" * 20, [1], [(0, 21)]),
        ("x" * 20, [1], [(False, 20)]),
        ("x" * 20, [1, 2], [(0, 9), (10, 20)]),
        ("x" * 20, [1, 2], [(0, 20), (10, 19)]),
        ("x" * 20, [1, 2], [(0, 0), (0, 20)]),
        ("x" * 20, [1, 2, 3], [(0, 10), (10, 10), (10, 20)]),
    ],
)
@pytest.mark.parametrize("segment", [math_step_boundaries, gsm8k_step_boundaries])
def test_invalid_alignment_fails_loudly(text, ids, spans, segment):
    with pytest.raises(AssertionError):
        segment(text, ids, spans)


@pytest.mark.parametrize(
    ("text", "steps"),
    [
        ("x", ["x"]),
        ("First. Second. $x=1.$", ["First. Second. $x=1.$"]),
        ("x" * 200, ["x" * 200]),
        ("a\nb", ["a", "\nb"]),
        ("a\n\nb", ["a", "\n", "\nb"]),
        ("a\n \nb", ["a", "\n ", "\nb"]),
        ("\n\n a\nb", ["\n\n a", "\nb"]),
        (" \n\t\nx\ny", [" \n\t\nx", "\ny"]),
        ("a\n", ["a", "\n"]),
        ("a\n\n", ["a", "\n", "\n"]),
        ("a\r\nb", ["a\r", "\nb"]),
        ("a\rb\u2028c", ["a\rb\u2028c"]),
        ("α\n😀", ["α", "\n😀"]),
        (" \t", [" \t"]),
        ("a\nb\n#### 2", ["a", "\nb\n#### 2"]),
        ("a\nb\n\n#### 2", ["a", "\nb\n\n#### 2"]),
        ("a\n\n#### 2", ["a\n\n#### 2"]),
        ("\n#### 2", ["\n#### 2"]),
        (" \n\n#### 2", [" \n\n#### 2"]),
        ("a\n#### 2\nextra", ["a\n#### 2\nextra"]),
        ("a\n#### ", ["a\n#### "]),
        ("a\n####2", ["a", "\n####2"]),
        ("#### 2\na", ["#### 2", "\na"]),
    ],
)
@pytest.mark.parametrize("eos", [False, True])
def test_gsm8k_newline_semantics(text, steps, eos):
    assert "".join(steps) == text
    ids = list(range(100, 100 + len(text))) + ([2] if eos else [])
    spans = [(i, i + 1) for i in range(len(text))]
    if eos:
        spans.append((len(text), len(text)))
    boundaries = gsm8k_step_boundaries(text, ids, spans)
    expected = [0]
    for step in steps:
        expected.append(expected[-1] + len(step))
    expected[-1] += eos
    assert boundaries == tuple(expected)
    assert ids == list(range(100, 100 + len(text))) + ([2] if eos else [])


@pytest.mark.parametrize("text", ["\n", "\n\n", "\n \n"])
def test_gsm8k_upstream_degenerate_whitespace_partition_is_rejected(text):
    with pytest.raises(AssertionError, match="Degenerate upstream partition"):
        gsm8k_step_boundaries(
            text, list(range(len(text))), [(i, i + 1) for i in range(len(text))]
        )


def test_gsm8k_repeated_answer_prefix_retains_upstream_failure():
    text = "a\n#### 1\n#### 2"
    with pytest.raises(ValueError):
        gsm8k_step_boundaries(text, [10], [(0, len(text))])


@pytest.mark.parametrize("spans", [[(0, 2), (2, 3)], [(0, 1), (0, 3)]])
def test_gsm8k_newline_cut_inside_sampled_token_is_rejected(spans):
    with pytest.raises(AssertionError, match="inside a sampled token"):
        gsm8k_step_boundaries("a\nb", [10, 11], spans)


def test_gsm8k_exact_token_boundaries_with_trailing_eos():
    ids = [900, 7, 51, 2]
    spans = [(0, 5), (5, 6), (6, 10), (10, 10)]
    assert gsm8k_step_boundaries("first\nlast", ids, spans) == (0, 1, 4)
    assert ids == [900, 7, 51, 2]
    assert spans == [(0, 5), (5, 6), (6, 10), (10, 10)]


def test_estimator_uses_exact_original_token_prefixes():
    from algorithms.vineppo import estimate_vineppo, vineppo_advantage_method
    from core.data import Episode
    from core.inference import Completion

    episode = Episode([99, 98], [900, 7, 3, 51, 2], 1.0, "stop")
    text = "First sentence is long enough. Second sentence is long enough."
    boundaries = math_step_boundaries(
        text, episode.response_ids, [(0, 8), (8, 30), (30, 45), (45, 62), (62, 62)]
    )
    seen_prefixes = []
    scored = []

    class Inference:
        def generate(self, prefixes: list[list[int]], **kwargs):
            assert kwargs["num_samples"] == 1
            seen_prefixes.extend(prefixes)
            return [[Completion([42, 2], "stop")]]

    def score(ids, finish):
        assert finish == "stop"
        scored.append(ids)
        return 0.5

    estimate = estimate_vineppo(
        "test",
        episode,
        step_boundaries=boundaries,
        inference=Inference(),
        score_response=score,
        num_samples=1,
        max_new_tokens=2,
        temperature=1.0,
    )
    assert seen_prefixes == [[99, 98], [99, 98, 900, 7]]
    assert scored == [[42, 2], [900, 7, 42, 2]]
    assert vineppo_advantage_method(estimate) == [0.0, 0.0, 0.5, 0.5, 0.5]


@pytest.fixture
def byte_tokenizer():
    # Real Rust/HF byte-level offsets, without downloads or model weights.
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>"
    )


@pytest.mark.parametrize("dataset", ["math", "gsm8k"])
@pytest.mark.parametrize("eos", [False, True])
def test_real_tokenizer_segmentation(byte_tokenizer, dataset, eos):
    first = "α😀 This explanation is long enough."
    text = first + "\nAnd this is the second reasoning step."
    ids = byte_tokenizer.encode(text, add_special_tokens=False)
    if eos:
        ids.append(byte_tokenizer.eos_token_id)
    original = ids.copy()
    boundaries = step_boundaries_from_tokens(
        ids,
        byte_tokenizer,
        dataset=dataset,
        eos_token_ids=[byte_tokenizer.eos_token_id],
    )
    assert boundaries == (0, len(first.encode("utf-8")), len(ids))
    assert ids == original
    assert byte_tokenizer.decode(ids[: boundaries[1]]) == first


@pytest.mark.parametrize(
    ("dataset", "steps"),
    [
        pytest.param(
            "gsm8k", ["First. Second. $x=1.$"], id="gsm8k-no-punctuation-split"
        ),
        pytest.param(
            "gsm8k",
            ["First step", "\nSecond step", "\nThird step"],
            id="gsm8k-newlines",
        ),
        pytest.param(
            "gsm8k", [" \n\nFirst", "\n", "\nSecond", "\n"], id="gsm8k-blank-lines"
        ),
        pytest.param(
            "gsm8k", ["Compute 2 + 2", "\nThe result is 4\n\n#### 4"], id="gsm8k-answer"
        ),
        pytest.param("gsm8k", ["α = 2 😀", "\nβ = 4"], id="gsm8k-unicode"),
        pytest.param(
            "math",
            ["First sentence is long enough.", " Second sentence is long enough."],
            id="math-period",
        ),
        pytest.param(
            "math",
            ["First sentence is long enough.", "\n\nSecond sentence is long enough."],
            id="math-newlines",
        ),
        pytest.param(
            "math",
            ["$x=1.$", " Then we get the result of our calculation."],
            id="math-tail-period",
        ),
        pytest.param(
            "math",
            ["We substitute $x=1, y=2$ into the given expression."],
            id="math-protected-comma",
        ),
        pytest.param("math", ["a" * 60 + ",", " " + "b" * 60], id="math-long-language"),
        pytest.param(
            "math", ["$" + "a" * 60 + "=", "b" * 60 + "$"], id="math-long-equation"
        ),
        pytest.param("math", ["x" * 101], id="math-unsplittable-long-step"),
    ],
)
@pytest.mark.parametrize("eos", [False, True], ids=["length", "eos"])
def test_token_boundaries_match_manual_steps(byte_tokenizer, dataset, steps, eos):
    # The strings above are the oracle: never derive expected cuts with a splitter.
    text = "".join(steps)
    ids = byte_tokenizer.encode(text, add_special_tokens=False)
    if eos:
        ids.append(byte_tokenizer.eos_token_id)
    original = ids.copy()
    boundaries = step_boundaries_from_tokens(
        ids,
        byte_tokenizer,
        dataset=dataset,
        eos_token_ids=[byte_tokenizer.eos_token_id],
    )
    slices = [ids[a:b] for a, b in pairwise(boundaries)]
    assert boundaries[0] == 0 and boundaries[-1] == len(ids)
    assert all(slices)
    assert [token for part in slices for token in part] == original
    assert ids == original
    assert [
        byte_tokenizer.decode(
            part, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for part in slices
    ] == steps
    assert all(byte_tokenizer.eos_token_id not in part for part in slices[:-1])
    if eos:
        assert slices[-1][-1] == byte_tokenizer.eos_token_id


@pytest.mark.parametrize(
    ("dataset", "text"),
    [
        ("math", "x"),
        ("math", "Short."),
        ("math", "$a\nb$ and $a\nb$"),
        ("gsm8k", "\n\n"),
    ],
)
def test_token_entry_point_preserves_segmentation_failures(
    byte_tokenizer, dataset, text
):
    ids = byte_tokenizer.encode(text, add_special_tokens=False)
    with pytest.raises(AssertionError):
        step_boundaries_from_tokens(
            ids, byte_tokenizer, dataset=dataset, eos_token_ids=[]
        )


def test_same_text_different_sampled_ids_is_rejected(byte_tokenizer):
    ids = byte_tokenizer.encode("ab", add_special_tokens=False)
    byte_tokenizer.add_tokens(["ab"])
    assert byte_tokenizer.decode(ids) == "ab"
    with pytest.raises(AssertionError, match="exact sampled token IDs"):
        step_boundaries_from_tokens(
            ids, byte_tokenizer, dataset="gsm8k", eos_token_ids=[]
        )


def test_real_tokenizer_token_straddling_newline_is_rejected(byte_tokenizer):
    byte_tokenizer.add_tokens(["a\nb"])
    ids = byte_tokenizer.encode("a\nb", add_special_tokens=False)
    with pytest.raises(AssertionError, match="inside a sampled token"):
        step_boundaries_from_tokens(
            ids, byte_tokenizer, dataset="gsm8k", eos_token_ids=[]
        )


@pytest.mark.parametrize("special", ["eos_token_id", "pad_token_id"])
def test_nonterminal_special_token_is_rejected(byte_tokenizer, special):
    ids = [getattr(byte_tokenizer, special)] + byte_tokenizer.encode(
        "reasoning", add_special_tokens=False
    )
    with pytest.raises(AssertionError, match="other special tokens"):
        step_boundaries_from_tokens(
            ids,
            byte_tokenizer,
            dataset="gsm8k",
            eos_token_ids=[byte_tokenizer.eos_token_id],
        )


def test_eos_only_response_is_rejected(byte_tokenizer):
    with pytest.raises(AssertionError, match="nonempty"):
        step_boundaries_from_tokens(
            [byte_tokenizer.eos_token_id],
            byte_tokenizer,
            dataset="gsm8k",
            eos_token_ids=[byte_tokenizer.eos_token_id],
        )


def test_prompt_context_resolves_standalone_encoding_mismatch(byte_tokenizer):
    # Like Llama's dummy prefix: a response is not the canonical encoding of its
    # own text, but is the canonical continuation of the prompt it followed.
    byte_tokenizer.add_tokens(["Q:a"])
    prompt = byte_tokenizer.encode("Q:", add_special_tokens=False)
    response = byte_tokenizer.encode(
        "a first step\nsecond step", add_special_tokens=False
    )
    merged = byte_tokenizer.encode(
        "Q:a first step\nsecond step", add_special_tokens=False
    )
    assert merged != prompt + response
    with pytest.raises(AssertionError, match="exact sampled token IDs"):
        step_boundaries_from_tokens(
            response,
            byte_tokenizer,
            dataset="gsm8k",
            eos_token_ids=[],
            context_ids=prompt,
        )
    byte_tokenizer_response = byte_tokenizer.encode(
        "first step\nsecond step", add_special_tokens=False
    )
    plain = step_boundaries_from_tokens(
        byte_tokenizer_response, byte_tokenizer, dataset="gsm8k", eos_token_ids=[]
    )
    assert plain == step_boundaries_from_tokens(
        byte_tokenizer_response,
        byte_tokenizer,
        dataset="gsm8k",
        eos_token_ids=[],
        context_ids=byte_tokenizer.encode("Q: ", add_special_tokens=False),
    )
    assert plain == (0, len("first step"), len(byte_tokenizer_response))


def test_context_with_special_tokens_is_rejected(byte_tokenizer):
    ids = byte_tokenizer.encode("first step\nsecond", add_special_tokens=False)
    with pytest.raises(AssertionError, match="Context contains special tokens"):
        step_boundaries_from_tokens(
            ids,
            byte_tokenizer,
            dataset="gsm8k",
            eos_token_ids=[],
            context_ids=[byte_tokenizer.pad_token_id, *ids],
        )
