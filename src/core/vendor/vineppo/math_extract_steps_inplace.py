"""Condensed VinePPO in-place MATH segmentation (not the text-rewriting splitter).

Source: McGill-NLP/VinePPO, revision 9f716e68bb934e71da86c226c1699c36b5553faf,
src/treetune/tasks/math_extract_steps_inplace.py (blob
2c37633abfa96f814317f1515b30df0a347b2672), with md5_hash from
math_extract_steps.py (blob 3963f35c44922d45047f1cd3c69f92cd9a75472f).
Copyright (c) 2024 McGill NLP. MIT license: adjacent LICENSE.VinePPO.

Local changes: inline hashing, remove unused placeholder types/helpers, factor
balanced splitting and cumulative offsets, retain the stage order and quirks.
In particular, placeholder length is 60, recovery is insertion-ordered, repeated
placeholders use str.replace, and the math exclusion range uses LOCAL offsets.
These are compatibility semantics, not fixes. Short inputs can fail upstream's final assertion; empty text returns [0].
The public token adapter requires a nonempty, complete partition.
"""

import hashlib
import re
from itertools import accumulate, pairwise

_MATH = (
    r"\[asy\].*?\[/asy\]",
    r"\$\\begin\{tabular\}.*?\\end\{tabular\}\$",
    r"\\begin\{tabular\}.*?\\end\{tabular\}",
    r"\\\((.*?)\\\)",
    r"\\\[.*?\\\]",
    r"\$\$(.*?)\$\$",
    r"(?<!\\)\$(.*?[^\\])\$(?!\$)",
    r"\\begin\{([^}]*)\}.*?\\end\{\1\}",
)
_PLACEHOLDER = re.compile(r"<\|PLACEHOLDER_[0-9a-f]{32}_PLACEHOLDER\|>")
_PERIOD = (
    r"(?<!\b[A-Za-z]\.)(?<!\s[A-Za-z]\.)(?<!\s[A-Za-z][A-Za-z]\.)(?<=\.)(?=\s+[A-Z])"
)
_TAILS = (r".$", r".$$", r".\)", r".\]")
_OPENINGS = (r"\$", r"\$\$", r"\\\(", r"\\\[", r"\\begin\{[^}]*\}", r"\[asy\]")
_LANGUAGE_SPLITS = ((r",", 1), (r"\sand\s", 4), (r":", 1), (r"\sto\s", 3))
_MATH_SPLITS = (
    (r",\$", 2),
    (r",\\\)", 3),
    (r",\\\]", 3),
    (r"=", 1),
    (r"\\equiv", 6),
    (r"\+", 1),
    (r"\\cancelto", 9),
)


def _disguise(text):
    placeholders = {}

    def replace(match):
        value = match.group()
        key = f"<|PLACEHOLDER_{hashlib.md5(value.encode('utf-8')).hexdigest()}_PLACEHOLDER|>"
        placeholders[key] = value
        return key

    for pattern in _MATH:
        text = re.sub(pattern, replace, text, flags=re.DOTALL)
    return text, placeholders


def _recover(text, placeholders):
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _offsets(parts):
    return list(accumulate(map(len, parts), initial=0))


def _parts(text, indices):
    return [text[a:b] for a, b in pairwise(indices)]


def _assert_placeholders(parts):
    for part in parts:
        assert "<|PLACEHOLDER_" not in part or "PLACEHOLDER|>" in part, (
            "Cut into a placeholder"
        )


def _initial_indices(text):
    parts = []
    for sentence in re.split(_PERIOD, text):
        parts.extend(
            p if i == 0 else "\n" + p for i, p in enumerate(sentence.split("\n"))
        )
    i = 0
    while i < len(parts) - 1:
        if parts[i] == "\n":
            parts[i + 1] = parts[i] + parts[i + 1]
            parts.pop(i)
        elif not parts[i].endswith(".") and len(parts[i]) < 20:
            assert parts[i + 1].startswith("\n"), "Next part should start with \\n"
            parts[i] += parts.pop(i + 1)
        else:
            i += 1
    _assert_placeholders(parts)
    return _offsets(parts)


def _tail_indices(text, indices, placeholders):
    result = [0]
    for part in _parts(text, indices):
        splits = [
            m.start() + 60
            for m in _PLACEHOLDER.finditer(part)
            if placeholders[m.group()].endswith(_TAILS)
        ]

        base = result[-1]
        result.extend(base + p for p in splits + [len(part)])
    i = 0
    while i < len(result) - 1:
        part = _recover(text[result[i] : result[i + 1]], placeholders)
        if not part.endswith(_TAILS) and len(part) < 20:
            result.pop(i if i == len(result) - 2 else i + 1)
        else:
            i += 1
    _assert_placeholders(_parts(text, result))
    return result


def _newline_indices(text, indices, placeholders):
    result = [0]
    for part in _parts(text, indices):
        splits = []
        while match := _PLACEHOLDER.search(part):
            actual = placeholders[match.group()]
            splits.extend(match.start() + m.start() for m in re.finditer("\n", actual))
            part = part.replace(match.group(), actual)
        base = result[-1]
        result.extend(base + p for p in splits + [len(part)])
    text = _recover(text, placeholders)
    i = 0
    while i < len(result) - 1:
        part = text[result[i] : result[i + 1]]
        if len(part) < 3 or any(re.fullmatch(p, part.strip()) for p in _OPENINGS):
            result.pop(i if i == len(result) - 2 else i + 1)
        else:
            i += 1
    return result, text


def _balanced_split(part, *, language):
    text, placeholders = _disguise(part) if language else (part, {})
    patterns = _LANGUAGE_SPLITS if language else _MATH_SPLITS
    for pattern, width in patterns:
        positions = [m.start() + width for m in re.finditer(pattern, text)]
        if not positions:
            continue
        pos = min(positions, key=lambda p: abs(2 * p - len(text)))
        halves = [
            _recover(text[:pos], placeholders),
            _recover(text[pos:], placeholders),
        ]
        if min(map(len, halves)) >= (40 if language else 20):
            assert "".join(halves) == part
            return _offsets(halves)
    assert _recover(text, placeholders) == part
    return [0, len(part)]


def _break_long(part):
    indices = [0, len(part)]
    math_begin, math_end = -1, 0
    i = 0
    while i < len(indices) - 1:
        subpart = part[indices[i] : indices[i + 1]]
        if len(subpart) > 100:
            if not math_begin <= indices[i] < math_end:
                split = _balanced_split(subpart, language=True)
                if len(split) > 2:
                    indices[i + 1 : i + 1] = [indices[i] + p for p in split[1:-1]]
                    continue
            split = _balanced_split(subpart, language=False)
            if len(split) > 2:
                math_begin = split[1] if math_begin == -1 else min(math_begin, split[1])
                math_end = max(math_end, split[-1])
                indices[i + 1 : i + 1] = [indices[i] + p for p in split[1:-1]]
                continue
        i += 1
    return indices


def split_solution_inplace(reasoning_latex: str) -> list[int]:
    """Return upstream character indices verbatim, including degenerate results.

    Compatibility, not a total parser: "x", "Short.", and some repeated
    multiline placeholders raise AssertionError; empty text returns [0].
    The token adapters reject empty/degenerate partitions. AGENT INSTRUCTIONS:
    do not silently repair these quirks here; tests/core/test_segmentation.py
    pins upstream outcomes, including failures, with a corpus fingerprint.
    """
    disguised, placeholders = _disguise(reasoning_latex)
    indices = _tail_indices(disguised, _initial_indices(disguised), placeholders)
    indices, text = _newline_indices(disguised, indices, placeholders)
    result = [0]
    for part in _parts(text, indices):
        base = result[-1]
        result.extend(
            base + p
            for p in (_break_long(part)[1:] if len(part) > 100 else [len(part)])
        )
    assert text == reasoning_latex
    assert result[-1] == len(text)
    return result
