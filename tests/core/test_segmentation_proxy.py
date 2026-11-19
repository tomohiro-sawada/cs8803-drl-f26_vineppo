"""Offline regression of fixed proxy text/spans, not sampled-token roundtripping."""

import json
import re
from itertools import pairwise
from pathlib import Path

import pytest

from core.segmentation import gsm8k_step_boundaries, math_step_boundaries

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/segmentation/openrouter_proxy.json"
)
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_recorded_proxy_boundaries(case):
    assert case["token_id_source"] == "locally_reencoded_proxy_text_NOT_sampled"
    segment = {"math": math_step_boundaries, "gsm8k": gsm8k_step_boundaries}[
        case["task"]
    ]
    ids, spans = case["reencoded_token_ids"], case["character_spans"]
    original = ids.copy()
    expected = case["segmentation"]
    if expected["success"]:
        boundaries = segment(case["text"], ids, spans)
        assert list(boundaries) == expected["boundaries"]
        assert boundaries[0] == 0 and boundaries[-1] == len(ids)
        assert all(a < b for a, b in pairwise(boundaries))
        assert [token for a, b in pairwise(boundaries) for token in ids[a:b]] == ids
    else:
        assert expected["category"] == "inside_token"
        # Independently establish why the recorded cut cannot be an exact token cut.
        cut = int(re.search(r"character (\d+)", expected["exception_message"])[1])
        assert any(start < cut < end for start, end in spans)
        with pytest.raises(
            AssertionError, match=re.escape(expected["exception_message"])
        ):
            segment(case["text"], ids, spans)
    assert ids == original
