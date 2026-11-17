# Vendored from McGill-NLP/VinePPO, commit 9f716e68bb934e71da86c226c1699c36b5553faf.
# MIT; see LICENSE.VinePPO and NOTICE.md for attribution and modifications.
# ruff: noqa
from typing import Optional, Dict, Any


class GSM8K:
    use_original_format = True

    def extract_predicted_answer_from_text(
        self, text: str, problem: Optional[str] = None
    ) -> Optional[str]:
        if self.use_original_format:
            # Extract the final answer based on ####
            if "####" not in text:
                return None
            parts = text.split("####")
            assert len(parts) >= 2
            return parts[-1].strip()

        raise AssertionError(
            "Only the official original-format configuration is vendored"
        )

    def grade_answer(
        self,
        *,
        given_answer: Optional[str] = None,
        ground_truth: str = None,
        item: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        if given_answer is None:
            return False

        assert ground_truth is not None
        return (
            given_answer.strip().replace(",", "").lower()
            == ground_truth.strip().lower()
        )
