"""Inference contract test template; no production imports until tests are implemented."""

import pytest

pytestmark = pytest.mark.skip(
    reason="Inference test template; implement tests and remove this marker."
)


def test_completion_token_ids_contain_only_continuation():
    """TODO: Verify Completion.token_ids excludes every supplied prefix token."""
    raise NotImplementedError


def test_completion_logprobs_are_optional_and_token_aligned():
    """TODO: Check absent logprobs are None and provided scores align one per token."""
    raise NotImplementedError


def test_policy_forward_returns_next_token_logprobs():
    """TODO: Check [B, T-1] scores equal log p(input_ids[:, 1:]) on tiny logits."""
    raise NotImplementedError


def test_policy_forward_respects_padding_mask():
    """TODO: Check masked padding does not change valid next-token logprobs."""
    raise NotImplementedError


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_generate_preserves_batch_and_sample_nesting(backend):
    """TODO: Check [prefix][sample] ordering and counts with multiple prefixes/samples."""
    raise NotImplementedError


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_generate_uses_exact_branch_prefix_token_ids(backend):
    """TODO: Check prompt + trajectory[:boundary] reaches backend unchanged, not retokenized."""
    raise NotImplementedError


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_generate_reports_stop_termination(backend):
    """TODO: Force EOS before the token budget and verify finish_reason is 'stop'."""
    raise NotImplementedError


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_generate_reports_length_termination(backend):
    """TODO: Force budget exhaustion without EOS; check 'length' and max_new_tokens."""
    raise NotImplementedError


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_generate_returns_logprobs_only_when_requested(backend):
    """TODO: Check return_logprobs=False yields None; True yields one score per token."""
    raise NotImplementedError


def test_hf_sync_weights_keeps_shared_policy_current():
    """TODO: Check HF sharing Policy sees updates; sync_weights may be a no-op."""
    raise NotImplementedError


def test_vllm_sync_weights_copies_current_policy_weights():
    """TODO: Update Policy, sync_weights, and verify the backend uses the new weights."""
    raise NotImplementedError
