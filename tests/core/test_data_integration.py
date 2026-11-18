"""Collated response targets agree with independently scored, unpadded episodes."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from core.data import EpisodeWithAdvantages, collate
from core.inference import token_logprobs


@torch.no_grad()
def test_collated_response_logprobs_match_unpadded_episodes():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=23,
                n_positions=24,
                n_embd=16,
                n_layer=1,
                n_head=2,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                resid_pdrop=0.5,
                embd_pdrop=0.5,
                attn_pdrop=0.5,
                attn_implementation="sdpa",
            )
        )
    # Zero is both padding and a real token, including a response target.
    episodes = [
        EpisodeWithAdvantages(
            prompt_ids=prompt,
            response_ids=response,
            reward=1.0,
            finish_reason="length",
            advantages=[advantage] * len(response),
            raw_step_advantages=[advantage],
            step_boundaries=(0, len(response)),
            mc_prefix_values=(0.5,),
        )
        for prompt, response, advantage in [
            ([4], [0, 6], 0.5),
            ([7, 0, 8], [9, 10, 11], -0.5),
            ([12, 13], [14], 1.0),
        ]
    ]
    expected = []
    for episode in episodes:
        ids = torch.tensor([episode.prompt_ids + episode.response_ids])
        scores = token_logprobs(model, ids, torch.ones_like(ids, dtype=torch.bool))
        expected.append(scores[0, len(episode.prompt_ids) - 1 :])

    old_logprobs = [scores.tolist() for scores in expected]
    ref_logprobs = [(scores - 0.25).tolist() for scores in expected]
    batch = collate(
        episodes,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        pad_token_id=0,
    )
    for tensor, shape, dtype in [
        (batch.input_ids, (3, 6), torch.int64),
        (batch.attention_mask, (3, 6), torch.bool),
        (batch.response_mask, (3, 5), torch.bool),
        (batch.old_logprobs, (3, 5), torch.float32),
        (batch.ref_logprobs, (3, 5), torch.float32),
        (batch.advantages, (3, 5), torch.float32),
    ]:
        assert tensor.device.type == "cpu"
        assert tensor.shape == shape
        assert tensor.dtype == dtype

    actual = token_logprobs(model, batch.input_ids, batch.attention_mask)
    for row, episode in enumerate(episodes):
        tokens = episode.prompt_ids + episode.response_ids
        length = len(tokens)
        torch.testing.assert_close(
            batch.input_ids[row], torch.tensor(tokens + [0] * (6 - length))
        )
        torch.testing.assert_close(batch.attention_mask[row], torch.arange(6) < length)
        targets = torch.arange(1, 6)
        torch.testing.assert_close(
            batch.response_mask[row],
            (targets >= len(episode.prompt_ids)) & (targets < length),
        )
        mask = batch.response_mask[row]
        torch.testing.assert_close(
            actual[row, mask], expected[row], rtol=1e-5, atol=1e-6
        )
        for tensor, values in [
            (batch.old_logprobs, old_logprobs[row]),
            (batch.ref_logprobs, ref_logprobs[row]),
            (batch.advantages, episode.advantages),
        ]:
            torch.testing.assert_close(tensor[row, mask], torch.tensor(values))
            assert torch.count_nonzero(tensor[row, ~mask]) == 0
