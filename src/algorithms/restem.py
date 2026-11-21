"""Rejection-sampled maximum likelihood on the original verified trajectories."""

from dataclasses import dataclass

import torch

from algorithms.base import AlgorithmConfig


@dataclass
class RestEMConfig(AlgorithmConfig):
    max_correct_per_prompt: int = 10


def keep_correct(episodes, samples_per_prompt, max_correct):
    assert samples_per_prompt > 0 and max_correct > 0
    assert len(episodes) % samples_per_prompt == 0
    kept = []
    for start in range(0, len(episodes), samples_per_prompt):
        group = episodes[start : start + samples_per_prompt]
        kept.extend(
            [e for e in group if e.finish_reason == "stop" and e.reward == 1][
                :max_correct
            ]
        )
    assert kept, "RestEM found no verified correct stopped responses"
    return kept


def restem_loss(logprobs, mask, tokens):
    assert logprobs.shape == mask.shape and mask.dtype == torch.bool
    valid = logprobs[mask]
    assert 0 < valid.numel() <= tokens and torch.isfinite(valid).all()
    loss = -valid.sum() / tokens
    return loss, {"loss": loss.detach()}
