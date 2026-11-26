"""The value replay records the restored critic and shared MC continuations."""

import json
from statistics import fmean

import pytest
import torch

from core.inference import Completion
from scripts.diagnostics import value_estimate_analysis as analysis
from scripts.diagnostics import value_estimate_replay as replay_module
from tests.runtime import test_ppo_train

ppo_case = test_ppo_train.ppo_case


def test_replay_records_critic_values_and_shared_continuations(
    ppo_case, monkeypatch, tmp_path
):
    make, inputs = ppo_case
    run = make()
    tokenizer = inputs["tokenizer"]
    monkeypatch.setattr(replay_module, "PROMPTS", 2)
    monkeypatch.setattr(replay_module, "CONTINUATIONS", 4)
    calls = []

    def generate(prefixes, *, num_samples, max_new_tokens, temperature):
        calls.append((len(prefixes), num_samples))
        primary = num_samples == run.config.num_samples
        text = "a\nb\n#### 1" if primary else "\n#### "
        return [
            [
                Completion(
                    tokenizer.encode(text + ("" if primary else str(1 + sample % 2)))
                    + [tokenizer.eos_token_id],
                    "stop",
                )
                for sample in range(num_samples)
            ]
            for _ in prefixes
        ]

    monkeypatch.setattr(run.inference, "generate", generate)
    before = test_ppo_train.states(run)
    replay_module.replay(run, tmp_path, **inputs, reference_is_solution=False)
    test_ppo_train.assert_exact(test_ppo_train.states(run), before)
    rows = [json.loads(line) for line in (tmp_path / "value-replay.jsonl").open()]
    # Two prompts x two identical two-step responses; each prefix is sampled once.
    assert calls[0] == (2, 2) and {n for _, n in calls[1:]} == {4}
    assert sum(count for count, _ in calls[1:]) == 4
    assert [(row["step"], row["steps"]) for row in rows] == [(0, 2), (1, 2)] * 4
    assert len({row["prefix_sha256"] for row in rows}) == 4
    for row in rows:
        assert len(row["continuation_rewards"]) == len(row["continuation_tokens"]) == 4
        assert row["prefix_tokens"] + row["remaining_tokens"] == len(
            tokenizer.encode("a\nb\n#### 1")
        ) + 1
    example = {e.id: e for e in inputs["examples"]}
    for row in rows:
        name = row["episode"].rsplit(":", 1)[0]
        prompt = tokenizer(inputs["formatter"](example[name].question))["input_ids"]
        prefix = prompt + tokenizer.encode("a\nb\n#### 1")[: row["prefix_tokens"]]
        ids = torch.tensor([prefix])
        with torch.no_grad():
            value = run.critic(ids, torch.ones_like(ids), torch.arange(len(prefix))[None])
        assert row["critic_value"] == pytest.approx(value[0, -1].item(), abs=1e-6)
        # Answers alternate 1, 2: half the continuations match each reference.
        assert fmean(row["continuation_rewards"]) == 0.5


def test_analysis_scores_estimators_on_distinct_states_without_shared_samples():
    def row(prefix, step, value, rewards):
        return {
            "episode": f"e{prefix}", "prefix_sha256": prefix, "step": step,
            "critic_value": value, "continuation_rewards": rewards,
            "continuation_tokens": [3] * len(rewards),
        }

    rows = [
        row("p", 0, 0.5, [1.0] * 10 + [0.0] * 10),
        row("p", 0, 0.5, [1.0] * 10 + [0.0] * 10),  # The same state, seen twice.
        row("q", 1, 0.5, [1.0] * 20),
        row("r", 9, 0.0, [0.0] * 20),
    ]
    result = analysis.summarize(rows)
    assert result["states"] == 3 and result["continuations_per_state"] == 20
    assert result["responses"] == 3  # Episode "ep" appears twice.
    assert result["continuation_tokens"] == 180
    critic = result["critic"]
    assert critic["mean_absolute_error"] == pytest.approx(0.5 / 3)
    assert critic["bias"] == pytest.approx(-0.5 / 3)
    # Errors 0, -0.5, 0 against references 0.5, 1, 0.
    assert critic["explained_variance"] == pytest.approx(1 - (1 / 18) / (1 / 6))
    assert list(critic["by_depth"]) == [0, 1, analysis.DEEPEST]
    assert critic["by_depth"][1]["mean_absolute_error"] == 0.5
    one = result["mc_1"]
    assert one["pairs"] == 60 and critic["pairs"] == 3
    # State p: a 1 is scored against 9/19 and a 0 against 10/19.
    assert one["by_depth"][0]["mean_absolute_error"] == pytest.approx(10 / 19)
    assert one["by_depth"][1]["mean_absolute_error"] == 0
    assert result["mc_9"]["pairs"] == 6
