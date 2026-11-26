"""Score value estimates against the replay's many-sample Monte Carlo values.

    python scripts/diagnostics/value_estimate_analysis.py LABEL=REPLAY.jsonl ...

Reads value-replay.jsonl files from value_estimate_replay.py and prints one
JSON line per file. A state is a distinct token prefix. Its reference value is
the mean reward of all its continuations. Reported per estimator, over all
states and by reasoning depth (the index of the step the state precedes):

- mean absolute error and bias against the reference;
- explained variance, 1 - Var(reference - estimate) / Var(reference).

"critic" is the restored PPO critic. "mc_K" is the mean of K continuations,
scored against the mean of the state's other continuations so the two share no
sample; it averages over the disjoint K-subsets in sampling order.
"reference_standard_error" is the mean standard error of the reference itself.
"""

import json
import sys
from statistics import fmean, pvariance

KS = (1, 3, 9)
DEEPEST = 6  # States at this depth or deeper are pooled.


def states(rows):
    """One row per distinct prefix."""
    return list({row["prefix_sha256"]: row for row in rows}.values())


def score(pairs):
    """pairs: (estimate, reference) per state."""
    errors = [estimate - reference for estimate, reference in pairs]
    spread = pvariance([reference for _, reference in pairs])
    return {
        "pairs": len(pairs),
        "mean_absolute_error": fmean(map(abs, errors)),
        "bias": fmean(errors),
        "explained_variance": 1 - pvariance(errors) / spread if spread else None,
    }


def estimator_pairs(row, name):
    rewards = row["continuation_rewards"]
    if name == "critic":
        return [(row["critic_value"], fmean(rewards))]
    k = int(name.removeprefix("mc_"))
    total = sum(rewards)
    return [
        (sum(chunk) / k, (total - sum(chunk)) / (len(rewards) - k))
        for chunk in (rewards[i : i + k] for i in range(0, len(rewards) - k + 1, k))
    ]


def summarize(rows):
    responses = len({row["episode"] for row in rows})
    rows = states(rows)
    samples = {len(row["continuation_rewards"]) for row in rows}
    assert len(samples) == 1 and min(samples) > max(KS), samples
    depths = sorted({min(row["step"], DEEPEST) for row in rows})
    result = {
        "states": len(rows),
        "responses": responses,
        "continuations_per_state": samples.pop(),
        "reference_mean": fmean(fmean(row["continuation_rewards"]) for row in rows),
        "reference_standard_error": fmean(
            (pvariance(r := row["continuation_rewards"]) / len(r)) ** 0.5 for row in rows
        ),
        "continuation_tokens": sum(sum(row["continuation_tokens"]) for row in rows),
    }
    for name in ("critic", *(f"mc_{k}" for k in KS)):
        pairs = {
            depth: [
                pair
                for row in rows
                if min(row["step"], DEEPEST) == depth
                for pair in estimator_pairs(row, name)
            ]
            for depth in depths
        }
        result[name] = score([pair for group in pairs.values() for pair in group]) | {
            "by_depth": {depth: score(group) for depth, group in pairs.items()}
        }
    return result


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        label, path = argument.split("=", 1)
        with open(path) as stream:
            print(json.dumps({"label": label} | summarize(list(map(json.loads, stream)))))
