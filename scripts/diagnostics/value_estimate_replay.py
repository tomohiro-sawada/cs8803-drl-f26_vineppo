"""Value-estimate replay: a PPO critic against many-sample Monte Carlo values.

Takes the training runner's CLI through launch.sh
(TRAINING_ENTRY=scripts/diagnostics/value_estimate_replay.py): the PPO run's
own options plus --resume CHECKPOINT, --num-iterations equal to that
checkpoint's iteration, and a new --output. The runner restores the actor and
the critic of that iteration and checks them as it does on resume. No weights
change and no iteration runs.

The restored actor then samples REPLAY_PROMPTS x --num-samples responses
(default 64 prompts, chosen by --seed). Each response is scored with the
training reward and segmented, and every distinct token prefix gets
REPLAY_CONTINUATIONS (default 256) continuations from the same actor, through
the VinePPO estimator. <output>/value-replay.jsonl holds one line per response
and step: the critic's value of that state and each continuation's reward and
length. Identical prefixes share their continuations and their prefix_sha256.
scripts/diagnostics/value_estimate_analysis.py reads these files.
"""

import json
import os
import sys
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from core.data import Episode
from core.segmentation import step_boundaries_from_tokens
from experiments.training import run as runner

PROMPTS = int(os.environ.get("REPLAY_PROMPTS", "64"))
CONTINUATIONS = int(os.environ.get("REPLAY_CONTINUATIONS", "256"))


def replay(
    trainer, output, *, examples, tokenizer, task, formatter, reference_is_solution
):
    """Write value-replay.jsonl for the trainer's restored actor and critic."""
    config = trainer.config
    prompts, _ = trainer._prepare_inputs(
        examples, tokenizer, task, formatter, reference_is_solution
    )
    trainer.inference.sync(trainer.iteration)
    chosen = torch.randperm(
        len(examples), generator=torch.Generator().manual_seed(config.seed)
    )[:PROMPTS].tolist()
    eos = trainer.policy.generation_config.eos_token_id
    items = []
    groups = trainer.collect_completions(
        [prompts[i] for i in chosen], num_samples=config.num_samples
    )
    for index, group in zip(chosen, groups, strict=True):
        score = partial(
            trainer._score_responses,
            example=examples[index],
            tokenizer=tokenizer,
            task=task,
            reference_is_solution=reference_is_solution,
        )
        for sample, (completion, reward) in enumerate(
            zip(group, score(group), strict=True)
        ):
            episode = Episode(
                prompts[index], completion.token_ids, reward, completion.finish_reason
            )
            boundaries = step_boundaries_from_tokens(
                completion.token_ids,
                tokenizer,
                dataset=task,
                eos_token_ids=[eos] if type(eos) is int else eos,
                context_ids=episode.prompt_ids,
                context_special_ids=trainer.prompt_special_ids,
            )
            items.append((f"{examples[index].id}:{sample}", episode, boundaries, score))
    _, estimates = trainer._estimate_ppo(
        [episode for _, episode, _, _ in items], [name for name, _, _, _ in items]
    )
    values = {e.episode_id: e.critic_value_estimates for e in estimates}
    episodes = {name: (episode, boundaries) for name, episode, boundaries, _ in items}
    trainer.config = replace(config, mc_num_samples=CONTINUATIONS)
    with (output / "value-replay.jsonl").open("x") as stream:

        def record(samples):
            episode, boundaries = episodes[samples.episode_id]
            start = samples.response_prefix_length
            prefix = episode.prompt_ids + episode.response_ids[:start]
            row = {
                "episode": samples.episode_id,
                "prefix_sha256": sha256(json.dumps(prefix).encode()).hexdigest(),
                "step": boundaries.index(start),
                "steps": len(boundaries) - 1,
                "prefix_tokens": start,
                "remaining_tokens": len(episode.response_ids) - start,
                "episode_reward": episode.reward,
                "critic_value": values[samples.episode_id][start],
                "continuation_rewards": [c.reward for c in samples.continuations],
                "continuation_tokens": [
                    len(c.response_ids) for c in samples.continuations
                ],
            }
            stream.write(json.dumps(row) + "\n")

        trainer.estimate_vineppo_batch(items, record_prefix_samples=record)


def main():
    args = runner.parse_args()
    assert args.algorithm == "ppo" and args.resume is not None
    assert args.evaluation == "disabled" and args.archive_checkpoints == "no"
    load = runner.MeasuredTrainer.load_checkpoint

    def load_then_replay(trainer, path, **inputs):
        load(trainer, path, **inputs)
        assert trainer.iteration == args.num_iterations, "Replay must not train"
        replay(trainer, args.output, **inputs)

    runner.MeasuredTrainer.load_checkpoint = load_then_replay
    runner.main()


if __name__ == "__main__":
    main()
