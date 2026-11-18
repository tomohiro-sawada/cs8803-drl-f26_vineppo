"""
Main training logic for policy gradient methods.

TODO: not sure how good of an idea it is to have a shared training logic across all of the baselines? Would it be harder to customize each one?

TODO: For the first complete training run, I would additionally log representative decoded trajectories, step boundaries, MC returns, and advantages. This will make mistakes visible that scalar training metrics cannot diagnose.

TODO: for now, DPO+ and RestEM are out of scope;
we would need to implement a separate trainer for them.
"""

# typing
from dataclasses import dataclass
from pathlib import Path
from jaxtyping import Tensor


from core.data import Episode, TrainBatch
from transformers import PreTrainedModel

from core.inference import Completion, HFInference
from core.verifier import RewardVerifier
from core.algorithm.base import AlgorithmConfig


@dataclass
class PPOTrainerConfig:
    # training parameters
    num_iterations: int = 100

    # saving
    checkpoint_dir: Path

    # algorithm specific configurations ...?
    algorithm_config: AlgorithmConfig


class PPOTrainer:
    def __init__(
        self,
        config: PPOTrainerConfig,
        policy: PreTrainedModel,
        inference: HFInference,
        verifier: RewardVerifier,
    ):
        """
        Trainer method for both PPO and VinePPO.
        TODO: we should add support for GRPO and RLOO

        TODO: what is the best way to specify advantage computation?
        We can specify the training objective (e.g. PPO, VinePPO) in the configuration, and find the advantage computation function. (If so, should we also define a registry for the advantage computation functions?)
        """
        self.config = config
        assert inference.model is policy, "Inference must share the training actor"
        self.policy = policy
        self.inference = inference
        self.verifier = verifier

        # get advantage method
        # TODO: refactor this later when we have more algorithms
        if self.config.algorithm_config.name == "PPO":
            from algorithms.ppo import ppo_advantage_method

            self.advantage_method = ppo_advantage_method
        elif self.config.algorithm_config.name == "VinePPO":
            from algorithms.vineppo import vineppo_advantage_method

            self.advantage_method = vineppo_advantage_method
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm_config.name}")

    def train(
        self,
    ):
        for iteration in range(self.config.num_iterations):
            # generate episodes and estimate advantages
            # For VinePPO: this generates the partial context episodes
            episodes = self.generate_episodes(...)

            # compute token-level advantages
            # TODO: again, refactor once we have more algorithms
            advantage: Tensor["float32", ...] = self.advantage_method(episodes, ...)

            # update policy and save checkpoint
            batch = make_train_batch(episodes, advantage)
            self.update_policy(batch, ...)
            self.save_checkpoint(...)

    def save_checkpoint(self):
        pass

    def update_policy(self, batch):
        """
        Update the policy using the given batch of episodes.
        This is shared between PPO and VinePPO.
        """
        # compute the loss
        loss = self.policy.compute_loss(batch)

        # update the policy
        self.policy.update(loss)

    def generate_episodes(
        self,
        # ...
    ):
        """
        Generate episodes using the current policy.
        Backend is the inference engine?
        """
        pass
