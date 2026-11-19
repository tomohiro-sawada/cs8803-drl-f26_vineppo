# README.

We assume everything in `scripts/setup` has been run correctly. 

## Directories.
- `core`: logic shared between training and evaluation. 
- `baselines`: Implementation of baselines.
- `benchmarking`: Infra for benchmarking trained policies on MATH and GSM8K. 

We only have two scripts at the root of `src` because there are only two types of experiments we run: training or evaluation.
This corresponds to `src/treetune/runtime` in [the original implementation](https://github.com/McGill-NLP/VinePPO/tree/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/runtime). 





## Policy-gradient training

`runtime.train.PPOTrainer` runs single-device PPO, VinePPO, GRPO or RLOO using caller-supplied models,
optimizer, fast tokenizer, ordered `Example` records, and a deterministic prompt
formatter. Dataset/model loading stays outside the trainer. For example, with
those objects already loaded:

```python
from pathlib import Path

import torch

from algorithms.vineppo import VinePPOConfig
from core.inference import HFInference
from runtime.config import PPOTrainerConfig
from runtime.train import PPOTrainer

config = PPOTrainerConfig(
    checkpoint_dir=Path("artifacts/training"),
    algorithm_config=VinePPOConfig(),
    num_iterations=100,
    prompts_per_iteration=4,
    num_samples=2,
    mc_num_samples=4,
    update_epochs=2,
    minibatch_size=4,
    microbatch_size=1,
)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)
trainer = PPOTrainer(
    config, policy, HFInference(policy, pad_token_id=tokenizer.pad_token_id),
    optimizer=optimizer, reference_policy=reference_policy,
)
inputs = {
    "examples": examples,
    "tokenizer": tokenizer,
    "task": "gsm8k",
    "formatter": format_prompt,
}
torch.manual_seed(0)  # Generation RNG; config.seed controls data/update shuffling.
history = trainer.train(**inputs)
```

The reference must be a separately loaded model with no shared actor storage;
the trainer freezes it. Optimizers must own exactly the trainable actor parameters
(filter parameters when using a partially frozen actor). No reference is copied
implicitly. Prompt formatting must match the selected model; tokenization adds
special tokens without truncation.

For VinePPO, each iteration samples prompts without replacement, scores primary and MC
responses with the same verifier/truncation policy, expands raw step advantages,
and caches old/reference scores before any update. Length truncations receive
`truncation_reward` (default `-1`), unlike evaluation. The existing actor updater
handles reshuffling, per-minibatch whitening, clipped PPO and reference KL.
Singleton minibatches require at least two response tokens in every episode;
unsupported plans fail before optimizer mutation. Exact segmentation errors also
propagate: there is no skipping, boundary rounding, or single-step fallback.

`microbatch_size` bounds score/backpropagation batches, **not generation**.
Generation groups primary prompts by token budget and samples MC continuations
per prefix; size `prompts_per_iteration`, `num_samples`, and `mc_num_samples` for
available KV-cache memory. The full collated rollout batch is stored on CPU.

`record_metrics`, `record_value_estimate`, and `record_prefix_samples` callbacks
on `train` expose iteration metrics, responses/step advantages, and MC samples.
Callbacks must not mutate training inputs, model state, or RNGs. Actor metrics
are means over optimizer steps, not globally token-weighted metrics.

Checkpoints are written to `checkpoint_dir/iteration-N.pt` at
`checkpoint_every` intervals and at completion. They contain actor/reference
weights, optimizer state, completed iteration count, training/input identities,
and torch RNG states. To resume, recreate the trainer and inputs, then call:

```python
trainer.load_checkpoint("artifacts/training/iteration-50.pt", **inputs)
history = trainer.train(**inputs)
```

`num_iterations` is the total target, not additional iterations. Only that target,
checkpoint interval, and output directory may change on resume. Model execution
configuration/dtypes, ordered data, tokenized prompts, tokenizer, optimizer
class/parameter order, and RNG device availability must match. Exact replay also
requires matching software/hardware and deterministic kernels. Python/NumPy RNGs
are not saved; formatters must be deterministic. Failed iterations require a
fresh trainer and loading the last completed checkpoint. Models/inference remain
caller-owned. Distributed training, schedulers, and AMP are not wired into this
trainer.

### LoRA and grouped baselines

For actor LoRA, wrap the actor **before** constructing inference and optimizer:

```python
from core.lora import apply_lora

policy = apply_lora(policy, rank=16, alpha=32)
optimizer = torch.optim.AdamW(
    [p for p in policy.parameters() if p.requires_grad], lr=1e-5,
)
```

The base can be BF16, but adapters remain FP32. Attention and MLP linear layers
are targeted, not the LM output head. Keep a separately loaded unadapted reference;
no shared storage or adapter-disable reference trick is used. Checkpoints retain
full model weights and validate adapter configuration/effective scaling on resume.
The PPO critic is not converted to LoRA by the experiment runner.

Select `algorithms.grpo.GRPOConfig()` or `algorithms.rloo.RLOOConfig()` with
`num_samples >= 2` for grouped baselines. They skip segmentation/MC/critic and fix
one advantage per completion before shuffling. GRPO population-standardizes each
prompt group and uses sequence-normalized clipped token ratios. RLOO uses an
unscaled leave-one-out baseline and a PPO-style clipped **sequence** importance
ratio, not vanilla single-step REINFORCE. Neither re-whitens advantages per
minibatch; single-token responses are supported. Value/prefix diagnostic callbacks
are rejected for these algorithms. See the experiment [algorithm documentation](../experiments/training/README.md#lora-and-grouped-algorithms)
and [cluster numerical checks](../experiments/training/NUMERICAL_CHECKS.md).

### Critic-based PPO baseline

Select `PPOConfig` and supply a separate critic and optimizer. `critic_backbone`
below is a separately loaded decoder-only base model (for example, loaded with
`AutoModel.from_pretrained` from the same SFT checkpoint as the actor), not an LM
head model and not the actor's shared backbone. It must use the same token IDs
and support the configured context length. `TokenCritic` adds a randomly
initialized scalar value head on the backbone's existing device/dtype. Place
both on the desired device before constructing the optimizer.

```python
from dataclasses import replace

from algorithms.ppo import PPOConfig, TokenCritic

config = replace(config, algorithm_config=PPOConfig(gamma=1.0, lam=0.95))
critic = TokenCritic(critic_backbone)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-6)
trainer = PPOTrainer(
    config, policy, HFInference(policy, pad_token_id=tokenizer.pad_token_id),
    optimizer=optimizer, reference_policy=reference_policy,
    critic=critic, critic_optimizer=critic_optimizer,
)
history = trainer.train(**inputs)
```

PPO skips reasoning-step segmentation and MC continuations. It caches scalar
values of the states **before** each response token, then computes token GAE
and fixed lambda-return targets (`returns = raw_advantages + old_values`).
Task reward is placed on the final response token; both EOS and length-truncated
responses are terminal with zero bootstrap. Length truncations receive the same
`truncation_reward` as VinePPO. This is not time-limit bootstrapping for robotics.

Reference KL remains an explicit actor-loss penalty, exactly as in VinePPO;
there is no KL reward shaping or double penalty. The actor updater whitens raw
GAE advantages per logical minibatch. The critic minimizes half the mean squared
error over response tokens against **unwhitened, fixed** returns, with no value
clipping. Because optimizers are separate, no combined-loss value coefficient
is used; control the critic learning rate through its optimizer.

All scores, values and targets are cached before either optimizer steps. Each
iteration runs the actor's update epochs, then the critic's update epochs using
the same `update_epochs`, minibatch/microbatch sizes and maximum gradient norm.
Memory-only microbatch splitting preserves the logical minibatch loss. Actor,
critic and reference must not share parameter or buffer storage; optimizer
ownership is validated before collection. Scoring disables dropout for both
trainable models without disabling autograd during updates.

`record_value_estimate` receives `PPOValueEstimate` for PPO (cached values and
return targets) or `VinePPOValueEstimate` for VinePPO. PPO never calls
`record_prefix_samples`. Critic metrics are prefixed `critic/`; `optimizer_steps`
counts actor steps and `critic/optimizer_steps` counts critic steps separately.
PPO checkpoints additionally save the critic, its optimizer, and their execution
identities. Recreate both when loading; changing GAE settings or the algorithm
on resume is rejected.
