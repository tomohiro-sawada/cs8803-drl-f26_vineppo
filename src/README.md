# README.

We assume everything in `scripts/setup` has been run correctly. 

## Directories.
- `core`: logic shared between training and evaluation. 
- `baselines`: Implementation of baselines.
- `benchmarking`: Infra for benchmarking trained policies on MATH and GSM8K. 

We only have two scripts at the root of `src` because there are only two types of experiments we run: training or evaluation.
This corresponds to `src/treetune/runtime` in [the original implementation](https://github.com/McGill-NLP/VinePPO/tree/9f716e68bb934e71da86c226c1699c36b5553faf/src/treetune/runtime). 





## VinePPO training

`runtime.train.PPOTrainer` runs single-device VinePPO using caller-supplied models,
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

Each iteration samples prompts without replacement, scores primary and MC
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
caller-owned. Critic PPO, distributed training, schedulers, and AMP are not wired
into this trainer.
