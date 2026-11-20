I’ll check the remaining training entry points, PPO implementation, model setup, and GPU assumptions so I can distinguish a working smoke test from a full training run.

**VinePPO is close to a first real training smoke test; PPO still needs substantial implementation. Neither is yet validated for an unattended full run on an A40.**

The last commit completed the **training library**, not the model/data launch script or GPU validation. The 817 passing tests establish useful correctness, but not RhoMath-scale memory fit, throughput, or learning.

## VinePPO: what remains

The main pieces exist: rollout collection, verification, MC values, advantages, actor optimization, reference KL, and checkpoint resume.

### 1. Build the actual training runner

`experiments/training/` is empty. We need a runner that:
- Loads the task-specific RhoMath SFT checkpoint and a separate frozen reference.
- Loads the **training split**, normalizes examples, and supplies the paper prompt.
- Creates the optimizer and training config.
- Records metrics and periodically evaluates held-out examples.
- Provides a Slurm launch command and resume option.

There is reusable setup in [`evaluate_sft.py`](cs8803-drl-f26_vineppo/scripts/paper_artifact_replications/evaluate_sft.py), including pinned model revisions, prompt formatting, CUDA checks, and model loading. **Its dataset loader is explicitly test-only**, so we cannot reuse it unchanged for training.

Also, the downloader is currently a stub despite its README describing a working command. This local `.cache/` is empty; I haven’t checked artifacts on the GPU host.

### 2. Validate segmentation on real sampled RhoMath outputs

**This is the biggest correctness/operability uncertainty.**

The current segmenter deliberately rejects:
- Text that does not round-trip to the exact sampled token IDs.
- Reasoning-step boundaries falling inside a token.
- Empty responses or unsupported special tokens.
- Certain short or degenerate MATH outputs.

Those failures abort training. The tests verify that behavior; they do not establish how frequently RhoMath produces those cases.

Before a long run, sample a small corpus for both tasks and measure these failures. If they occur, we need an explicit segmentation-policy decision—not a silent fallback.

### 3. Establish a numerical and memory configuration on A40

The existing evaluation loader uses BF16. **That is not automatically a validated full-finetuning recipe.** We need to choose parameter/optimizer precision deliberately and test actual updates.

Other current constraints:
- No integrated AMP or gradient-checkpointing path.
- Scoring materializes full vocabulary logits and float32 log-softmax.
- `microbatch_size` limits scoring/backpropagation, **not generation**.
- Generation can batch all selected primary prompts and all MC samples for a prefix.

Start with microbatch size one and modest rollout counts; measure peak memory **after optimizer state has been initialized**, not just after loading models.

## PPO: more than wiring

[`src/algorithms/ppo.py`](cs8803-drl-f26_vineppo/src/algorithms/ppo.py) is still a stub, with an invalid `core.algorithm.base` import. Its test file is empty.

We still need:
- A critic backbone/value head and optimizer.
- Cached pre-update values.
- GAE and fixed return targets.
- Explicit terminal/truncation handling.
- Critic regression loss and update scheduling.
- Critic state in checkpoints.
- Integration and regression tests.

The critic data structures and collation exist, and actor-update machinery is reusable. But the trainer currently **explicitly rejects non-VinePPO configurations**. The name `PPOTrainer` should not be interpreted as implemented critic PPO.

## Is one A40 enough?

**A 48-GB A40 is a plausible target for 1.1B VinePPO, but we have not demonstrated fit. PPO is tighter.**

For scale, these are *rough persistent-state estimates*, not measured peaks:

- A 1.1B model in FP32: approximately **4.4 GB** of weights.
- Actor weights + gradients + two FP32 Adam moments: approximately **17.6 GB**.
- A frozen BF16 reference adds approximately **2.2 GB**.
- A separate similarly sized FP32 critic with gradients/Adam adds another approximately **17.6 GB**.

That puts a conservative actor/reference/critic arrangement around **37.4 GB before activations, logits, temporary optimizer allocations, and allocator overhead**. VinePPO avoids that critic cost. Precision choices can change these numbers substantially.

For VinePPO, **runtime may be a bigger limitation than weights**. With \(N\) primary episodes, \(K\) reasoning steps per episode, and \(M\) MC samples per prefix, each iteration generates roughly:

\[
N + NKM
\]

continuations, with different lengths. The current implementation processes MC prefixes sequentially. We should benchmark before predicting whether a full run takes hours or days.

**FlashAttention/vLLM are not prerequisites for the first smoke test.** Start with the existing HF/SDPA path. Installing vLLM alone does not accelerate the current training loop.

## Recommended next milestones

1. **GSM8K VinePPO runner:** correct train split, pinned artifacts, task prompt, explicit optimizer/precision.
2. **Real-output preflight:** tokenizer/segmentation compatibility and truncation statistics.
3. **A40 smoke run:** several complete iterations, actual weight changes, finite gradients, peak memory, checkpoint/restart.
4. **Short pilot:** enough iterations to measure throughput and evaluate before/after.
5. **MATH validation:** repeat segmentation and verifier checks before scaling.
6. **Implement critic PPO** against that working experimental setup.

**My assessment:** VinePPO needs one focused integration pass plus real-GPU debugging—not another algorithm rewrite. PPO needs an additional critic/GAE implementation phase. A paper-comparable experiment also needs a pinned protocol: the current trainer deliberately uses temperature 1 without nucleus filtering, so “runs end to end” is not yet “reproduces the paper.”

I would make the **GSM8K VinePPO runner and A40 smoke test** the next task, before adding acceleration or critic PPO.
