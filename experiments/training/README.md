# Single-GPU cluster training and benchmarks

Submit these launchers **from the repository root**. They call
`experiments/training/run.py`, the shared offline runner for both training and
synchronized throughput measurements. Segmentation validation remains a separate
cluster task; errors propagate rather than being skipped.
Numerical acceptance is a separate gate; successful logging does not establish
GPU numerical correctness or learning quality.

## Cluster profiles

- **Skynet:** partition/account `kargo-lab`, QoS `debug`, one A40 GPU,
  one node/rank, 6 CPUs, 32G host RAM, 4 hours. These settings come from
  `scripts/paper_artifact_replications/evaluate_sft.slurm`.
  `tmp/skynet_launch_template.slurm` is mislabeled and contains PACE settings;
  it is deliberately not used as the Skynet resource profile.
- **PACE ICE:** partition `ice-gpu`, account `math`, QoS `coc-ice`, one GPU,
  one node/rank, 6 CPUs, 96G host RAM, 2 hours. Based on
  `tmp/pace-ice_launch_template.slurm`, with increased host memory and allowed
  constraint `gpu-h100|gpu-h200`. Host RAM is not GPU memory.
  Pin the GPU constraint at submission and pass matching `--expected-gpu H100`
  or `--expected-gpu H200`; the helper requires this explicit expectation.

VinePPO is the default and primary benchmark on both clusters. PACE also supports
`--algorithm ppo`. **PPO on Skynet fails explicitly**, rather than changing the
algorithm. GRPO and RLOO are supported on both clusters, as are full-parameter
and actor-LoRA tuning. Memory fit depends on the local model and workload: one GPU
does not guarantee a given model will fit, especially PPO with its full-parameter
critic. See [algorithm and LoRA details](#lora-and-grouped-algorithms) and the
[cluster numerical checklist](NUMERICAL_CHECKS.md) before launching long runs.

## Prerequisites and paths

Provision the environment and local artifacts separately, before submission.
The launchers never install packages, download model/data artifacts, SSH, source
`bash.sh` or shell secrets, or change `CUDA_VISIBLE_DEVICES`. W&B online logging
is enabled by default in the Slurm wrappers; see [tracking](#wandb-tracking). SLURM controls GPU visibility.
They use the preinstalled `.venv/bin/python`; optionally export `PYTHON` with an
explicit executable path (not a command with arguments). The runner and its
required dependencies must already be present.

Set these variables on the cluster's login node, from the repository root.
Replace the example artifact paths with existing paths visible on compute nodes.
The examples are shell commands to adapt, not paths embedded in the launchers.

```bash
export GSM8K_MODEL=/path/to/local/gsm8k-model
export MATH_MODEL=/path/to/local/math-model
export GSM8K_DATA=/path/to/local/gsm8k-dataset
export MATH_DATA=/path/to/local/math-dataset
# Set this separately on each cluster, outside all individual run outputs.
export TRAINING_CACHE=/path/to/cluster-storage/shared-training-cache
# Optional: export PYTHON=/path/to/preinstalled/environment/bin/python

# SLURM opens stdout/stderr BEFORE running the script: create these first.
mkdir -p artifacts/training/skynet/slurm artifacts/training/pace_ice/slurm
```

`--task`, `--model-path`, `--dataset-path`, and `--output` are required explicitly.
No remote model/dataset IDs are resolved. Dataset content/format and checkpoint
compatibility are the runner's responsibility. Relative paths resolve against
`SLURM_SUBMIT_DIR`. Each output must be new; existing outputs (including resume
sources) are never reused. Use a unique run name under the appropriate cluster
subdirectory. Do not submit the same output path concurrently.

The shared helper directs Hugging Face, XDG, Torch, and Triton caches into
`TRAINING_CACHE` and enables Hugging Face offline mode. Keep this cache outside
run outputs so model/cache copies do not multiply per experiment. Offline mode
will fail on missing cached/local assets rather than fetching them. Do not pass
`sbatch --export=NONE`: the configured paths must reach the job.

## Skynet: both tasks, training and benchmark

Use a new `RUN_TAG` for each batch of submissions; choose descriptive names when
comparing configurations. The shell expands these variables before `sbatch`.

```bash
RUN_TAG=trial01

sbatch experiments/training/skynet/train.slurm \
  --task gsm8k --model-path "$GSM8K_MODEL" --dataset-path "$GSM8K_DATA" \
  --output "artifacts/training/skynet/gsm8k-train-$RUN_TAG"

sbatch experiments/training/skynet/train.slurm \
  --task math --model-path "$MATH_MODEL" --dataset-path "$MATH_DATA" \
  --output "artifacts/training/skynet/math-train-$RUN_TAG"

sbatch experiments/training/skynet/benchmark.slurm \
  --task gsm8k --model-path "$GSM8K_MODEL" --dataset-path "$GSM8K_DATA" \
  --output "artifacts/training/skynet/gsm8k-benchmark-$RUN_TAG"

sbatch experiments/training/skynet/benchmark.slurm \
  --task math --model-path "$MATH_MODEL" --dataset-path "$MATH_DATA" \
  --output "artifacts/training/skynet/math-benchmark-$RUN_TAG"
```

Skynet defaults to `--expected-gpu A40`; any different expectation is rejected.

## PACE ICE: both tasks, training and benchmark

These examples pin H100. For **H200**, change both `--constraint=gpu-h100` to
`--constraint=gpu-h200` and `--expected-gpu H100` to `--expected-gpu H200`.
Scheduler options go **before** the script; runner options go **after** it.
Although the profile permits either GPU, do not leave the union constraint when
expecting one specific GPU model.

```bash
RUN_TAG=trial01

sbatch --constraint=gpu-h100 experiments/training/pace_ice/train.slurm \
  --expected-gpu H100 --task gsm8k \
  --model-path "$GSM8K_MODEL" --dataset-path "$GSM8K_DATA" \
  --output "artifacts/training/pace_ice/gsm8k-train-$RUN_TAG"

sbatch --constraint=gpu-h100 experiments/training/pace_ice/train.slurm \
  --expected-gpu H100 --task math \
  --model-path "$MATH_MODEL" --dataset-path "$MATH_DATA" \
  --output "artifacts/training/pace_ice/math-train-$RUN_TAG"

sbatch --constraint=gpu-h100 experiments/training/pace_ice/benchmark.slurm \
  --expected-gpu H100 --task gsm8k \
  --model-path "$GSM8K_MODEL" --dataset-path "$GSM8K_DATA" \
  --output "artifacts/training/pace_ice/gsm8k-benchmark-$RUN_TAG"

sbatch --constraint=gpu-h100 experiments/training/pace_ice/benchmark.slurm \
  --expected-gpu H100 --task math \
  --model-path "$MATH_MODEL" --dataset-path "$MATH_DATA" \
  --output "artifacts/training/pace_ice/math-benchmark-$RUN_TAG"
```

To run PPO on PACE, add `--algorithm ppo` and use a distinct output run name.
For example, an H200 PPO benchmark:

```bash
sbatch --constraint=gpu-h200 experiments/training/pace_ice/benchmark.slurm \
  --expected-gpu H200 --algorithm ppo --task gsm8k \
  --model-path "$GSM8K_MODEL" --dataset-path "$GSM8K_DATA" \
  --output artifacts/training/pace_ice/gsm8k-ppo-h200-benchmark-trial01
```

## Defaults and overrides

`train.slurm` fixes `--mode train` and explicitly sets **100 iterations**, rather
than inheriting the runner's three-iteration default. It uses zero warmup
iterations and checkpoints every 10 iterations. `benchmark.slurm` fixes
`--mode benchmark`: **3 total iterations, of which the first is warmup**, leaving
2 measured iterations. Warmup still performs real updates. The runner defaults to
final-only checkpointing in benchmark mode; omit `--checkpoint-every` to keep that
behavior (zero is invalid). Both modes keep the latest **2 checkpoints**.

Both start with a small workload: 2 prompts per iteration, 2 samples per prompt,
2 Monte Carlo samples, 1 update epoch, minibatch size 2, microbatch size 1,
`float32`, **2048 maximum sequence tokens and 1024 maximum new tokens**.
Small batch dimensions do not shorten the generation budget. Benchmark length
limits remain the original limits unless explicitly overridden.

Use `--option value` syntax. Explicit runner arguments override helper defaults;
`--mode` cannot be overridden because the wrapper selects it. Supported tuning
arguments are:

```text
--num-iterations --warmup-iterations --prompts-per-iteration
--num-samples --mc-num-samples --update-epochs --minibatch-size
--microbatch-size --max-sequence-length --max-new-tokens
--learning-rate --critic-learning-rate --critic-lora-rank --math-grader-workers --seed
--kl-coef --lr-schedule --model-protocol
--checkpoint-every --checkpoint-format --keep-checkpoints --resume --dtype
--resume-checkpoint-every --archive-checkpoints
--finetuning --lora-rank --lora-alpha
--wandb-mode --wandb-project --wandb-entity --wandb-group --wandb-name
--evaluation --validation-rounds --eval-batch-size --eval-seed --eval-timeout-seconds
```

`--dtype float32` is required for full tuning. LoRA also accepts
`--dtype bfloat16` for the frozen base and reference. PPO loads a separate
FP32 critic by default. With `--critic-lora-rank`, the critic freezes its
backbone at the requested dtype and trains all-linear LoRA adapters with
alpha twice the rank, plus a scalar value head. Its checkpoints store the
trainable critic, optimizer and frozen-backbone fingerprint. Resume needs the
same initial critic weights. Trainable parameters, gradients, and AdamW
moments use FP32. Learning rates default to `1e-6` (full actor),
`1e-5` (LoRA actor), and `1e-5` (critic); seed is `42`. Explicit learning rates win. Numeric ranges and shape consistency are
validated by the runner; the launchers do not duplicate its validation logic.
For example, append `--num-iterations 200 --checkpoint-every 20` to a training
command. Resource/time overrides belong to `sbatch`, not the runner; requesting
more time does not guarantee the cluster QoS allows it.

MATH and GSM8K VinePPO assign a token spanning a reasoning-step cut to the earlier
step. It advances the cut to the next complete sampled-token boundary and
merges coincident cuts. Both MC branching and PPO credit use those boundaries
on the original token IDs. Invalid token alignment still aborts the run.
Prompt controls marked special in the added-token registry must be explicitly
allowed by the prompt protocol.

The vendored MATH splitter restores nested math placeholders in reverse
insertion order, a departure from upstream, which rejects a math span captured
inside a later-matched span. A response the splitter still rejects becomes one
step in training: its advantage is its reward minus the prompt's value.
Upstream drops such responses. `metadata.json` records the policy as
`math_segmentation_policy`.

`--math-grader-workers` defaults to 1. With more workers, training grades
distinct answer tuples concurrently within a primary or Monte Carlo sample
group and keeps the results in sample order. Reserve one CPU and up to 1 GiB
per worker in addition to learner resources.

## Resume into a new output

PACE unattended jobs use `pace_ice/train_resumable.slurm RUN_ROOT [runner options]`.
Submit from a detached worktree with absolute paths for the Python environments,
data, cache, and `PACE_CHAIN_LOG_DIR`. Pass an absolute run root and explicit
`--num-iterations` and `--resume-checkpoint-every`. Each allocation stores its
progress in `status.json`, holds one afterany successor, and creates a new
`attempt-NNNN` only when it needs to train. The runner observes the wall-clock
deadline and the Slurm warning file at iteration boundaries. With
`--archive-checkpoints yes`, immutable candidates and the shared frozen blob
are copied to checkpoint history and SHA256-verified before candidate pruning.
The chain stops after a no-progress link and records `HALTED`.

Long RhoMath runs may select `--lr-schedule polynomial` for 3% warmup and
degree-one decay to zero over optimizer steps. A resumed schedule requires the
same total budget and base learning rates. Pass `--kl-coef 0.003` explicitly for
the paper-coefficient GRPO/RLOO arm. PPO and VinePPO use `0.0001` by default.

RestEM keeps up to ten verified correct stopped responses per sampled prompt,
resets the adapter and optimizer before each update round, and optimizes
response-token NLL. Use `--algorithm restem --finetuning lora`. The matched
RhoMath run uses 10 rounds of 320 prompts, eight samples per prompt, and eight
update epochs. Sampling uses temperature 1 and includes the full distribution.
The last update of each round is its validation candidate and next sampler.

`--model-protocol olmo3-sft` selects the pinned OLMo-3-7B-Instruct-SFT
chat prompt. MATH requests a boxed answer and GSM8K requests a `####` line.
Evaluation stops at chat end or model EOS and uses a 2,048-token context with a
1,024-token generation cap. Base weights are pinned to
`e1452fc572d51966ff4aaeb25118b891eb93e549`.

`--checkpoint-format compact-lora` stores the frozen actor and reference once
in `checkpoints/frozen-lora.pt`. Each iteration checkpoint stores the adapter
tensors, optimizer and RNG state, and the frozen file's absolute path and SHA256.
Keep that file while any retained or selected checkpoint references it, including
checkpoints written by later resume attempts. The default format remains `full`.

Use an existing compatible checkpoint path, not its old run output as the new
output. Keep task, algorithm, model, and other compatibility-sensitive options
consistent with the checkpoint, including dtype and optimizer learning rates.
`--num-iterations` is the total target, not additional work. Checkpoints live under
`<output>/checkpoints/iteration-N.pt`. Resume always writes into a new output;
source checkpoints are never pruned.

```bash
RESUME_CHECKPOINT=/path/to/existing/compatible/checkpoint
sbatch --constraint=gpu-h100 experiments/training/pace_ice/train.slurm \
  --expected-gpu H100 --algorithm vineppo --task gsm8k \
  --model-path "$GSM8K_MODEL" --dataset-path "$GSM8K_DATA" \
  --resume "$RESUME_CHECKPOINT" --num-iterations 200 \
  --output artifacts/training/pace_ice/gsm8k-resumed-trial02
```

For Skynet, use `skynet/train.slurm`, omit the PACE constraint/expectation (or set
`--expected-gpu A40`), and choose `artifacts/training/skynet/<new-runname>`.

## Storage and interpreting benchmarks

Plan against approximately **10 TB on Skynet** and **300 GB on PACE**, but these
are storage quotas, **not available free disk**. Check current quota usage and
filesystem capacity before a run. Shared models, datasets, caches, logs, old
runs, and resume-source checkpoints all count toward storage consumption.

Retention applies within a run, not across previous outputs. Keeping two
checkpoints requires transient space for **three full checkpoints**: two retained
files plus the incoming `.tmp` file, which is renamed before pruning.
Optimizer/critic state can greatly exceed
model weights. A resumed run initially coexists with its source checkpoints.
Benchmark final-only checkpointing reduces write frequency, not the final
checkpoint size. No per-token rollout dumps are enabled by these launchers;
there is no rollout-dump flag passed to the runner. Remove old runs deliberately,
not via automatic launcher deletion.

The runner writes `metadata.json` (configuration, data fingerprint, GPU/software,
git revision), `metrics.jsonl` (every completed iteration), and `summary.json`
(warmup-excluded aggregate measurements). It loads only the local DatasetDict's
`train` split: GSM8K final answers are extracted after `####`, and MATH solutions
are passed with `reference_is_solution=True`. The paper prompt is explicit. Local
model weights are not content-hashed; preserve their pinned provenance separately.

Phase wall times synchronize CUDA before and after primary generation, MC or GAE
estimation, old/reference score caching, actor/critic updates, and checkpoint I/O.
MC generation timing is nested inside MC estimation: do not add both. Other time
includes primary verification, segmentation and collation. Iteration totals
include scheduled checkpoint writes/pruning, but exclude initial loading/prompt
tokenization and metrics JSONL writes. Sampling counts include primary **and MC**
continuations/tokens, not prompt/padding tokens; tokens/second divides their total
by full iteration time, not decoding time alone. GPU allocated/reserved peaks
reset each iteration and include initialized optimizer state after the first
update. They are PyTorch allocator statistics, not total `nvidia-smi` usage.

Benchmarking really changes model weights. Three iterations are a short pilot,
not a stable estimate across problem lengths: repeat with more iterations and
representative sample counts before estimating a full run's duration. Final
checkpoint I/O is included, so inspect its phase time separately when comparing
different checkpoint schedules. Held-out checkpoint evaluation is opt-in; see [evaluation and selection](EVALUATION.md).
There is no model export; checkpoints are resumable trainer state.
Use identical model/task, sample counts, token limits, dtype, and update shapes
for meaningful comparisons, and record the GPU type. No throughput, completion
time, or actual cluster execution is claimed here.

## LoRA and grouped algorithms

Install the updated locked dependencies before submitting jobs (`uv sync --locked`
for the default environment). PEFT is a project dependency; jobs do not install it.

`--finetuning lora` wraps only the actor with standard PEFT LoRA: rank 16,
alpha 32, dropout 0, no bias training, and `all-linear` targets (attention **and
MLP** projections; the LM output head is excluded). Only adapter parameters enter
the actor optimizer. The frozen reference is a separate unadapted copy of the
starting SFT model. This is not QLoRA, adapter merging, or a shared-base reference
optimization. `--lora-rank` and `--lora-alpha` require LoRA mode; full tuning
remains the default.

[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) motivates including
MLP projections and tuning the learning rate separately from full tuning. The
10x LR default here is a starting point inspired by that study, not an established
RhoMath optimum. Sweep LR and rank while holding rollout/update budgets fixed.
LoRA saves trainable gradient/optimizer memory, not MC generation or KV-cache work.

Resume checkpoints are still **full state dictionaries**, including frozen base
and reference weights. Adapter-only export is not implemented. LoRA saves
optimizer-state space but does not reduce the whole checkpoint to adapter size.
The trainer checks adapter configuration, active layers and effective scaling,
along with dtypes and optimizer identity before loading; changing rank/alpha or
switching full/LoRA on resume is rejected. This identity extension is not a
migration layer for older incompatible checkpoints.

### GRPO

`--algorithm grpo --num-samples G` requires `G >= 2` completions for each prompt.
Rewards are centered and divided by the **population** standard deviation plus
`1e-6`, independently per prompt. Constant groups have exactly zero advantages.
The clipped token-ratio surrogate and unclamped sampled k3 penalty are averaged
over response tokens within each completion, then over completions equally.
Advantages are fixed before shuffling, with no subsequent minibatch whitening.
These conventions follow `tmp/hw5_student/src/grpo.py`'s contracts; the homework
implementation itself has not been changed. They differ from the global
response-token mean used by the existing PPO/VinePPO updater.

### RLOO

`--algorithm rloo --num-samples G` also requires `G >= 2` and uses
`A_i = r_i - mean(r_j for j != i)` with no standard-deviation normalization.
This implementation is explicitly **PPO-style clipped sequence-ratio RLOO**:
`ratio_i = exp(sum_response(new_logp - old_logp))`, a clipped sequence-level
surrogate, and separate token-summed k3 averaged over sequences. It is not the
vanilla single-step on-policy REINFORCE loss, and KL is not folded into rewards.
Ratio/clip/approx-KL metrics refer to sequences for RLOO, tokens for GRPO.

Both methods use the same verifier, fixed old/reference scores, truncation
penalty, and response/EOS masks as the other algorithms, but have **no critic,
segmentation, or MC continuations**. `mc_num_samples` is unused. Benchmark MC
counts/times should be zero. KL is an old-rollout sampled regularizer, not an
unbiased current-policy KL estimate after updates. Unlike PPO/VinePPO, grouped
losses do not clamp k3 to 10; do not assume equal coefficients imply equal loss
scales across algorithms.

For example, on Skynet after setting paths as above:

```bash
sbatch experiments/training/skynet/benchmark.slurm \
  --algorithm grpo --finetuning lora --dtype bfloat16 \
  --lora-rank 16 --lora-alpha 32 --learning-rate 1e-5 \
  --num-samples 4 --task gsm8k \
  --model-path "$GSM8K_MODEL" --dataset-path "$GSM8K_DATA" \
  --output artifacts/training/skynet/gsm8k-grpo-lora-benchmark-trial01
```

Change `grpo` to `rloo` or `vineppo` and choose a new output directory for those
experiments. PACE uses its own launcher and explicit GPU constraint/expectation
as above. Run the numerical checks before treating this as a learning result.

## Local syntax check (no submission)

```bash
bash -n experiments/training/launch.sh
bash -n experiments/training/skynet/train.slurm
bash -n experiments/training/skynet/benchmark.slurm
bash -n experiments/training/pace_ice/train.slurm
bash -n experiments/training/pace_ice/benchmark.slurm
```

## W&B tracking

Slurm train/benchmark wrappers default to `--wandb-mode online` and project
`vineppo`. Direct `run.py` calls default to `disabled` to preserve offline/local
usage. Explicit arguments override wrapper defaults:

```sh
# Append to a normal training/benchmark submission:
--wandb-mode online --wandb-project vineppo --wandb-group rho-gsm8k-lora
# Disconnected compute node: write a real local W&B run for later sync:
--wandb-mode offline
# Deliberately disable tracking:
--wandb-mode disabled
```

`--wandb-entity` selects a team/user; otherwise W&B uses the authenticated default.
`--wandb-name` supplies a display name. Online mode uses an exported
`WANDB_API_KEY`, or reads **only that key** from the repository-root `.env` via
`python-dotenv` without variable interpolation or shell execution. An empty
exported key is an error, not a request to replace it. `.env` stays git-ignored;
never pass keys as CLI arguments. Offline/disabled modes do not read `.env`.
`uv sync --locked` installs both tracking dependencies before submission.

Online mode sends scalar metrics, allowlisted run configuration/provenance, and
SDK system telemetry to W&B; network access is required. It does not upload
model/checkpoint weights, dataset rows, sampled text, `.env`, console output,
source code, or Git patches. SDK metadata/requirements capture is disabled.
Online authentication/initialization failures propagate; there is no silent
switch to offline mode. Mid-run SDK unrecoverable errors stop the run. Local
`metrics.jsonl` remains the primary per-iteration record and is written first.
W&B logging and final upload time are excluded from iteration compute timings.

The output contains `wandb.json` (run ID, URL, mode) and the SDK's local `wandb/`
records. Each invocation, **including trainer checkpoint resume**, creates a
fresh W&B run. `iteration` is the absolute completed trainer iteration;
`budget/*` counters include warmup but restart at zero for this invocation.
They must not be interpreted as whole-training budgets after a resume. Use a
shared group and the recorded resume path to relate runs. Failures after tracker
initialization finish with nonzero status and log the exception **type**, not
its potentially sensitive message. Startup failures before tracking initializes
remain in the Slurm/local logs. Run summary mirrors `summary.json` and excludes
benchmark warmup. No automatic checkpoint artifact uploads occur.

### Diagnostics and paper interpretation

All existing finite scalar metrics are logged unchanged, including policy loss,
`grad_norm` (before clipping), `clip_frac`, `ratio_mean`, `approx_kl`,
`reference_kl`, `kl_loss`, PPO `critic/*`, actor/critic learning rates, phase
seconds, primary/MC tokens and continuations, memory peaks and checkpoint bytes.
Cumulative `budget/primary_tokens`, `budget/mc_tokens`, `budget/sampled_tokens`,
`budget/primary_continuations`, `budget/mc_continuations`,
`budget/mc_expansions`, and `budget/iteration_seconds` support cost comparisons.
MC generation timing is nested inside MC estimation, not additive.
VinePPO samples each distinct token prefix once per iteration and shares the
continuations among every episode with that prefix (`mc_prefix_policy` in
`metadata.json`). `mc_expansions`, `mc_continuations` and `mc_tokens` count what
was sampled; `vineppo/step_count` still counts one value per episode and step.
Runs without `mc_prefix_policy` sampled every repeated prefix separately.

Additional cheap diagnostics use existing rollouts only:
- `response_length_mean/max` include generated special tokens;
  `truncation_fraction` distinguishes capped outputs.
- `reward_mean` includes truncation penalties. `stopped_correctness_mean` is
  correctness **conditional on stopped responses**, omitted if none stopped;
  neither is held-out test accuracy.
- `raw_advantage_token_std/zero_fraction` are token-weighted, before whitening.
- `group_zero_reward_variance_fraction` identifies GRPO/RLOO groups without
  policy signal.
- `vineppo/step_count`, `vineppo/steps_per_response_mean/std`, and
  `vineppo/mc_prefix_value_mean/std` describe step structure and prefix-weighted
  values. Prefix-value dispersion is **not** Monte Carlo standard error.
- `vineppo/splitter_rejected_responses` counts the iteration's responses that
  the MATH splitter rejected and that trained as one step.

Actor/critic iteration diagnostics average optimizer-step metrics. `approx_kl`
is relative to the old policy; RLOO uses sequence log-ratios, GRPO/PPO/VinePPO
use token ratios. `reference_kl` is the sampled k3 regularizer: PPO/VinePPO clamp
per-token k3 to 10 and sum per sequence, GRPO averages per completion token,
RLOO sums per sequence. These scales are not directly interchangeable and are
not an exact current-policy/reference KL after updates.

The original paper (https://arxiv.org/html/2410.01679) motivates accuracy versus
primary episodes, total sampling cost, wall time and reference-policy drift;
its value-network analysis motivates MC-value diagnostics. **Implemented separately:** [held-out evaluation and validation-best checkpoint
selection](EVALUATION.md), including parent/child W&B metrics. MC return standard
errors and independent high-K critic probes are not implemented.
Figure 4 requires validation-selected test Pass@1, averaged over 16 evaluation
sampling rounds (not pass@16). The paper's 256-continuation value probes are
separate expensive evaluations; critic training loss is not a substitute.
Do not infer replication readiness or learning success from these logging checks.

With validation selection enabled, retention protects the best in addition to
the latest N; allow N+2 transient full checkpoints. See [EVALUATION.md](EVALUATION.md)
for GPU coexistence, evaluation cost, provenance, and final-test gating.
