"""Offline PPO/VinePPO/GRPO/RLOO training; the learner owns one CUDA device.

--backend hf samples from the live actor on the learner GPU. --backend vllm keeps
scoring, the frozen reference and updates on the learner and samples from
--num-workers persistent vLLM replicas (experiments/training/vllm_worker.py), which
are republished the LoRA actor before every iteration. --worker-placement
dedicated gives every replica its own GPU; colocated runs the single replica on
the learner's GPU, where the two alternate rather than compete.

Both modes perform real updates; --num-iterations (default 3) is the TOTAL target,
including resumed iterations. Set it explicitly for longer training. Benchmark
excludes the first --warmup-iterations of THIS invocation only from its summary.
Trainable parameters, gradients and AdamW moments use FP32. Critic-free LoRA
can load the frozen actor base and reference in BF16 through --dtype.
Core scoring and losses use FP32. No quantization, gradient checkpointing or DDP.

Checkpoint retention applies only to this fresh output directory. Atomic writes
need space for keep-checkpoints + 1 full checkpoints before pruning (three with
the default retention). A checkpoint includes actor, reference, optimizer and,
for PPO, critic/optimizer. Resume requires matching core execution/data identity.

Timing uses synchronized CUDA wall clocks, not CUDA events. MC estimation includes
MC generation and continuation verification, but not primary verification or
segmentation. Those, collation, sampling and instrumentation are in other_seconds.
MC generation is a nested diagnostic: do not add it to MC estimation again.
Iteration totals include scheduled checkpoint serialization/pruning but exclude
JSONL writing and initial model/data loading/tokenization. Token counts include
sampled EOS, exclude padding/prompt tokens, and count all MC continuations.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial, wraps
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, time

# Must precede all HF imports, including transitive runtime imports.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from algorithms.grpo import GRPOConfig
from algorithms.ppo import PPOConfig, TokenCritic
from algorithms.restem import RestEMConfig
from algorithms.rloo import RLOOConfig
from algorithms.vineppo import MC_PREFIX_POLICY, VinePPOConfig
from core.data import Example
from core.inference import HFInference
from core.lora import apply_lora, lora_identity
from core.prompt_protocol import OLMO3, PROMPT, RHOMATH, prompt_protocol
from core.segmentation import MATH_SEGMENTATION_POLICY
from core.vllm_inference import VLLMClient, VLLMInference, gpu_identity, slurm_attempt
from experiments.training.archive import archive_checkpoint
from experiments.training.resume import (
    BoundaryDeadline,
    ResumeSlot,
    atomic_record,
    checkpoint_position,
)
from experiments.training.selection import CheckpointSelection
from experiments.training.tracking import WandBTracker
from runtime.config import PPOTrainerConfig
from runtime.schedule import POLYNOMIAL
from runtime.train import PPOTrainer

PHASES = (
    "primary_generation",
    "mc_estimation",
    "ppo_estimation",
    "score_cache",
    "actor_refresh",
    "actor_update",
    "critic_update",
    "checkpoint",
    "validation",
)


def paper_prompt(question: str) -> str:
    return PROMPT.format(question=question)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=("train", "benchmark", "evaluate"), required=True
    )
    parser.add_argument(
        "--algorithm",
        choices=("vineppo", "ppo", "grpo", "rloo", "restem"),
        required=True,
    )
    parser.add_argument("--task", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--model-protocol", choices=(RHOMATH, OLMO3), default=RHOMATH)
    for name in ("model-path", "dataset-path", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--expected-gpu", help="Case-insensitive substring of CUDA device name"
    )
    for name, default in (
        ("num-iterations", 3),
        ("prompts-per-iteration", 2),
        ("num-samples", 1),
        ("mc-num-samples", 4),
        ("update-epochs", 1),
        ("minibatch-size", 2),
        ("microbatch-size", 1),
        ("max-sequence-length", 2048),
        ("max-new-tokens", 1024),
        ("keep-checkpoints", 2),
    ):
        parser.add_argument(f"--{name}", type=positive_int, default=default)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Benchmark-only summary exclusion; still real updates (default: 1)",
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        help="Default: 1e-6 full tuning, 1e-5 LoRA; tune independently",
    )
    parser.add_argument("--critic-learning-rate", type=positive_float, default=1e-5)
    parser.add_argument("--kl-coef", type=float)
    parser.add_argument(
        "--lr-schedule", choices=("constant", POLYNOMIAL), default="constant"
    )
    parser.add_argument("--math-grader-workers", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-every",
        type=positive_int,
        help="Default: every train iteration; benchmark final only. Final always saved.",
    )
    parser.add_argument(
        "--checkpoint-format", choices=("full", "compact-lora"), default="full"
    )
    parser.add_argument(
        "--resume", type=Path, help="Core iteration-N.pt; output must still be new"
    )
    parser.add_argument("--resume-checkpoint-every", type=positive_int)
    parser.add_argument("--archive-checkpoints", choices=("yes", "no"), default="no")
    parser.add_argument("--resume-audit", choices=("yes", "no"), default="no")
    parser.add_argument(
        "--resume-selection",
        type=Path,
        help="Source selection.json; requires --resume and automatic evaluation",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--finetuning", choices=("full", "lora"), default="full")
    parser.add_argument("--lora-rank", type=positive_int)
    parser.add_argument("--lora-alpha", type=positive_float)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="vineppo")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--evaluation",
        choices=("disabled", "validation", "validation-test"),
        default="disabled",
    )
    parser.add_argument("--validation-rounds", type=positive_int, default=16)
    parser.add_argument("--eval-batch-size", type=positive_int, default=4)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-timeout-seconds", type=positive_int, default=7200)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--num-workers", type=positive_int)
    parser.add_argument(
        "--worker-dir", type=Path, help="Set by launch.sh: worker identity files"
    )
    parser.add_argument(
        "--worker-placement", choices=("dedicated", "colocated"), default="dedicated"
    )
    args = parser.parse_args(argv)
    if (args.backend == "vllm") != (args.num_workers is not None) or (
        args.backend == "vllm"
    ) != (args.worker_dir is not None):
        parser.error(
            "--backend vllm requires --num-workers and --worker-dir, hf neither"
        )
    if args.worker_placement == "colocated" and args.num_workers != 1:
        parser.error("--worker-placement colocated requires --num-workers 1")
    if args.backend == "vllm" and args.finetuning != "lora":
        parser.error("--backend vllm requires a LoRA actor")
    if not 0 <= args.eval_seed <= 2**32 - max(16, args.validation_rounds):
        parser.error("Evaluation seed range exceeds NumPy bounds")
    if args.mode == "evaluate":
        if (
            args.resume_selection is None
            or args.resume is not None
            or args.evaluation == "disabled"
        ):
            parser.error(
                "Evaluate mode requires a frozen --resume-selection and automatic evaluation"
            )
    elif args.evaluation != "disabled":
        if args.mode != "train":
            parser.error("Automatic checkpoint selection requires train mode")
        if (args.resume is None) != (args.resume_selection is None):
            parser.error(
                "Automatic selection resume requires --resume and --resume-selection together"
            )
    elif args.resume_selection is not None:
        parser.error("--resume-selection requires automatic evaluation")
    if args.finetuning == "full" and (
        args.lora_rank is not None or args.lora_alpha is not None
    ):
        parser.error("LoRA options require --finetuning lora")
    if args.checkpoint_format == "compact-lora" and args.finetuning != "lora":
        parser.error("--checkpoint-format compact-lora requires --finetuning lora")
    if args.math_grader_workers > 1 and args.task != "math":
        parser.error("--math-grader-workers applies to MATH")
    if (
        args.mode != "evaluate"
        and args.dtype == "bfloat16"
        and args.finetuning == "full"
    ):
        parser.error("Full tuning requires --dtype float32")
    if args.finetuning == "lora":
        args.lora_rank = 16 if args.lora_rank is None else args.lora_rank
        args.lora_alpha = 32.0 if args.lora_alpha is None else args.lora_alpha
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.finetuning == "lora" else 1e-6
    if args.kl_coef is None:
        args.kl_coef = 0 if args.algorithm == "restem" else 1e-4
    if args.algorithm == "restem" and (
        args.finetuning != "lora" or args.kl_coef != 0 or args.lr_schedule != "constant"
    ):
        parser.error("RestEM requires LoRA, zero KL, and a constant learning rate")
    if args.algorithm in ("grpo", "rloo") and args.num_samples < 2:
        parser.error("GRPO/RLOO require --num-samples >= 2 per prompt")
    if args.warmup_iterations < 0 or not 0 <= args.seed < 2**64:
        parser.error("warmup must be nonnegative; seed must be in [0, 2**64)")
    if args.kl_coef is not None and (
        not math.isfinite(args.kl_coef) or args.kl_coef < 0
    ):
        parser.error("--kl-coef must be finite and nonnegative")
    if args.mode == "benchmark" and args.warmup_iterations >= args.num_iterations:
        parser.error("benchmark needs at least one non-warmup iteration")
    for name in ("model_path", "dataset_path"):
        path = getattr(args, name).expanduser().resolve(strict=True)
        assert path.is_dir(), f"Expected local directory: {path}"
        setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    if args.mode == "evaluate":
        assert args.output.is_dir(), "Evaluate mode needs an existing run root"
    else:
        assert not args.output.exists(), (
            "Output must be a new directory, including on resume"
        )
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve(strict=True)
        assert args.resume.is_file(), "Resume must be a checkpoint file"
    if args.resume_selection is not None:
        args.resume_selection = args.resume_selection.expanduser().resolve(strict=True)
        assert args.resume_selection.is_file()
    return args


def load_examples(path: Path, task: str):
    """Use only train rows; IDs preserve deterministic source split/index order."""
    dataset = load_from_disk(str(path))
    assert isinstance(dataset, DatasetDict), "Expected a locally saved DatasetDict"
    train = dataset["train"]
    examples = []
    for index, row in enumerate(train):
        question = row["question" if task == "gsm8k" else "problem"]
        answer = row["answer" if task == "gsm8k" else "solution"]
        assert isinstance(question, str) and question.strip(), (
            f"Empty question: {index}"
        )
        assert isinstance(answer, str) and answer.strip(), f"Empty reference: {index}"
        if task == "gsm8k":
            parts = answer.split("####")
            assert len(parts) == 2, f"Malformed GSM8K reference: {index}"
            answer = parts[1].strip()
        assert answer.strip(), f"Empty final reference: {index}"
        examples.append(Example(f"{task}:train:{index}", question.strip(), answer))
    assert examples, "Empty train split"
    digest = hashlib.sha256()
    for example in examples:
        digest.update(
            (
                json.dumps(asdict(example), sort_keys=True, ensure_ascii=False) + "\n"
            ).encode()
        )
    return examples, {
        "split": "train",
        "rows": len(train),
        "fingerprint": train._fingerprint,
        "normalized_sha256": digest.hexdigest(),
    }


def timed(phase):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self.meter.phase(phase):
                return method(self, *args, **kwargs)

        return wrapped

    return decorate


class Meter:
    """Scalar-only instrumentation; never samples, reseeds, or retains rollouts."""

    def __init__(self):
        self.active = None
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)

    def start(self):
        self.seconds.clear()
        self.counts.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = perf_counter()

    @contextmanager
    def phase(self, name):
        previous = self.active
        self.active = name
        torch.cuda.synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.seconds[name] += perf_counter() - started
            self.active = previous


class Measured:
    """Count and time whichever sampler follows in the MRO."""

    def __init__(self, *args, meter, **kwargs):
        super().__init__(*args, **kwargs)
        self.meter = meter

    def sync(self, version):
        with self.meter.phase("actor_refresh"):
            super().sync(version)

    def generate(self, prefixes, **kwargs):
        assert self.meter.active in ("primary_generation", "mc_estimation")
        kind = "mc" if self.meter.active == "mc_estimation" else "primary"
        with self.meter.phase(f"{kind}_generate_call"):
            result = super().generate(prefixes, **kwargs)
        self.meter.counts[f"{kind}_continuations"] += sum(map(len, result))
        self.meter.counts[f"{kind}_tokens"] += sum(
            len(c.token_ids) for group in result for c in group
        )
        if kind == "mc":
            self.meter.counts["mc_expansions"] += len(prefixes)
        return result


class MeasuredInference(Measured, HFInference):
    pass


class MeasuredVLLMInference(Measured, VLLMInference):
    pass


class MeasuredTrainer(PPOTrainer):
    """Thin wrappers around core hooks; checkpoint-aware metric finalization."""

    def __init__(
        self,
        *args,
        meter,
        stream,
        warmup_iterations,
        keep_checkpoints,
        resume_checkpoint_every=None,
        deadline=None,
        archive=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.meter = meter
        self.stream = stream
        self.warmup_iterations = warmup_iterations
        self.keep_checkpoints = keep_checkpoints
        self.rows = []
        self.pending = None
        self.saved = []
        self.tracker = None
        self.selection = None
        self.resume_every = resume_checkpoint_every
        self.resume_slot = ResumeSlot(self.config.checkpoint_dir)
        self.deadline = deadline
        self.archive = archive
        self.archived_frozen = False
        self.split_critic = (
            self.critic is not None
            and self.compact_lora_checkpoints
            and (archive or deadline is not None)
        )
        self.chain_status = (
            Path(os.environ["PACE_CHAIN_STATUS_FILE"]) if deadline is not None else None
        )

    def _prepare_inputs(self, *args, **kwargs):
        result = super()._prepare_inputs(*args, **kwargs)
        assert all(len(ids) < self.config.max_sequence_length for ids in result[0]), (
            "A train prompt exhausts context; increase --max-sequence-length"
        )
        self.meter.start()
        return result

    @timed("primary_generation")
    def collect_completions(self, *args, **kwargs):
        return super().collect_completions(*args, **kwargs)

    @timed("mc_estimation")
    def estimate_vineppo_batch(self, *args, **kwargs):
        return super().estimate_vineppo_batch(*args, **kwargs)

    @timed("ppo_estimation")
    def _estimate_ppo(self, *args, **kwargs):
        return super()._estimate_ppo(*args, **kwargs)

    @timed("score_cache")
    def _cache_scores(self, *args, **kwargs):
        return super()._cache_scores(*args, **kwargs)

    @timed("actor_update")
    def update_policy(self, *args, **kwargs):
        return super().update_policy(*args, **kwargs)

    @timed("critic_update")
    def update_value(self, *args, **kwargs):
        return super().update_value(*args, **kwargs)

    def record_metrics(self, metrics):
        assert self.pending is None
        self.pending = metrics
        if (
            self.iteration % self.config.checkpoint_every
            and self.iteration != self.config.num_iterations
        ):
            if self.resume_every and self.iteration % self.resume_every == 0:
                self.save_resume()
            self.finish_iteration()

    def save_resume(self):
        # The last candidate is already a complete resume boundary.
        if self.iteration > 0 and self.iteration % self.config.checkpoint_every:
            with self.meter.phase("checkpoint"):
                path = self.resume_slot.save(self.iteration, self.save_boundary)
                self.meter.counts["checkpoint_bytes"] = path.stat().st_size

    def stop_at_boundary(self):
        if self.deadline is None or not self.deadline.should_stop():
            return False
        if self.rows:
            self.save_resume()
        print(
            f"Stopping at completed iteration {self.iteration} for the allocation deadline"
        )
        return True

    def save_checkpoint(self, path=None):
        assert path is None, "Runner checkpoints must stay in this run's directory"
        with self.meter.phase("checkpoint"):
            saved = self.save_boundary()
            self.meter.counts["checkpoint_bytes"] = saved.stat().st_size
            self.saved.append(saved)
        assert self.pending is not None
        if self.selection is not None:
            with self.meter.phase("validation"):
                torch.cuda.empty_cache()
                self.pending.update(self.selection.validate(saved, self.iteration))
        self.archive_candidate(saved)
        with self.meter.phase("checkpoint"):
            self.prune_candidates()
        self.finish_iteration()
        return saved

    def save_boundary(self, path=None):
        if not self.split_critic:
            return (
                super().save_checkpoint()
                if path is None
                else super().save_checkpoint(path)
            )
        self.prune_critic_blobs()
        saved = super().save_checkpoint(path, split_critic=True)
        self.prune_critic_blobs()
        return saved

    def prune_critic_blobs(self):
        directory = self.config.checkpoint_dir
        directories = (
            directory.parent.parent.glob("attempt-*/checkpoints")
            if self.chain_status is not None
            else (directory,)
        )
        for folder in directories:
            # The run owns this directory. A killed write may leave staging files.
            for staging in folder.glob(".critic-*.pt.*.tmp"):
                staging.unlink()
            for blob in folder.glob("critic-*.pt"):
                if blob == self.critic_checkpoint:
                    continue
                iteration = int(blob.stem.removeprefix("critic-"))
                if self.archive and (folder / f"iteration-{iteration}.pt").is_file():
                    assert blob.with_suffix(".archive.json").is_file(), blob
                blob.unlink()

    def prune_candidates(self):
        own = self.saved
        attempts = [own[-1].parent]
        if self.archive and self.chain_status is not None:
            root = self.config.checkpoint_dir.parent.parent
            attempts = sorted(root.glob("attempt-*/checkpoints"))
            own = sorted(
                (
                    path
                    for directory in attempts
                    for path in directory.glob("iteration-*.pt")
                ),
                key=lambda path: (*checkpoint_position(path), path.parent.parent.name),
            )
        retained = set(own[-self.keep_checkpoints :])
        if self.selection is not None:
            retained.add(self.selection.best_path)
        for old in own:
            if old not in retained:
                if self.archive:
                    assert old.with_suffix(".archive.json").is_file()
                old.unlink()
        self.saved = [path for path in self.saved if path in retained]
        if self.resume_every:
            for directory in attempts:
                for slot in directory.glob("resume-*.pt"):
                    assert checkpoint_position(slot)[0] < self.iteration
                    slot.unlink()

    def archive_candidate(self, path):
        if not self.archive:
            return
        assert path.name == f"iteration-{self.iteration}.pt"
        with self.meter.phase("checkpoint"):
            if self._frozen_checkpoint is not None and not self.archived_frozen:
                archive_checkpoint(Path(self._frozen_checkpoint[0]["path"]))
                self.archived_frozen = True
            if self.critic_checkpoint is not None:
                archive_checkpoint(self.critic_checkpoint)
            archive_checkpoint(path)

    def finish_iteration(self):
        torch.cuda.synchronize()
        elapsed = perf_counter() - self.meter.started
        phases = {f"{name}_seconds": self.meter.seconds[name] for name in PHASES}
        counts = {
            name: self.meter.counts[name]
            for name in (
                "primary_tokens",
                "mc_tokens",
                "primary_continuations",
                "mc_continuations",
                "mc_expansions",
                "checkpoint_bytes",
            )
        }
        tokens = counts["primary_tokens"] + counts["mc_tokens"]
        row = (
            self.pending
            | counts
            | phases
            | {
                "warmup": len(self.rows) < self.warmup_iterations,
                "iteration_seconds": elapsed,
                "other_seconds": elapsed - sum(phases.values()),
                "mc_generation_seconds": self.meter.seconds["mc_generate_call"],
                "sampled_tokens": tokens,
                "sampled_tokens_per_second": tokens / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocated_bytes_end": torch.cuda.memory_allocated(),
                "reserved_bytes_end": torch.cuda.memory_reserved(),
            }
        )
        if self.tracker is not None:
            row["actor_learning_rate"] = self.optimizer.param_groups[0]["lr"]
            if self.critic_optimizer is not None:
                row["critic_learning_rate"] = self.critic_optimizer.param_groups[0][
                    "lr"
                ]
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        self.rows.append(row)
        if self.deadline is not None:
            self.deadline.observe(elapsed, phases["checkpoint_seconds"])
            status = json.loads(self.chain_status.read_text())
            atomic_record(self.chain_status, status | {"iteration": self.iteration})
            rollover = self.chain_status.with_name("ROLLOVER")
            if status["position"][0] > 0 and not rollover.exists():
                atomic_record(
                    rollover,
                    {
                        "job": status["job"],
                        "resumed_from": status["position"][0],
                        "completed": self.iteration,
                    },
                )
        if self.tracker is not None:
            self.tracker.log_iteration(row)
        self.pending = None
        self.meter.start()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=str, allow_nan=False)
        stream.write("\n")


def git(*args):
    return subprocess.check_output(
        ["git", "--no-pager", "--no-optional-locks", *args], cwd=ROOT, text=True
    ).strip()


def trainer_config(args):
    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        checkpoint_every = args.num_iterations if args.mode == "benchmark" else 1
    return PPOTrainerConfig(
        checkpoint_dir=args.output / "checkpoints",
        algorithm_config={
            "vineppo": VinePPOConfig,
            "ppo": lambda: PPOConfig(lam=1.0, value_clip_eps=0.2),
            "grpo": GRPOConfig,
            "rloo": RLOOConfig,
            "restem": RestEMConfig,
        }[args.algorithm](),
        checkpoint_every=checkpoint_every,
        **{
            name: getattr(args, name)
            for name in (
                "num_iterations",
                "prompts_per_iteration",
                "num_samples",
                "mc_num_samples",
                "update_epochs",
                "minibatch_size",
                "microbatch_size",
                "max_sequence_length",
                "max_new_tokens",
                "kl_coef",
                "seed",
            )
        },
    )


def validate_grading_resources(workers):
    if workers == 1:
        return
    cpus = os.process_cpu_count()
    assert cpus is not None, "Unable to determine available CPU count"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        cpus = min(cpus, int(os.environ["SLURM_CPUS_PER_TASK"]))
    assert workers <= cpus, "MATH grader workers exceed available CPUs"
    if "SLURM_MEM_PER_NODE" in os.environ:
        memory_mib = int(os.environ["SLURM_MEM_PER_NODE"])
    elif "SLURM_MEM_PER_CPU" in os.environ:
        memory_mib = cpus * int(os.environ["SLURM_MEM_PER_CPU"])
    else:
        return
    assert workers * 1024 < memory_mib, (
        "Reserve 1 GiB per MATH worker plus learner memory"
    )


def main(argv=None):
    args = parse_args(argv)
    profile = prompt_protocol(args.model_protocol, args.task)
    assert (
        args.evaluation == "disabled"
        or args.max_sequence_length >= profile.context_size
    )
    deadline = None
    if deadline_file := os.environ.get("PACE_CHAIN_STOP_FILE"):
        deadline = BoundaryDeadline(
            int(os.environ["SLURM_JOB_END_TIME"]), Path(deadline_file), clock=time
        )
        signal.signal(signal.SIGUSR1, deadline.request_stop)
        status_path = Path(os.environ["PACE_CHAIN_STATUS_FILE"])
        status = json.loads(status_path.read_text())
        atomic_record(
            status_path, status | {"learner_step": os.environ["SLURM_STEP_ID"]}
        )
    validate_grading_resources(args.math_grader_workers)
    assert args.mode in ("train", "benchmark"), (
        "Dispatch final testing through the launcher"
    )
    learner_gpu = gpu_identity()
    gpu = torch.cuda.get_device_properties(0)
    assert (
        args.expected_gpu is None or args.expected_gpu.casefold() in gpu.name.casefold()
    ), f"Expected GPU {args.expected_gpu!r}, found {gpu.name!r}"
    if args.dtype == "bfloat16":
        assert torch.cuda.is_bf16_supported(including_emulation=False), (
            "Native BF16 required"
        )
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    examples, dataset_info = load_examples(args.dataset_path, args.task)
    assert len(examples) >= args.prompts_per_iteration
    config = trainer_config(args)
    checkpoint_every = config.checkpoint_every
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    assert tokenizer.is_fast and tokenizer.eos_token_id is not None
    tokenizer_sha256 = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")
    ).hexdigest()
    pad_was_missing = tokenizer.pad_token_id is None
    if pad_was_missing:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": getattr(torch, args.dtype),
        "attn_implementation": "sdpa",
        "use_safetensors": True,
    }
    actor = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), **model_kwargs
    ).to("cuda:0")
    reference = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), **model_kwargs
    ).to("cuda:0")
    if args.model_protocol == OLMO3:
        assert actor.config.model_type == "olmo3"
        assert actor.generation_config.eos_token_id == profile.eos_ids(tokenizer)
    reference.requires_grad_(False).eval()
    if args.finetuning == "lora":
        actor = apply_lora(actor, rank=args.lora_rank, alpha=args.lora_alpha)
    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    critic = critic_optimizer = None
    if args.algorithm == "ppo":
        critic = TokenCritic(
            AutoModel.from_pretrained(
                str(args.model_path), **(model_kwargs | {"dtype": torch.float32})
            ).to("cuda:0")
        )
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(), lr=args.critic_learning_rate, weight_decay=0.0
        )
    meter = Meter()
    workers = client = None
    if args.backend == "vllm":
        client, workers = VLLMClient.from_directory(
            args.worker_dir,
            count=args.num_workers,
            api_key=os.environ["VLLM_API_KEY"],
            model_path=args.model_path,
            other_gpu_uuids=[learner_gpu["uuid"]]
            if args.worker_placement == "dedicated"
            else [],
        )
        assert args.worker_placement == "dedicated" or (
            workers[0]["gpu"]["uuid"] == learner_gpu["uuid"]
        ), "A colocated worker must share the learner's GPU"
        assert all(
            args.expected_gpu is None
            or args.expected_gpu.casefold() in worker["gpu"]["name"].casefold()
            for worker in workers
        ), f"Expected worker GPUs {args.expected_gpu!r}: {workers}"
        make_inference = lambda: MeasuredVLLMInference(
            actor,
            client,
            pad_token_id=tokenizer.pad_token_id,
            export_dir=args.output / "actors",
            seed=args.seed,
            manifest={
                "base_model": str(args.model_path),
                "base_config_sha256": hashlib.sha256(
                    (args.model_path / "config.json").read_bytes()
                ).hexdigest(),
                "tokenizer_sha256": tokenizer_sha256,
            },
            meter=meter,
        )
    else:
        make_inference = lambda: MeasuredInference(
            actor, pad_token_id=tokenizer.pad_token_id, meter=meter
        )
    # Reserve only after configuration, data and model validation; never reuse an output.
    args.output.mkdir(parents=True, exist_ok=False)
    warmup = args.warmup_iterations if args.mode == "benchmark" else 0
    with (
        ExitStack() as tracking_context,
        (args.output / "metrics.jsonl").open("x") as stream,
        make_inference() as inference,
    ):
        trainer = MeasuredTrainer(
            config,
            actor,
            inference,
            optimizer=optimizer,
            reference_policy=reference,
            critic=critic,
            critic_optimizer=critic_optimizer,
            compact_lora_checkpoints=args.checkpoint_format == "compact-lora",
            math_grader_workers=args.math_grader_workers,
            meter=meter,
            stream=stream,
            warmup_iterations=warmup,
            keep_checkpoints=args.keep_checkpoints,
            resume_checkpoint_every=args.resume_checkpoint_every,
            deadline=deadline,
            archive=args.archive_checkpoints == "yes",
            polynomial_schedule=args.lr_schedule == POLYNOMIAL,
            add_prompt_special_tokens=profile.add_special_tokens,
            prompt_special_ids=profile.context_special_ids(tokenizer),
        )
        inputs = {
            "examples": examples,
            "tokenizer": tokenizer,
            "task": args.task,
            "formatter": (
                paper_prompt
                if args.model_protocol == RHOMATH
                else partial(profile.format, tokenizer)
            ),
            "reference_is_solution": args.task == "math",
        }
        audit = args.resume_audit == "yes"
        if audit and args.backend == "vllm":
            requests = tracking_context.enter_context(
                (args.output / "sampling-requests.jsonl").open("x", buffering=1)
            )

            def record_requests(iteration, call, bodies):
                requests.write(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "call": call,
                            "requests": bodies,
                        }
                    )
                    + "\n"
                )

            inference.record_requests = record_requests
        if args.resume is not None:
            trainer.load_checkpoint(
                args.resume, **inputs, **({"audit": True} if audit else {})
            )
            if audit:
                write_json(
                    args.output / "resume-audit.json",
                    {
                        "checkpoint": str(args.resume),
                        "iteration": trainer.iteration,
                        "state_bitwise_equal": True,
                    },
                )
            lr_key = "initial_lr" if trainer.schedule is not None else "lr"
            assert all(
                g[lr_key] == args.learning_rate for g in optimizer.param_groups
            ), "--learning-rate must match the resumed optimizer"
            if critic_optimizer is not None:
                assert all(
                    g[lr_key] == args.critic_learning_rate
                    for g in critic_optimizer.param_groups
                ), "--critic-learning-rate must match the resumed optimizer"
        assert args.num_iterations - trainer.iteration > warmup or (
            args.resume is not None and trainer.iteration == args.num_iterations
        ), "Need non-warmup new iterations after resume"
        metadata = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "cli": vars(args),
            "trainer_config": asdict(config),
            "learning_rate_schedule": (
                trainer.schedule.identity() if trainer.schedule is not None else None
            ),
            "model": {
                "path": str(args.model_path),
                "config": actor.config.to_dict(),
                "config_sha256": hashlib.sha256(
                    (args.model_path / "config.json").read_bytes()
                ).hexdigest(),
                "parameters": sum(p.numel() for p in actor.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in actor.parameters() if p.requires_grad
                ),
                "trainable_dtypes": sorted(
                    {str(p.dtype) for p in actor.parameters() if p.requires_grad}
                ),
                "lora": lora_identity(actor) if args.finetuning == "lora" else None,
                "weights_content_hashed": False,
            },
            "dataset": {"path": str(args.dataset_path), **dataset_info},
            "math_segmentation_policy": MATH_SEGMENTATION_POLICY
            if args.task == "math" and args.algorithm == "vineppo"
            else None,
            "mc_prefix_policy": MC_PREFIX_POLICY
            if args.algorithm == "vineppo"
            else None,
            "prompt": profile.template,
            "tokenizer_sha256": tokenizer_sha256,
            "pad_uses_eos_because_missing": pad_was_missing,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                **{
                    name: version(name)
                    for name in ("torch", "transformers", "datasets", "peft")
                },
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
            "gpu": {
                "name": gpu.name,
                "total_memory": gpu.total_memory,
                "capability": [gpu.major, gpu.minor],
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                **learner_gpu,
            },
            "workers": workers,
            **slurm_attempt(),
            "git": {
                "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")),
            },
            "start_iteration": trainer.iteration,
            "precision": (
                "FP32 actor adapters/gradients/moments; frozen base/reference use dtype; "
                "PPO critic uses FP32"
                if args.finetuning == "lora"
                else "FP32 parameters, gradients and AdamW moments"
            ),
            "measurement_contract": __doc__,
        }
        if args.resume_selection is not None:
            parent_wandb = args.resume_selection.parent / "wandb.json"
            metadata["selection"] = {
                "resume_selection": str(args.resume_selection),
                "manifest_sha256": hashlib.sha256(
                    args.resume_selection.read_bytes()
                ).hexdigest(),
                "training_run": {
                    "path": str(args.resume_selection.parent),
                    "wandb": json.loads(parent_wandb.read_text())
                    if parent_wandb.exists()
                    else None,
                },
            }
        trainer.evaluation_provenance = {
            "schema_version": 1,
            "task": args.task,
            "prompt": profile.template,
            "tokenizer_sha256": tokenizer_sha256,
            "dataset": metadata["dataset"],
        }
        if args.model_protocol != RHOMATH:
            trainer.evaluation_provenance["model_protocol"] = args.model_protocol
        write_json(args.output / "metadata.json", metadata)
        tracker = tracking_context.enter_context(WandBTracker(args, metadata))
        if args.wandb_mode != "disabled":
            trainer.tracker = tracker
        if args.evaluation != "disabled":
            trainer.selection = CheckpointSelection(
                args, checkpoint_every, trainer.iteration
            )
            trainer.selection.recover(trainer.iteration)
            if args.resume is None:
                with meter.phase("validation"):
                    trainer.selection.baseline()
        if args.resume is not None and args.resume.name.startswith("iteration-"):
            trainer.archive_candidate(args.resume)
        if trainer.iteration < args.num_iterations:
            trainer.train(
                **inputs,
                record_metrics=trainer.record_metrics,
                stop_at_boundary=trainer.stop_at_boundary,
            )
        assert trainer.pending is None
        measured = [row for row in trainer.rows if not row["warmup"]]
        seconds = sum(row["iteration_seconds"] for row in measured)
        tokens = sum(row["sampled_tokens"] for row in measured)
        write_json(
            args.output / "summary.json",
            {
                "schema_version": 1,
                "mode": args.mode,
                "completed_iteration": trainer.iteration,
                "iterations_executed": len(trainer.rows),
                "iterations_measured": len(measured),
                "warmup_iterations_excluded": warmup,
                "iteration_seconds_total": seconds,
                "iteration_seconds_mean": seconds / len(measured) if measured else None,
                "sampled_tokens": tokens,
                "sampled_tokens_per_second": tokens / seconds if measured else None,
                "phase_seconds_total": {
                    name: sum(row[f"{name}_seconds"] for row in measured)
                    for name in PHASES
                },
                "sampling_totals": {
                    name: sum(row[name] for row in measured)
                    for name in (
                        "primary_tokens",
                        "mc_tokens",
                        "primary_continuations",
                        "mc_continuations",
                        "mc_expansions",
                    )
                },
                "peak_allocated_bytes": max(
                    (row["peak_allocated_bytes"] for row in measured), default=0
                ),
                "peak_reserved_bytes": max(
                    (row["peak_reserved_bytes"] for row in measured), default=0
                ),
                "checkpoints": [
                    {"path": str(path), "bytes": path.stat().st_size}
                    for path in trainer.saved
                ],
                "worker_stats": client.stats if client is not None else None,
            },
        )
        tracker.finish_summary(json.loads((args.output / "summary.json").read_text()))
        if trainer.selection is not None and trainer.iteration == args.num_iterations:
            frozen = trainer.selection.freeze()
            tracker.log_selection(frozen)
            if (
                args.evaluation == "validation-test"
                and deadline
                and deadline.should_stop()
            ):
                write_json(
                    args.output / "test-deferred.json",
                    {"selection": str(trainer.selection.frozen_path)},
                )
            elif args.evaluation == "validation-test":
                # No more updates: release all training GPU owners before final test.
                inference.close()
                trainer.policy = trainer.reference_policy = trainer.critic = None
                trainer.optimizer = trainer.critic_optimizer = None
                trainer.schedule = None
                del actor, reference, critic, optimizer, critic_optimizer
                gc.collect()
                torch.cuda.empty_cache()
                result, seconds = trainer.selection.test()
                tracker.log_final_test(result)
                write_json(
                    args.output / "final-evaluation.json",
                    {
                        "selection": str(trainer.selection.frozen_path),
                        "result": str(
                            args.output / "evaluations/test-selected/result.json"
                        ),
                        "seconds": seconds,
                    },
                )
    print(
        f"Completed iteration {trainer.iteration}; metrics and summary: {args.output}"
    )


if __name__ == "__main__":
    main()
