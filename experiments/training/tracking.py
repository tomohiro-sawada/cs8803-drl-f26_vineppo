"""Opt-in scalar tracking; each invocation owns a fresh run and budget counters."""

import json
import math
import os
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FIELDS = (
    "cli",
    "trainer_config",
    "dataset",
    "gpu",
    "software",
    "git",
    "model",
    "precision",
    "start_iteration",
)
CLI_FIELDS = frozenset(
    [
        "mode",
        "algorithm",
        "task",
        "model_path",
        "dataset_path",
        "output",
        "expected_gpu",
        "num_iterations",
        "prompts_per_iteration",
        "num_samples",
        "mc_num_samples",
        "update_epochs",
        "minibatch_size",
        "microbatch_size",
        "max_sequence_length",
        "max_new_tokens",
        "keep_checkpoints",
        "warmup_iterations",
        "learning_rate",
        "critic_learning_rate",
        "seed",
        "checkpoint_every",
        "resume",
        "dtype",
        "finetuning",
        "lora_rank",
        "lora_alpha",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_name",
    ]
)
BUDGET_FIELDS = (
    "primary_tokens",
    "mc_tokens",
    "sampled_tokens",
    "primary_continuations",
    "mc_continuations",
    "mc_expansions",
    "iteration_seconds",
)
SUMMARY_FIELDS = frozenset(
    [
        "schema_version",
        "mode",
        "completed_iteration",
        "iterations_executed",
        "iterations_measured",
        "warmup_iterations_excluded",
        "iteration_seconds_total",
        "iteration_seconds_mean",
        "sampled_tokens",
        "sampled_tokens_per_second",
        "phase_seconds_total",
        "sampling_totals",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "checkpoints",
    ]
)


def _json_value(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


class WandBTracker:
    """Use as ``with WandBTracker(args, metadata) as tracker``.

    The runner supplies trusted metadata/summary and scalar iteration rows. CLI
    fields are allowlisted independently to keep future credential flags private.
    Budgets include warmup and reset on every invocation, including checkpoint
    resume; ``iteration`` retains the trainer's absolute iteration number.
    The runner must create ``args.output`` before entering an enabled tracker.
    """

    def __init__(self, args, metadata):
        assert args.wandb_mode in ("disabled", "offline", "online")
        self.args = args
        self.metadata = metadata
        self.run = None
        self._entered = False
        self._last_iteration = metadata.get("start_iteration", 0)
        self._budget = dict.fromkeys(BUDGET_FIELDS, 0)

    def __enter__(self):
        assert not self._entered, "Tracker contexts cannot be reused"
        self._entered = True
        if self.args.wandb_mode == "disabled":
            return self
        if self.args.wandb_mode == "online":
            if "WANDB_API_KEY" not in os.environ:
                from dotenv import dotenv_values

                key = dotenv_values(ROOT / ".env", interpolate=False).get(
                    "WANDB_API_KEY"
                )
                if key:
                    os.environ["WANDB_API_KEY"] = key
            if not os.environ.get("WANDB_API_KEY", "").strip():
                raise RuntimeError(
                    "Online W&B requires WANDB_API_KEY in the environment or ROOT/.env"
                )

        import wandb

        config = {
            key: self.metadata[key] for key in CONFIG_FIELDS if key in self.metadata
        }
        config["cli"] = {
            key: value
            for key, value in self.metadata.get("cli", {}).items()
            if key in CLI_FIELDS
        }
        config["budget_scope"] = "invocation"
        self.run = wandb.init(
            mode=self.args.wandb_mode,
            project=getattr(self.args, "wandb_project", "vineppo"),
            entity=getattr(self.args, "wandb_entity", None),
            group=getattr(self.args, "wandb_group", None),
            name=getattr(self.args, "wandb_name", None),
            dir=str(self.args.output),
            resume="never",
            id=uuid4().hex,
            config=_json_value(config),
            settings=wandb.Settings(
                console="off",
                disable_code=True,
                disable_git=True,
                x_disable_meta=True,
                x_save_requirements=False,
                stop_on_fatal_error=True,
                finish_timeout=60,
                finish_timeout_raises=True,
            ),
        )
        try:
            self.run.define_metric("iteration")
            self.run.define_metric("*", step_metric="iteration")
            with (self.args.output / "wandb.json").open("x") as stream:
                json.dump(
                    {
                        "id": self.run.id,
                        "url": self.run.url
                        if self.args.wandb_mode == "online"
                        else None,
                        "mode": self.args.wandb_mode,
                    },
                    stream,
                    indent=2,
                )
                stream.write("\n")
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def log_iteration(self, row):
        """Log unchanged scalar keys plus cumulative ``budget/<counter>`` keys."""
        if self.args.wandb_mode == "disabled":
            return
        assert self.run is not None, "Enter the tracker before logging"
        assert all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in row.values()
        ), "Iteration metrics must be finite scalars"
        iteration = row["iteration"]
        assert type(iteration) is int and iteration > self._last_iteration, (
            "Iterations must increase"
        )
        assert all(row[key] >= 0 for key in BUDGET_FIELDS), "Budgets cannot decrease"
        assert not any(key.startswith("budget/") for key in row), (
            "Budget keys belong to the tracker"
        )
        for key in BUDGET_FIELDS:
            self._budget[key] += row[key]
        self.run.log(
            row | {f"budget/{key}": value for key, value in self._budget.items()},
            step=iteration,
        )
        self._last_iteration = iteration

    def finish_summary(self, summary):
        """Copy the runner's known summary schema without renaming or flattening."""
        if self.args.wandb_mode == "disabled":
            return
        assert self.run is not None, "Enter the tracker before logging"
        self.run.summary.update(
            _json_value(
                {key: value for key, value in summary.items() if key in SUMMARY_FIELDS}
            )
        )

    def __exit__(self, exc_type, exc, traceback):
        if self.run is None:
            return False
        try:
            try:
                if exc_type is not None:
                    self.run.summary["failure_type"] = exc_type.__name__
            finally:
                self.run.finish(exit_code=int(exc_type is not None))
        except Exception as cleanup_error:
            if exc is None:
                raise
            # Never replace the training failure with a secondary SDK failure.
            exc.add_note(f"W&B cleanup failed: {type(cleanup_error).__name__}")
        finally:
            self.run = None
        return False
