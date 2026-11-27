"""Draw the results-summary figures from collect_results_summary.py's JSON.

    uv run --no-project --with matplotlib python \
        scripts/figures/plot_results_summary.py RESULTS.json OUT_DIR

Writes the PNGs to OUT_DIR. matplotlib is not a project dependency, hence
the uv invocation.
"""

import json
import sys
from pathlib import Path
from statistics import linear_regression, mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "SFT": "#7f7f7f",
    "RestEM": "#8c6d31",
    "RLOO": "#6a51a3",
    "GRPO": "#1f6f8b",
    "PPO": "#d98c2b",
    "VinePPO": "#b5493a",
}
PHASES = {
    "Primary sampling": "primary_generation_seconds",
    "MC value estimation": "mc_estimation_seconds",
    "Critic values": "ppo_estimation_seconds",
    "Old log-probs": "score_cache_seconds",
    "Actor update": "actor_update_seconds",
    "Critic update": "critic_update_seconds",
    "Checkpoint": "checkpoint_seconds",
    "Validation": "validation_seconds",
}


def color(arm):
    return COLORS[arm.split()[0]]


def label(arm):
    """Tick label: the method, with the critic rank for LoRA-critic PPO."""
    method, *rest = arm.split()
    return (
        f"{method}\n{rest[0]}" if rest[0:1] and rest[0].startswith("rank") else method
    )


def style(axis, ylabel=None, title=None):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.8)
    axis.set_axisbelow(True)
    if ylabel:
        axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title, fontsize=11, fontweight="bold")


def interval_bars(axis, arms, key="test", offset=0.0, width=0.6, **kwargs):
    for index, (arm, runs) in enumerate(arms.items()):
        (run,) = runs
        score, low, high = run[key]
        axis.bar(index + offset, score, width, color=color(arm), **kwargs)
        axis.errorbar(
            index + offset,
            score,
            yerr=[[score - low], [high - score]],
            color="black",
            capsize=3,
            linewidth=1,
        )
        axis.text(index + offset, high + 0.3, f"{score:.1f}", ha="center", fontsize=8)
    axis.set_xticks(range(len(arms)), [label(arm) for arm in arms])


def seed_bars(axis, rho, arms):
    """Bar: mean over seeds. Dots: seeds. One seed: its 95% interval by question."""
    for index, arm in enumerate(arms):
        runs = rho[arm]
        scores = [run["test"][0] for run in runs]
        top = max(scores)
        axis.bar(
            index,
            mean(scores),
            0.6,
            color=color(arm),
            alpha=0.35,
            hatch="//" if "rank" in arm else None,
        )
        axis.scatter([index] * len(scores), scores, color=color(arm), zorder=3)
        if len(runs) == 1:
            score, low, top = runs[0]["test"]
            axis.errorbar(
                index,
                score,
                yerr=[[score - low], [top - score]],
                color="black",
                capsize=3,
                linewidth=1,
            )
        axis.text(index, top + 0.4, f"{mean(scores):.1f}", ha="center", fontsize=8)
    axis.set_xticks(range(len(arms)), [label(arm) for arm in arms])


def test_figure(rho, out):
    figure, (short, long) = plt.subplots(
        1, 2, figsize=(12, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.4, 1]}
    )
    baseline = rho["SFT"][0]["test"][0]
    seed_bars(
        short,
        rho,
        ["RestEM matched", "RLOO 200", "GRPO 200", "PPO 200", "PPO rank-16 200"]
        + ["VinePPO 200"],
    )
    style(short, "Test Pass@1 (%)", "25,600 primary samples")
    seed_bars(long, rho, ["RLOO 650", "GRPO 650", "PPO 650", "VinePPO 650"])
    style(long, title="332,800 primary samples")
    for axis in (short, long):
        axis.axhline(baseline, color=COLORS["SFT"], linestyle="--", linewidth=1)
        axis.set_xlim(left=-1.1)
        axis.text(-1.05, baseline + 0.3, f"SFT {baseline:.1f}", fontsize=8)
    short.set_ylim(36, 62)
    figure.suptitle(
        "RhoMath 1.1B on GSM8K: validation-selected checkpoint, 16 test rounds. "
        "Dots: seeds; error bars: one seed's 95% CI"
    )
    figure.tight_layout()
    figure.savefig(out / "rhomath-gsm8k-test.png", dpi=150)


def smooth(values, window=25):
    return [
        mean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def curve_figure(rho, out):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm in ("RLOO 200", "GRPO 200", "PPO 200", "VinePPO 200"):
        curves = [{i: score for i, score, _ in run["validation"]} for run in rho[arm]]
        steps = sorted(curves[0])
        assert all(sorted(curve) == steps for curve in curves), arm
        axes[0].plot(
            steps,
            [mean(curve[step] for curve in curves) for step in steps],
            marker="o",
            markersize=3,
            color=color(arm),
            label=arm.split()[0],
        )
    style(axes[0], "Validation Pass@1 (%)", "200 iterations x 128 (mean of 3 seeds)")
    for arm in ("RLOO 650", "GRPO 650", "PPO 650", "VinePPO 650"):
        for index, run in enumerate(rho[arm]):
            faint = {"alpha": 0.45, "linewidth": 1} if index else {}
            steps, scores, _ = zip(*run["validation"])
            axes[1].plot(
                steps, scores, marker="o", markersize=3, color=color(arm), **faint
            )
            axes[1].scatter(
                [run["selected"]],
                [scores[steps.index(run["selected"])]],
                s=90,
                facecolors="none",
                edgecolors=color(arm),
            )
            steps, rewards = zip(*run["reward"])
            axes[2].plot(steps, smooth(rewards), color=color(arm), **faint)
    style(axes[1], title="650 iterations x 512 (faint: seeds 43, 44; ring: selected)")
    style(axes[2], "Mean training reward", "650 x 512: reward, 25-iteration mean")
    for axis in axes:
        axis.set_xlabel("Training iteration")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(out / "rhomath-gsm8k-curves.png", dpi=150)


def cost_figure(rho, out):
    figure, (phases, curve) = plt.subplots(1, 2, figsize=(14, 4.6))
    arms = ["RLOO 650", "GRPO 650", "VinePPO 650", "PPO 650"]
    left = [0.0] * len(arms)
    palette = plt.get_cmap("tab10")
    for index, (label, key) in enumerate(PHASES.items()):
        hours = [rho[arm][0]["phase_seconds"][key] / 3600 for arm in arms]
        phases.barh(arms, hours, left=left, color=palette(index), label=label)
        left = [a + b for a, b in zip(left, hours)]
    for row, arm in enumerate(arms):
        total = rho[arm][0]["phase_seconds"]["iteration_seconds"] / 3600
        phases.text(total + 0.2, row, f"{total:.1f} h", va="center", fontsize=8)
        _, scores, hours = zip(*rho[arm][0]["validation"])
        curve.plot(hours, scores, marker="o", markersize=3, color=color(arm), label=arm)
    phases.invert_yaxis()
    phases.set_xlim(right=30)
    phases.spines[["top", "right"]].set_visible(False)
    phases.set_xlabel("Hours in recorded training iterations (one H100)")
    phases.set_title("Where the time goes", fontsize=11, fontweight="bold")
    phases.legend(
        frameon=False,
        fontsize=8,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    style(curve, "Validation Pass@1 (%)", "Validation against time spent")
    curve.set_xlabel("Hours in recorded training iterations (one H100)")
    curve.legend(frameon=False, fontsize=8)
    figure.suptitle("RhoMath 1.1B on GSM8K, 650 iterations x 512, seed 42")
    figure.tight_layout()
    figure.savefig(out / "rhomath-650-cost.png", dpi=150)


def generalization_figure(rho, out):
    """Validation against training accuracy per checkpoint (the paper's Figure 5).

    Each arm's line is the least-squares fit over all its seeds' checkpoints.
    """
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    for arm in ("RLOO 650", "GRPO 650", "PPO 650", "VinePPO 650"):
        points = [
            (accuracy, {i: score for i, score, _ in run["validation"]}[i])
            for run in rho[arm]
            for i, accuracy in run["train_accuracy"]
        ]
        train, scores = zip(*points)
        slope, intercept = linear_regression(train, scores)
        axis.scatter(train, scores, s=12, color=color(arm), alpha=0.6)
        ends = [min(train), max(train)]
        axis.plot(
            ends,
            [intercept + slope * x for x in ends],
            color=color(arm),
            label=f"{arm.split()[0]}: slope {slope:.2f}, {len(rho[arm])} seed(s)",
        )
    style(axis, "Validation Pass@1 (%)", "Validation against training accuracy")
    axis.set_xlabel("Training accuracy over the preceding 25 iterations (%)")
    axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "RhoMath 1.1B on GSM8K, 650 iterations x 512, one dot per checkpoint"
    )
    figure.tight_layout()
    figure.savefig(out / "rhomath-650-generalization.png", dpi=150)


CRITICS = {
    "PPO 200": "Full FP32",
    "PPO rank-16 200": "LoRA 16",
    "PPO rank-64 200": "LoRA 64",
    "PPO rank-256 200": "LoRA 256",
}
REPLAY_STYLE = {
    "Full critic, seed 43": ("#d98c2b", "-"),
    "Rank 16, seed 43": ("#8c510a", "--"),
    "Rank 16, seed 42": ("#8c510a", ":"),
    "Rank 64, seed 42": ("#bf812d", ":"),
    "Rank 256, seed 42": ("#dfc27d", ":"),
}


def critic_figure(data, out):
    """Full against LoRA critics: accuracy, time per update and value error."""
    rho, replay = data["rhomath-gsm8k"], data["value_replay"]
    figure, (test, seconds, error) = plt.subplots(
        1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.5]}
    )
    for index, (arm, name) in enumerate(CRITICS.items()):
        scores = [run["test"][0] for run in rho[arm]]
        test.bar(index, mean(scores), 0.6, color=color(arm), alpha=0.35)
        test.scatter([index] * len(scores), scores, color=color(arm), zorder=3)
        test.text(
            index, max(scores) + 0.3, f"{mean(scores):.1f}", ha="center", fontsize=8
        )
        per_update = {
            key: mean(run["phase_seconds"][key] / run["iterations"] for run in rho[arm])
            for key in ("critic_update_seconds", "actor_update_seconds")
        }
        bottom = 0.0
        for key, shade in (
            ("critic_update_seconds", 1.0),
            ("actor_update_seconds", 0.4),
        ):
            seconds.bar(
                index,
                per_update[key],
                0.6,
                bottom=bottom,
                color=color(arm),
                alpha=shade,
            )
            bottom += per_update[key]
        seconds.text(
            index,
            bottom + 0.4,
            f"{per_update['critic_update_seconds']:.1f} + "
            f"{per_update['actor_update_seconds']:.1f}",
            ha="center",
            fontsize=8,
        )
    for axis in (test, seconds):
        axis.set_xticks(range(len(CRITICS)), list(CRITICS.values()))
    test.set_ylim(40, 52)
    style(test, "Test Pass@1 (%)", "PPO, 200 x 128 (dots: seeds)")
    style(seconds, "Seconds per update", "Critic update (dark) + actor update")
    for name, (shade, line) in REPLAY_STYLE.items():
        critic = replay[name]["critic"]
        depths = sorted(critic["by_depth"], key=int)
        error.plot(
            [int(depth) for depth in depths],
            [critic["by_depth"][depth]["mean_absolute_error"] for depth in depths],
            color=shade,
            linestyle=line,
            marker="o",
            markersize=3,
            label=f"{name} (EV {critic['explained_variance']:.2f})",
        )
    reference = replay["Full critic, seed 43"]
    for k, shade in ((1, "#f4a582"), (3, "#d6604d"), (9, "#b2182b")):
        estimate = reference[f"mc_{k}"]
        depths = sorted(estimate["by_depth"], key=int)
        error.plot(
            [int(depth) for depth in depths],
            [estimate["by_depth"][depth]["mean_absolute_error"] for depth in depths],
            color=shade,
            marker="s",
            markersize=3,
            label=f"MC, K = {k} (EV {estimate['explained_variance']:.2f})",
        )
    style(error, "Mean absolute value error", "Value error by reasoning depth")
    error.set_xlabel("Step the state precedes (last point: that step or later)")
    error.set_ylim(0, 0.6)
    error.legend(frameon=False, fontsize=7, ncols=2, loc="upper center")
    figure.suptitle(
        "RhoMath 1.1B on GSM8K: full and LoRA critics at iteration 200. "
        "Value error is against 256 rollouts per state"
    )
    figure.tight_layout()
    figure.savefig(out / "rhomath-lora-critic.png", dpi=150)


def olmo_figure(data, out):
    figure, (math, gsm) = plt.subplots(
        1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    interval_bars(math, data["olmo-math"])
    math.set_ylim(60, 80)
    style(math, "Test Pass@1 (%)", "MATH, 500 questions")
    interval_bars(gsm, data["olmo-gsm8k"], offset=-0.2, width=0.38)
    interval_bars(
        gsm,
        data["olmo-gsm8k"],
        key="test_lenient",
        offset=0.2,
        width=0.38,
        alpha=0.45,
        hatch="//",
    )
    gsm.set_ylim(55, 95)
    style(
        gsm, title="GSM8K, 1,319 questions (solid: official grader; hatched: lenient)"
    )
    figure.suptitle(
        "OLMo 3 7B Instruct SFT: 6,400 primary samples, seed 42, bars: 95% CI"
    )
    figure.tight_layout()
    figure.savefig(out / "olmo-test.png", dpi=150)


def main(results, out):
    data = json.loads(Path(results).read_text())
    out = Path(out)
    rho = data["rhomath-gsm8k"]
    test_figure(rho, out)
    curve_figure(rho, out)
    cost_figure(rho, out)
    generalization_figure(rho, out)
    critic_figure(data, out)
    olmo_figure(data, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
