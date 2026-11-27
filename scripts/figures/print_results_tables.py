"""Print the results summary's tables as markdown from the collected JSON.

    python scripts/figures/print_results_tables.py RESULTS.json

Every number in the summary and the report draft is one of these cells. Scores
are test Pass@1 in percent; brackets are 95% intervals clustered by question;
differences are in points and paired on the same questions. "t interval" is
over the three training seeds.
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev

T_975_DF2 = 4.302652729911275


def score(triple):
    return "{:.2f}% [{:.2f}, {:.2f}]".format(*triple)


def gap(triple):
    return "{:+.2f} [{:+.2f}, {:+.2f}]".format(*triple)


def over_seeds(values):
    """Mean with a t interval over three seeds."""
    assert len(values) == 3
    half = T_975_DF2 * stdev(values) / 3**0.5
    return gap((mean(values), mean(values) - half, mean(values) + half))


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def seeds_table(title, arms, reference, paired):
    print(f"### {title}\n")
    rows = []
    for arm, runs in arms.items():
        scores = [run["test"][0] for run in runs]
        cells = [
            arm,
            f"{mean(scores):.2f} ± {stdev(scores):.2f}%"
            if len(runs) > 1
            else score(runs[0]["test"]),
            ", ".join(f"{run['seed']}: {run['test'][0]:.2f}" for run in runs),
            ", ".join(str(run["selected"]) for run in runs)
            if runs[0]["selected"] is not None
            else "",
        ]
        gaps = paired.get(f"{reference} minus {arm}", {})
        cells.append("; ".join(f"{seed}: {gap(gaps[seed])}" for seed in sorted(gaps)))
        cells.append(
            over_seeds([g[0] for g in gaps.values()]) if len(gaps) == 3 else ""
        )
        rows.append(cells)
    header = ["Arm", "Test Pass@1", "By seed", "Selected iteration"]
    header += [f"{reference} minus arm, same seed", "Mean gap, t interval"]
    table(header, rows)


def cost_table(title, arms):
    """Recorded training-iteration time and sampled tokens; shares are of that time."""
    print(f"### {title}\n")
    shares = {
        "MC estimation": "mc_estimation_seconds",
        "Critic update": "critic_update_seconds",
    }
    shares |= {
        "Actor update": "actor_update_seconds",
        "Checkpoint": "checkpoint_seconds",
    }
    rows = []
    for arm, runs in arms.items():
        for run in runs:
            phases = run["phase_seconds"]
            total = phases["iteration_seconds"]
            rows.append(
                [
                    arm,
                    str(run["seed"]),
                    str(run["iterations"]),
                    f"{total / 3600:.2f}",
                    f"{run['primary_tokens'] / 1e6:.2f}M",
                    f"{run['mc_tokens'] / 1e6:.1f}M" if run["mc_tokens"] else "",
                ]
                + [f"{100 * phases[key] / total:.1f}%" for key in shares.values()]
            )
    header = [
        "Arm",
        "Seed",
        "Logged iterations",
        "Hours",
        "Primary tokens",
        "MC tokens",
    ]
    table(header + list(shares), rows)


def critic_table(rho):
    print("### PPO critics, RhoMath GSM8K, 200 x 128\n")
    rows = []
    for arm in ("PPO 200", "PPO rank-16 200", "PPO rank-64 200", "PPO rank-256 200"):
        for run in rho[arm]:
            n, phases = run["iterations"], run["phase_seconds"]
            validation = {i: accuracy for i, accuracy, _ in run["validation"]}
            rows.append(
                [
                    "Full FP32" if arm == "PPO 200" else "LoRA " + arm.split()[1][5:],
                    str(run["seed"]),
                    str(run["selected"]),
                    f"{validation[run['selected']]:.2f}%",
                    f"{run['test'][0]:.2f}%",
                    "{:.3f} / {:.3f}".format(
                        run["health"]["critic/value_loss"],
                        run["health_tail"]["critic/value_loss"],
                    ),
                    "{:.3f} / {:.3f}".format(
                        run["health"]["raw_advantage_token_std"],
                        run["health_tail"]["raw_advantage_token_std"],
                    ),
                    f"{phases['iteration_seconds'] / n:.1f}",
                    f"{phases['critic_update_seconds'] / n:.1f}",
                    f"{phases['actor_update_seconds'] / n:.1f}",
                ]
            )
    header = [
        "Critic",
        "Seed",
        "Selected",
        "Validation",
        "Test",
        "Value loss, mean / last 25",
    ]
    header += ["Residual spread, mean / last 25", "s per update", "Critic s", "Actor s"]
    table(header, rows)


def replay_tables(replay):
    print("### Value estimates against 256 rollouts per state\n")
    estimators = ("critic", "mc_1", "mc_3", "mc_9")
    names = {
        "critic": "Critic",
        "mc_1": "MC, K = 1",
        "mc_3": "MC, K = 3",
        "mc_9": "MC, K = 9",
    }
    header = ["Checkpoint", "Responses", "States", "Mean value", "Reference SE"]
    for name in estimators:
        header += [f"{names[name]}: MAE", "EV"]
    header.insert(7, "Critic bias")
    rows = []
    for label, result in replay.items():
        row = [
            label,
            f"{result['responses']:,}",
            f"{result['states']:,}",
            f"{result['reference_mean']:.3f}",
            f"{result['reference_standard_error']:.3f}",
        ]
        for name in estimators:
            row += [
                f"{result[name]['mean_absolute_error']:.3f}",
                f"{result[name]['explained_variance']:.2f}",
            ]
        row.insert(7, f"{result['critic']['bias']:+.3f}")
        rows.append(row)
    table(header, rows)
    print("Mean absolute error by the step the state precedes (states in brackets):\n")
    depths = sorted(
        {d for r in replay.values() for d in r["critic"]["by_depth"]}, key=int
    )
    last = depths[-1]
    header = ["Estimator"] + [d if d != last else f"{d}+" for d in depths]
    rows = []
    for label, result in replay.items():
        cells = result["critic"]["by_depth"]
        rows.append(
            [f"Critic: {label}"]
            + [
                f"{cells[d]['mean_absolute_error']:.3f} ({cells[d]['pairs']:,})"
                if d in cells
                else ""
                for d in depths
            ]
        )
    for name in estimators[1:]:
        values = [
            mean(
                r[name]["by_depth"][d]["mean_absolute_error"]
                for r in replay.values()
                if d in r[name]["by_depth"]
            )
            for d in depths
        ]
        rows.append(
            [f"{names[name]}, mean of checkpoints"] + [f"{v:.3f}" for v in values]
        )
    table(header, rows)


def olmo_table(title, arms, paired, lenient):
    print(f"### {title}\n")
    header = ["Arm", "Selected", "Test, strict", "Gain over SFT, strict"]
    header += ["Test, lenient"] if lenient else []
    rows = []
    for arm, (run,) in arms.items():
        row = [arm, "" if run["selected"] is None else str(run["selected"])]
        row += [score(run["test"])]
        row += [gap(paired[f"{arm} minus SFT"]["42"]) if arm != "SFT" else ""]
        row += [score(run["test_lenient"])] if lenient else []
        rows.append(row)
    table(header, rows)
    for name, by_seed in paired.items():
        if "SFT" not in name:
            print(f"- {name}: {gap(by_seed['42'])}")
    print()


def sft_table(data):
    print("### SFT models under the validation protocol, GSM8K\n")
    tests = {
        "RhoMath": data["rhomath-gsm8k"]["SFT"],
        "OLMo 3": data["olmo-gsm8k"]["SFT"],
    }
    rows = []
    for model, result in data["sft_validation_protocol"].items():
        (test,) = tests[model]
        for reading in ("strict", "lenient"):
            rows.append(
                [
                    model,
                    reading,
                    score(result["validation"][reading]),
                    score(result["train373"][reading]),
                    score(test["test" if reading == "strict" else "test_lenient"])
                    if reading == "strict" or "test_lenient" in test
                    else "",
                ]
            )
    header = ["Model", "Reading", "Validation split (373)", "373 training questions"]
    table(header + ["Test split (1,319), 16 rounds"], rows)


def main(path):
    data = json.loads(Path(path).read_text())
    rho, paired = data["rhomath-gsm8k"], data["paired"]
    short = ["SFT", "RestEM matched", "RLOO 200", "GRPO 200", "PPO 200"]
    short += ["PPO rank-16 200", "VinePPO 200"]
    seeds_table(
        "RhoMath GSM8K, 200 x 128",
        {arm: rho[arm] for arm in short},
        "VinePPO 200",
        paired["rhomath-gsm8k"],
    )
    seeds_table(
        "RhoMath GSM8K, 650 x 512",
        {arm: rho[arm] for arm in rho if arm.endswith("650")},
        "VinePPO 650",
        paired["rhomath-gsm8k"],
    )
    for name in ("PPO rank-16 200 minus PPO 200", "PPO 650 minus GRPO 650"):
        gaps = paired["rhomath-gsm8k"][name]
        line = "; ".join(f"{s}: {gap(g)}" for s, g in sorted(gaps.items()))
        if len(gaps) == 3:
            line += "; mean, t interval: " + over_seeds([g[0] for g in gaps.values()])
        print(f"- {name}: {line}")
    print()
    long = {arm: rho[arm] for arm in rho if arm.endswith("650")}
    cost_table("RhoMath GSM8K, 650 x 512: time and tokens", long)
    cost_table(
        "RhoMath GSM8K, 200 x 128: time and tokens",
        {arm: rho[arm] for arm in short[2:]},
    )
    for task, name in (("olmo-gsm8k", "GSM8K"), ("olmo-math", "MATH")):
        trained = {arm: runs for arm, runs in data[task].items() if arm != "SFT"}
        cost_table(f"OLMo 3 on {name}: time and tokens", trained)
    critic_table(rho)
    replay_tables(data["value_replay"])
    olmo_table("OLMo 3 on GSM8K", data["olmo-gsm8k"], paired["olmo-gsm8k"], True)
    olmo_table("OLMo 3 on MATH", data["olmo-math"], paired["olmo-math"], False)
    sft_table(data)


if __name__ == "__main__":
    main(sys.argv[1])
