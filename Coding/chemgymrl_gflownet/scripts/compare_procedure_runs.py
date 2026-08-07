#!/usr/bin/env python3
"""Compare analysed ChemGymRL procedure GFlowNet runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELD_PATTERN = re.compile(r"([A-Za-z_]+):(-?[0-9]+(?:\.[0-9]+)?)")
ITERATION_PATTERN = re.compile(r"logger - iteration ([0-9]+) :")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "SUMMARY_JSON", "TRAIN_LOG"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("at least two --run entries are required")
    return args


def parse_train_log(path: Path) -> list[dict[str, float | int]]:
    records = []
    with path.open(encoding="utf8") as input_file:
        for line in input_file:
            iteration_match = ITERATION_PATTERN.search(line)
            if not iteration_match:
                continue
            values = {name: float(value) for name, value in FIELD_PATTERN.findall(line)}
            records.append({"iteration": int(iteration_match.group(1)), **values})
    if not records:
        raise ValueError(f"no training iterations found in {path}")
    return records


def load_run(label: str, summary_path: Path, log_path: Path) -> dict[str, object]:
    with summary_path.open(encoding="utf8") as input_file:
        summary = json.load(input_file)
    records = parse_train_log(log_path)
    required = {"loss", "logZ", "batch_entropy", "sampled_reward_avg"}
    if any(not required.issubset(record) for record in records):
        raise ValueError(f"training log {log_path} is missing required metrics")
    return {
        "label": label,
        "summary_path": str(summary_path.resolve()),
        "train_log": str(log_path.resolve()),
        "summary": summary,
        "training": records,
    }


def run_row(run: dict[str, object]) -> dict[str, object]:
    summary = run["summary"]
    records = run["training"]
    first_window = records[: min(5, len(records))]
    last_window = records[-min(5, len(records)) :]
    generated_rates = summary["category_rates"]["generated"]
    target_rates = summary["category_rates"]["target"]
    category_total_variation = 0.5 * sum(
        abs(generated_rates[category] - target_rates[category])
        for category in ("negative", "zero", "positive")
    )
    return {
        "label": run["label"],
        "training_steps": records[-1]["iteration"],
        "final_loss": records[-1]["loss"],
        "first_window_mean_loss": float(np.mean([record["loss"] for record in first_window])),
        "last_window_mean_loss": float(np.mean([record["loss"] for record in last_window])),
        "final_logZ": records[-1]["logZ"],
        "exact_log_partition": summary["exact_log_partition"],
        "final_batch_entropy": records[-1]["batch_entropy"],
        "sample_count": summary["sample_count"],
        "unique_procedure_count": summary["unique_procedure_count"],
        "target_expected_unique_count": summary["target_sampling_benchmarks"][
            "expected_unique_procedure_count"
        ],
        "duplicate_fraction": summary["concentration"]["duplicate_fraction"],
        "target_expected_duplicate_fraction": summary["target_sampling_benchmarks"][
            "expected_duplicate_fraction"
        ],
        "most_repeated_count": summary["concentration"]["most_repeated_count"],
        "exact_target_mass_covered": summary["exact_target_mass_covered_by_unique_samples"],
        "target_expected_mass_covered": summary["target_sampling_benchmarks"][
            "expected_target_mass_covered"
        ],
        "mean_raw_reward": summary["raw_reward"]["mean"],
        "mean_reward_absolute_error": abs(
            summary["raw_reward"]["mean"]
            - summary["raw_reward_benchmarks"]["exact_target_expected"]
        ),
        "category_total_variation": category_total_variation,
        "negative_rate": generated_rates["negative"],
        "zero_rate": generated_rates["zero"],
        "positive_rate": generated_rates["positive"],
        "best_exact_rank": summary["rank"]["best"],
        "top_10_rate": summary["top_k"]["top_10"]["generated_rate"],
        "top_100_rate": summary["top_k"]["top_100"]["generated_rate"],
        "top_1000_rate": summary["top_k"]["top_1000"]["generated_rate"],
        "unique_top_10": summary["top_k"]["top_10"]["generated_unique_count"],
        "target_expected_unique_top_10": summary["top_k"]["top_10"][
            "target_expected_unique_count"
        ],
        "unique_top_100": summary["top_k"]["top_100"]["generated_unique_count"],
        "target_expected_unique_top_100": summary["top_k"]["top_100"][
            "target_expected_unique_count"
        ],
        "unique_top_1000": summary["top_k"]["top_1000"]["generated_unique_count"],
        "target_expected_unique_top_1000": summary["top_k"]["top_1000"][
            "target_expected_unique_count"
        ],
        "replay_size": records[-1].get("replay_size"),
        "replay_retained": records[-1].get("replay_retained"),
        "replay_raw_reward_avg": records[-1].get("replay_raw_reward_avg"),
        "replay_positive_fraction": records[-1].get("replay_positive_fraction"),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_runs(runs: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    loss_axis, logz_axis, category_axis, reward_axis = axes.flat

    for run in runs:
        iterations = [record["iteration"] for record in run["training"]]
        losses = [record["loss"] for record in run["training"]]
        logz = [record["logZ"] for record in run["training"]]
        loss_axis.plot(iterations, losses, label=run["label"])
        logz_axis.plot(iterations, logz, label=run["label"])
    loss_axis.set(xlabel="Training step", ylabel="TB loss", title="Training loss", yscale="log")
    loss_axis.legend(frameon=False)
    exact_logz = float(runs[0]["summary"]["exact_log_partition"])
    logz_axis.axhline(exact_logz, color="#333333", linestyle="--", label="Exact log partition")
    logz_axis.set(xlabel="Training step", ylabel="logZ", title="Partition estimate")
    logz_axis.legend(frameon=False)

    baseline = runs[0]["summary"]["category_rates"]
    series = [
        ("Uniform", baseline["uniform"]),
        *[(run["label"], run["summary"]["category_rates"]["generated"]) for run in runs],
        ("Exact target", baseline["target"]),
    ]
    categories = ["negative", "zero", "positive"]
    x = np.arange(len(categories))
    width = 0.8 / len(series)
    for index, (label, rates) in enumerate(series):
        values = [rates[category] for category in categories]
        category_axis.bar(x + (index - (len(series) - 1) / 2) * width, values, width, label=label)
    category_axis.set(
        xticks=x,
        xticklabels=[category.title() for category in categories],
        ylabel="Fraction",
        title="Final reward categories",
    )
    category_axis.legend(frameon=False, fontsize=8)

    reward_labels = ["Uniform", *[run["label"] for run in runs], "Exact target"]
    reward_values = [
        baseline_value(runs, "uniform_expected"),
        *[run["summary"]["raw_reward"]["mean"] for run in runs],
        baseline_value(runs, "exact_target_expected"),
    ]
    reward_colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(reward_labels)))
    reward_axis.bar(reward_labels, reward_values, color=reward_colours)
    reward_axis.axhline(0, color="#333333", linewidth=0.8)
    reward_axis.set(ylabel="Mean raw reward", title="Final mean reward")
    reward_axis.tick_params(axis="x", rotation=15)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("ChemGymRL procedure GFlowNet convergence comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def baseline_value(runs: list[dict[str, object]], key: str) -> float:
    return float(runs[0]["summary"]["raw_reward_benchmarks"][key])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        load_run(label, Path(summary_path).resolve(), Path(log_path).resolve())
        for label, summary_path, log_path in args.run
    ]
    rows = [run_row(run) for run in runs]

    write_csv(rows, output_dir / "run_comparison.csv")
    with (output_dir / "comparison_summary.json").open("w", encoding="utf8") as output_file:
        json.dump({"runs": rows}, output_file, indent=2)
        output_file.write("\n")
    plot_runs(runs, output_dir / "run_comparison.png")
    print(json.dumps({"runs": rows}, indent=2))


if __name__ == "__main__":
    main()
