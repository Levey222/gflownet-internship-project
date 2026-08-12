#!/usr/bin/env python3
"""Summarise matched procedure GFlowNet methods across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib.pyplot as plt


METRICS = {
    "mean_raw_reward": ("Mean raw reward", "raw_reward", "mean", "raw_reward_benchmarks", "exact_target_expected"),
    "positive_rate": ("Positive fraction", "category_rates", "generated", "category_rates", "target", "positive"),
    "category_total_variation": ("Category TV error", None, None, None, None),
    "top_100_rate": ("Top-100 fraction", "top_k", "top_100", "generated_rate", "top_k", "top_100", "target_rate"),
    "exact_target_mass_covered": (
        "Target mass covered",
        None,
        None,
        "target_sampling_benchmarks",
        "expected_target_mass_covered",
    ),
    "unique_procedure_count": (
        "Unique procedures",
        None,
        None,
        "target_sampling_benchmarks",
        "expected_unique_procedure_count",
    ),
    "duplicate_fraction": (
        "Duplicate fraction",
        "concentration",
        "duplicate_fraction",
        "target_sampling_benchmarks",
        "expected_duplicate_fraction",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("SEED", "NO_REPLAY_JSON", "REPLAY_JSON"),
    )
    input_group.add_argument(
        "--triplet",
        action="append",
        nargs=4,
        metavar=("SEED", "NO_REPLAY_JSON", "POSITIVE_REPLAY_JSON", "MIXED_REPLAY_JSON"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def nested_value(data: dict, keys: tuple[str, ...]) -> float:
    value = data
    for key in keys:
        value = value[key]
    return float(value)


def metric_value(summary: dict, metric: str) -> float:
    if metric == "category_total_variation":
        generated = summary["category_rates"]["generated"]
        target = summary["category_rates"]["target"]
        return 0.5 * sum(abs(float(generated[key]) - float(target[key])) for key in target)
    if metric == "exact_target_mass_covered":
        return float(summary["exact_target_mass_covered_by_unique_samples"])
    if metric == "unique_procedure_count":
        return float(summary["unique_procedure_count"])

    spec = METRICS[metric]
    if metric == "mean_raw_reward":
        return nested_value(summary, (spec[1], spec[2]))
    if metric == "positive_rate":
        return nested_value(summary, (spec[1], spec[2], "positive"))
    if metric == "top_100_rate":
        return nested_value(summary, (spec[1], spec[2], spec[3]))
    if metric == "duplicate_fraction":
        return nested_value(summary, (spec[1], spec[2]))
    raise KeyError(metric)


def reference_value(summary: dict, metric: str) -> float:
    spec = METRICS[metric]
    if metric == "category_total_variation":
        return 0.0
    if metric == "mean_raw_reward":
        return nested_value(summary, (spec[3], spec[4]))
    if metric == "positive_rate":
        return nested_value(summary, (spec[3], spec[4], spec[5]))
    if metric == "top_100_rate":
        return nested_value(summary, (spec[4], spec[5], spec[6]))
    return nested_value(summary, tuple(key for key in spec[3:] if key is not None))


def sample_stats(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "mean": mean(values),
        "sample_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def write_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "method", "mean", "sample_standard_deviation", "reference", "n"],
        )
        writer.writeheader()
        for metric, methods in summary["by_metric"].items():
            for method in summary["methods"]:
                row = methods[method]
                writer.writerow(
                    {
                        "metric": metric,
                        "method": method,
                        "mean": row["mean"],
                        "sample_standard_deviation": row["sample_standard_deviation"],
                        "reference": methods["reference"],
                        "n": row["n"],
                    }
                )


def write_plot(summary: dict, path: Path) -> None:
    plot_metrics = [
        "mean_raw_reward",
        "positive_rate",
        "category_total_variation",
        "top_100_rate",
        "exact_target_mass_covered",
        "unique_procedure_count",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    colours = ["#2878b5", "#d95f02", "#2b8c4b"]
    labels = {
        "no_replay": "No replay",
        "positive_replay_8": "Positive replay 8",
        "mixed_replay_8": "Mixed replay 8",
    }
    methods = summary["methods"]
    for axis, metric in zip(axes.flat, plot_metrics):
        data = summary["by_metric"][metric]
        means = [data[method]["mean"] for method in methods]
        errors = [
            data[method]["sample_standard_deviation"]
            for method in methods
        ]
        positions = list(range(len(methods)))
        axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=5,
            color=colours[: len(methods)],
            alpha=0.82,
        )
        for position, method in zip(positions, methods):
            axis.scatter(
                [position] * len(data[method]["values"]),
                data[method]["values"],
                color="#222222",
                s=18,
                zorder=3,
            )
        axis.axhline(data["reference"], color="#222222", linestyle="--", linewidth=1.5, label="Reference")
        axis.set_xticks(positions, [labels[method] for method in methods], rotation=12)
        axis.set_title(METRICS[metric][0])
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("ChemGymRL procedure GFlowNet, mean and sample SD across three seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pairs = []
    methods = ["no_replay", "positive_replay_8"]
    rows = args.pair
    if args.triplet:
        methods.append("mixed_replay_8")
        rows = args.triplet
    for row in rows:
        seed, *paths = row
        pair = {"seed": int(seed)}
        for method, path in zip(methods, paths):
            with Path(path).open(encoding="utf-8") as handle:
                pair[method] = json.load(handle)
        pairs.append(pair)

    by_metric = {}
    for metric in METRICS:
        method_values = {
            method: [metric_value(pair[method], metric) for pair in pairs]
            for method in methods
        }
        metric_summary = {
            method: sample_stats(values) for method, values in method_values.items()
        }
        no_replay_values = method_values["no_replay"]
        for method in methods[1:]:
            differences = [
                treatment - baseline
                for baseline, treatment in zip(no_replay_values, method_values[method])
            ]
            metric_summary[f"paired_difference_{method}_minus_no_replay"] = sample_stats(
                differences
            )
        metric_summary["reference"] = reference_value(pairs[0]["no_replay"], metric)
        by_metric[metric] = metric_summary

    summary = {
        "seeds": [pair["seed"] for pair in pairs],
        "pair_count": len(pairs),
        "methods": methods,
        "standard_deviation_definition": "sample standard deviation across independent seeds (n - 1 denominator)",
        "by_metric": by_metric,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_seed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    write_csv(summary, args.output_dir / "paired_seed_summary.csv")
    write_plot(summary, args.output_dir / "paired_seed_summary.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
