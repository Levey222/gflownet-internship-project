#!/usr/bin/env python3
"""Summarise paired baseline and upper-tail GFlowNet runs across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("baseline", "tail_shaped")
METHOD_LABELS = {"baseline": "Baseline target", "tail_shaped": "Tail-shaped target"}
METRICS = {
    "mean_raw_reward": "Mean raw reward",
    "tail_fraction": "Defined-tail fraction",
    "unique_tail_count": "Unique defined-tail procedures",
    "top_100_fraction": "Top-100 fraction",
    "unique_top_100": "Unique top-100 procedures",
    "duplicate_fraction": "Duplicate fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=5,
        action="append",
        required=True,
        metavar=(
            "SEED",
            "BASELINE_JSON",
            "BASELINE_TRAIN_DB",
            "TAIL_JSON",
            "TAIL_TRAIN_DB",
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf8") as input_file:
        return json.load(input_file)


def training_tail_metrics(path: Path, threshold: float) -> dict[str, float | int]:
    with sqlite3.connect(path) as connection:
        count, hits, unique_count, maximum, average = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(fr_1 > ?),
                COUNT(DISTINCT CASE WHEN fr_1 > ? THEN smi END),
                MAX(fr_1),
                AVG(fr_1)
            FROM results
            """,
            (threshold, threshold),
        ).fetchone()
    if not count:
        raise ValueError(f"training database is empty: {path}")
    return {
        "sample_count": int(count),
        "tail_hit_count": int(hits),
        "tail_hit_fraction": float(hits / count),
        "unique_tail_count": int(unique_count),
        "maximum_raw_reward": float(maximum),
        "mean_raw_reward": float(average),
    }


def final_metrics(summary: dict) -> dict[str, float]:
    tail = summary["defined_tail"]
    if tail is None:
        raise ValueError("summary does not contain a fixed-tail evaluation")
    return {
        "mean_raw_reward": float(summary["raw_reward"]["mean"]),
        "tail_fraction": float(tail["generated_fraction"]),
        "unique_tail_count": float(tail["generated_unique_count"]),
        "top_100_fraction": float(summary["top_k"]["top_100"]["generated_rate"]),
        "unique_top_100": float(summary["top_k"]["top_100"]["generated_unique_count"]),
        "duplicate_fraction": float(summary["concentration"]["duplicate_fraction"]),
        "unique_procedure_count": float(summary["unique_procedure_count"]),
    }


def reference_metrics(summary: dict) -> dict[str, float]:
    tail = summary["defined_tail"]
    return {
        "mean_raw_reward": float(summary["raw_reward_benchmarks"]["exact_target_expected"]),
        "tail_fraction": float(tail["target_probability"]),
        "unique_tail_count": float(tail["target_expected_unique_count"]),
        "top_100_fraction": float(summary["top_k"]["top_100"]["target_rate"]),
        "unique_top_100": float(
            summary["top_k"]["top_100"]["target_expected_unique_count"]
        ),
        "duplicate_fraction": float(
            summary["target_sampling_benchmarks"]["expected_duplicate_fraction"]
        ),
        "unique_procedure_count": float(
            summary["target_sampling_benchmarks"]["expected_unique_procedure_count"]
        ),
    }


def stats(values: list[float]) -> dict[str, float | int | list[float]]:
    return {
        "values": values,
        "mean": mean(values),
        "sample_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def write_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "metric",
                "method",
                "mean",
                "sample_standard_deviation",
                "reference",
                "n",
            ],
        )
        writer.writeheader()
        for metric, methods in summary["final_metrics"].items():
            for method in METHODS:
                row = methods[method]
                writer.writerow(
                    {
                        "metric": metric,
                        "method": method,
                        "mean": row["mean"],
                        "sample_standard_deviation": row["sample_standard_deviation"],
                        "reference": methods["reference"][method],
                        "n": row["n"],
                    }
                )


def plot_final_metrics(summary: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    colours = ("#2878B5", "#0B6E4F")
    for axis, (metric, label) in zip(axes.flat, METRICS.items()):
        data = summary["final_metrics"][metric]
        positions = (0, 1)
        means = [data[method]["mean"] for method in METHODS]
        errors = [data[method]["sample_standard_deviation"] for method in METHODS]
        axis.bar(positions, means, yerr=errors, capsize=5, color=colours, alpha=0.84)
        for position, method, colour in zip(positions, METHODS, colours):
            values = data[method]["values"]
            axis.scatter([position] * len(values), values, color="#222222", s=20, zorder=3)
            reference = data["reference"][method]
            axis.plot(
                [position - 0.28, position + 0.28],
                [reference, reference],
                color=colour,
                linestyle="--",
                linewidth=2,
            )
        axis.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS], rotation=8)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "ChemGymRL upper-tail target experiment, mean and sample SD across three seeds\n"
        "Coloured dashed markers show each method's exact target expectation"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_training_exposure(summary: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    colours = ("#2878B5", "#0B6E4F")
    for axis, field, label in (
        (axes[0], "tail_hit_count", "Tail trajectories encountered"),
        (axes[1], "unique_tail_count", "Unique tail procedures encountered"),
    ):
        values_by_method = summary["training_exposure"][field]
        means = [values_by_method[method]["mean"] for method in METHODS]
        errors = [
            values_by_method[method]["sample_standard_deviation"] for method in METHODS
        ]
        axis.bar((0, 1), means, yerr=errors, capsize=5, color=colours, alpha=0.84)
        for position, method in enumerate(METHODS):
            values = values_by_method[method]["values"]
            axis.scatter([position] * len(values), values, color="#222222", s=20, zorder=3)
        axis.set_xticks((0, 1), [METHOD_LABELS[method] for method in METHODS], rotation=8)
        axis.set(ylabel="Count in 32,000 training samples", title=label)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Observed training exposure to the fixed raw-reward > 0.10 tail")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pairs = []
    for seed, baseline_json, baseline_db, tail_json, tail_db in args.pair:
        baseline = load_json(Path(baseline_json))
        tail_shaped = load_json(Path(tail_json))
        thresholds = {
            float(baseline["defined_tail"]["threshold"]),
            float(tail_shaped["defined_tail"]["threshold"]),
        }
        if len(thresholds) != 1:
            raise ValueError(f"seed {seed} uses mismatched tail thresholds")
        threshold = thresholds.pop()
        pairs.append(
            {
                "seed": int(seed),
                "baseline": baseline,
                "tail_shaped": tail_shaped,
                "baseline_training": training_tail_metrics(Path(baseline_db), threshold),
                "tail_shaped_training": training_tail_metrics(Path(tail_db), threshold),
            }
        )

    final = {}
    for metric in (*METRICS, "unique_procedure_count"):
        method_values = {
            method: [final_metrics(pair[method])[metric] for pair in pairs]
            for method in METHODS
        }
        final[metric] = {
            method: stats(values) for method, values in method_values.items()
        }
        final[metric]["paired_difference_tail_minus_baseline"] = stats(
            [
                tail_value - baseline_value
                for baseline_value, tail_value in zip(
                    method_values["baseline"], method_values["tail_shaped"]
                )
            ]
        )
        final[metric]["reference"] = {
            method: reference_metrics(pairs[0][method])[metric] for method in METHODS
        }

    exposure = {}
    for field in ("tail_hit_count", "tail_hit_fraction", "unique_tail_count"):
        method_values = {
            method: [pair[f"{method}_training"][field] for pair in pairs]
            for method in METHODS
        }
        exposure[field] = {method: stats(values) for method, values in method_values.items()}
        exposure[field]["paired_difference_tail_minus_baseline"] = stats(
            [
                tail_value - baseline_value
                for baseline_value, tail_value in zip(
                    method_values["baseline"], method_values["tail_shaped"]
                )
            ]
        )

    summary = {
        "seeds": [pair["seed"] for pair in pairs],
        "pair_count": len(pairs),
        "tail_definition": "raw ChemGymRL reward > 0.10",
        "training_samples_per_run": pairs[0]["baseline_training"]["sample_count"],
        "final_samples_per_run": pairs[0]["baseline"]["sample_count"],
        "standard_deviation_definition": (
            "sample standard deviation across independent seeds (n - 1 denominator)"
        ),
        "final_metrics": final,
        "training_exposure": exposure,
        "per_seed": pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "upper_tail_summary.json").open("w", encoding="utf8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    write_csv(summary, args.output_dir / "upper_tail_summary.csv")
    plot_final_metrics(summary, args.output_dir / "upper_tail_final_comparison.png")
    plot_training_exposure(summary, args.output_dir / "upper_tail_training_exposure.png")
    print(json.dumps({key: summary[key] for key in summary if key != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
