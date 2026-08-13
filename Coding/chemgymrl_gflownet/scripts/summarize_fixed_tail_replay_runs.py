#!/usr/bin/env python3
"""Summarise matched upper-tail target runs with and without fixed-tail replay."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("no_replay", "fixed_tail_replay")
METHOD_LABELS = {
    "no_replay": "No replay",
    "fixed_tail_replay": "Fixed-tail replay",
}
PLOT_METRICS = {
    "mean_raw_reward": "Mean raw reward",
    "tail_fraction": "Defined-tail fraction",
    "unique_tail_count": "Unique tail procedures",
    "top_100_fraction": "Top-100 fraction",
    "unique_top_100": "Unique top-100 procedures",
    "negative_fraction": "Negative fraction",
    "target_mass_covered": "Target mass covered",
    "duplicate_fraction": "Duplicate fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=6,
        action="append",
        required=True,
        metavar=(
            "SEED",
            "NO_REPLAY_JSON",
            "NO_REPLAY_TRAIN_DB",
            "REPLAY_JSON",
            "REPLAY_TRAIN_DB",
            "REPLAY_TRAIN_LOG",
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf8") as input_file:
        return json.load(input_file)


def final_metrics(summary: dict) -> dict[str, float]:
    generated = summary["category_rates"]["generated"]
    target = summary["category_rates"]["target"]
    tail = summary["defined_tail"]
    return {
        "mean_raw_reward": float(summary["raw_reward"]["mean"]),
        "tail_fraction": float(tail["generated_fraction"]),
        "unique_tail_count": float(tail["generated_unique_count"]),
        "top_100_fraction": float(summary["top_k"]["top_100"]["generated_rate"]),
        "unique_top_100": float(
            summary["top_k"]["top_100"]["generated_unique_count"]
        ),
        "negative_fraction": float(generated["negative"]),
        "category_tv_error": 0.5
        * sum(abs(float(generated[key]) - float(target[key])) for key in target),
        "target_mass_covered": float(
            summary["exact_target_mass_covered_by_unique_samples"]
        ),
        "unique_procedure_count": float(summary["unique_procedure_count"]),
        "duplicate_fraction": float(summary["concentration"]["duplicate_fraction"]),
    }


def reference_metrics(summary: dict) -> dict[str, float]:
    tail = summary["defined_tail"]
    return {
        "mean_raw_reward": float(
            summary["raw_reward_benchmarks"]["exact_target_expected"]
        ),
        "tail_fraction": float(tail["target_probability"]),
        "unique_tail_count": float(tail["target_expected_unique_count"]),
        "top_100_fraction": float(summary["top_k"]["top_100"]["target_rate"]),
        "unique_top_100": float(
            summary["top_k"]["top_100"]["target_expected_unique_count"]
        ),
        "negative_fraction": float(summary["category_rates"]["target"]["negative"]),
        "category_tv_error": 0.0,
        "target_mass_covered": float(
            summary["target_sampling_benchmarks"]["expected_target_mass_covered"]
        ),
        "unique_procedure_count": float(
            summary["target_sampling_benchmarks"]["expected_unique_procedure_count"]
        ),
        "duplicate_fraction": float(
            summary["target_sampling_benchmarks"]["expected_duplicate_fraction"]
        ),
    }


def training_metrics(path: Path, threshold: float) -> dict[str, float | int]:
    with sqlite3.connect(path) as connection:
        count, hits, unique_hits = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(fr_1 > ?),
                COUNT(DISTINCT CASE WHEN fr_1 > ? THEN smi END)
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
        "unique_tail_count": int(unique_hits),
    }


def replay_diagnostics(path: Path) -> dict[str, float | int | bool]:
    rows = []
    pattern = re.compile(r"([a-z_]+):([-+0-9.]+)")
    iteration_pattern = re.compile(r" - iteration (\d+) ")
    with path.open(encoding="utf8") as input_file:
        for line in input_file:
            iteration_match = iteration_pattern.search(line)
            if iteration_match is None:
                continue
            values = {key: float(value) for key, value in pattern.findall(line)}
            if "replay_active" in values:
                values["iteration"] = float(iteration_match.group(1))
                rows.append(values)
    if not rows:
        raise ValueError(f"training log has no replay diagnostics: {path}")
    active_rows = [row for row in rows if row["replay_active"] == 1]
    return {
        "logged_training_rows": len(rows),
        "logged_active_rows": len(active_rows),
        "first_logged_active_iteration": int(active_rows[0]["iteration"]),
        "all_logged_active_batches_are_4_plus_4": all(
            row["replay_tail_sample_count"] == 4
            and row["replay_non_tail_sample_count"] == 4
            for row in active_rows
        ),
        "final_tail_pool_size": int(rows[-1]["replay_tail_pool_size"]),
        "final_non_tail_pool_size": int(rows[-1]["replay_non_tail_pool_size"]),
        "maximum_invalid_trajectories": max(
            row["invalid_trajectories"] for row in rows
        ),
        "maximum_invalid_logprob": max(row["invalid_logprob"] for row in rows),
        "maximum_invalid_losses": max(row["invalid_losses"] for row in rows),
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
                writer.writerow(
                    {
                        "metric": metric,
                        "method": method,
                        "mean": methods[method]["mean"],
                        "sample_standard_deviation": methods[method][
                            "sample_standard_deviation"
                        ],
                        "reference": methods["reference"],
                        "n": methods[method]["n"],
                    }
                )


def plot_metrics(summary: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    colours = ("#2878B5", "#D95F02")
    for axis, (metric, label) in zip(axes.flat, PLOT_METRICS.items()):
        data = summary["final_metrics"][metric]
        positions = (0, 1)
        values = [data[method]["mean"] for method in METHODS]
        errors = [data[method]["sample_standard_deviation"] for method in METHODS]
        axis.bar(positions, values, yerr=errors, capsize=5, color=colours, alpha=0.84)
        for position, method in zip(positions, METHODS):
            axis.scatter(
                [position] * len(data[method]["values"]),
                data[method]["values"],
                color="#222222",
                s=20,
                zorder=3,
            )
        axis.axhline(
            data["reference"],
            color="#222222",
            linestyle="--",
            linewidth=1.4,
        )
        axis.set_xticks(
            positions,
            [METHOD_LABELS[method] for method in METHODS],
            rotation=10,
        )
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Wang-inspired upper-tail target with fixed-tail replay\n"
        "Mean and sample SD across seeds 0, 17 and 42; dashed line is exact target"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pairs = []
    for (
        seed,
        no_replay_json,
        no_replay_train_db,
        replay_json,
        replay_train_db,
        replay_train_log,
    ) in args.pair:
        no_replay = load_json(Path(no_replay_json))
        replay = load_json(Path(replay_json))
        thresholds = {
            float(no_replay["defined_tail"]["threshold"]),
            float(replay["defined_tail"]["threshold"]),
        }
        if len(thresholds) != 1:
            raise ValueError(f"seed {seed} uses mismatched tail thresholds")
        if no_replay["terminal_target"] != replay["terminal_target"]:
            raise ValueError(f"seed {seed} uses mismatched terminal targets")
        threshold = thresholds.pop()
        pairs.append(
            {
                "seed": int(seed),
                "no_replay": no_replay,
                "fixed_tail_replay": replay,
                "no_replay_training": training_metrics(
                    Path(no_replay_train_db), threshold
                ),
                "fixed_tail_replay_training": training_metrics(
                    Path(replay_train_db), threshold
                ),
                "replay_diagnostics": replay_diagnostics(Path(replay_train_log)),
            }
        )

    final = {}
    metric_names = final_metrics(pairs[0]["no_replay"])
    for metric in metric_names:
        values = {
            method: [final_metrics(pair[method])[metric] for pair in pairs]
            for method in METHODS
        }
        final[metric] = {method: stats(method_values) for method, method_values in values.items()}
        final[metric]["paired_difference_replay_minus_no_replay"] = stats(
            [
                replay_value - no_replay_value
                for no_replay_value, replay_value in zip(
                    values["no_replay"], values["fixed_tail_replay"]
                )
            ]
        )
        final[metric]["reference"] = reference_metrics(pairs[0]["no_replay"])[
            metric
        ]

    exposure = {}
    for metric in ("tail_hit_count", "tail_hit_fraction", "unique_tail_count"):
        values = {
            method: [pair[f"{method}_training"][metric] for pair in pairs]
            for method in METHODS
        }
        exposure[metric] = {
            method: stats(method_values) for method, method_values in values.items()
        }
        exposure[metric]["paired_difference_replay_minus_no_replay"] = stats(
            [
                replay_value - no_replay_value
                for no_replay_value, replay_value in zip(
                    values["no_replay"], values["fixed_tail_replay"]
                )
            ]
        )

    summary = {
        "seeds": [pair["seed"] for pair in pairs],
        "pair_count": len(pairs),
        "tail_definition": "raw ChemGymRL reward > 0.10",
        "terminal_target": pairs[0]["no_replay"]["terminal_target"],
        "new_simulator_evaluations_per_run": pairs[0]["no_replay_training"][
            "sample_count"
        ],
        "replayed_trajectories_per_active_update": 8,
        "final_samples_per_run": pairs[0]["no_replay"]["sample_count"],
        "standard_deviation_definition": (
            "sample standard deviation across independent seeds (n - 1 denominator)"
        ),
        "final_metrics": final,
        "training_exposure": exposure,
        "replay_diagnostics": {
            str(pair["seed"]): pair["replay_diagnostics"] for pair in pairs
        },
        "per_seed": pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fixed_tail_replay_summary.json").open(
        "w", encoding="utf8"
    ) as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    write_csv(summary, args.output_dir / "fixed_tail_replay_summary.csv")
    plot_metrics(summary, args.output_dir / "fixed_tail_replay_summary.png")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "per_seed"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
