#!/usr/bin/env python3
"""Compare generated ChemGymRL procedures with an exhaustive reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--generated-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    args = parser.parse_args()
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    return args


def parse_reference(path: Path) -> tuple[list[tuple[int, ...]], np.ndarray]:
    with path.open(newline="", encoding="utf8") as input_file:
        rows = list(csv.DictReader(input_file))
    procedures = [tuple(int(token) for token in row["tokens"].split()) for row in rows]
    raw_rewards = np.asarray([float(row["raw_reward"]) for row in rows], dtype=float)
    if not rows or len(set(procedures)) != len(procedures):
        raise ValueError("the exhaustive reference is empty or contains duplicate procedures")
    if not np.isfinite(raw_rewards).all():
        raise ValueError("the exhaustive reference contains a non-finite reward")
    return procedures, raw_rewards


def parse_generated(path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(results)").fetchall()
        }
        required = {"smi", "fr_0", "fr_1"}
        if not required.issubset(columns):
            raise ValueError(f"generated database is missing columns: {sorted(required - columns)}")
        rows = connection.execute("SELECT smi, fr_0, fr_1 FROM results").fetchall()
    if not rows:
        raise ValueError("the generated database contains no procedures")
    parsed = []
    for encoded, positive_reward, raw_reward in rows:
        procedure = tuple(int(token) for token in json.loads(encoded)["tokens"])
        parsed.append(
            {
                "tokens": procedure,
                "positive_reward": float(positive_reward),
                "raw_reward": float(raw_reward),
            }
        )
    return parsed


def category_rates(raw_rewards: np.ndarray) -> dict[str, float]:
    return {
        "negative": float(np.mean(raw_rewards < 0)),
        "zero": float(np.mean(raw_rewards == 0)),
        "positive": float(np.mean(raw_rewards > 0)),
    }


def top_k_metrics(
    ranks: np.ndarray,
    target_probabilities: np.ndarray,
    rank_order: np.ndarray,
    state_count: int,
) -> dict[str, dict[str, float | int]]:
    metrics = {}
    for k in (10, 100, 1000):
        limit = min(k, state_count)
        count = int(np.sum(ranks <= limit))
        metrics[f"top_{k}"] = {
            "generated_count": count,
            "generated_rate": float(count / len(ranks)),
            "uniform_rate": float(limit / state_count),
            "target_rate": float(target_probabilities[rank_order[:limit]].sum()),
        }
    return metrics


def write_generated_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["tokens", "raw_reward", "positive_reward", "exact_rank"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    reference_rewards: np.ndarray,
    generated_rewards: np.ndarray,
    rates: dict[str, dict[str, float]],
    top_k: dict[str, dict[str, float | int]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    bins = np.linspace(reference_rewards.min(), reference_rewards.max(), 35)
    axes[0].hist(reference_rewards, bins=bins, density=True, alpha=0.55, label="Uniform reference")
    axes[0].hist(generated_rewards, bins=bins, density=True, alpha=0.55, label="Generated model")
    axes[0].set(xlabel="Raw reward", ylabel="Density", title="Reward distribution")
    axes[0].legend(frameon=False)

    categories = ["negative", "zero", "positive"]
    x = np.arange(len(categories))
    width = 0.25
    for offset, (label, colour) in enumerate(
        [("Uniform", "#648FFF"), ("Generated", "#DC267F"), ("Exact target", "#FFB000")]
    ):
        key = label.lower().replace("exact ", "")
        values = [rates[key][category] for category in categories]
        axes[1].bar(x + (offset - 1) * width, values, width, label=label, color=colour)
    axes[1].set(
        xticks=x,
        xticklabels=[category.title() for category in categories],
        ylabel="Fraction",
        title="Reward categories",
    )
    axes[1].legend(frameon=False)

    labels = list(top_k)
    x = np.arange(len(labels))
    for offset, (field, label, colour) in enumerate(
        [
            ("uniform_rate", "Uniform", "#648FFF"),
            ("generated_rate", "Generated", "#DC267F"),
            ("target_rate", "Exact target", "#FFB000"),
        ]
    ):
        values = [float(top_k[name][field]) for name in labels]
        axes[2].bar(x + (offset - 1) * width, values, width, label=label, color=colour)
    axes[2].set(
        xticks=x,
        xticklabels=[label.replace("_", "-") for label in labels],
        ylabel="Fraction",
        title="Top-ranked procedure hits",
    )
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("ChemGymRL procedure GFlowNet reference comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    reference_path = args.reference_csv.resolve()
    generated_path = args.generated_db.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    procedures, reference_rewards = parse_reference(reference_path)
    generated = parse_generated(generated_path)
    reward_by_procedure = dict(zip(procedures, reference_rewards))
    index_by_procedure = {procedure: index for index, procedure in enumerate(procedures)}
    rank_order = np.argsort(-reference_rewards)
    rank_by_procedure = {
        procedures[index]: rank for rank, index in enumerate(rank_order, start=1)
    }

    missing = sorted({row["tokens"] for row in generated} - set(reward_by_procedure))
    if missing:
        raise ValueError(f"{len(missing)} generated procedures are absent from the reference")

    generated_rewards = np.asarray([row["raw_reward"] for row in generated], dtype=float)
    generated_positive = np.asarray([row["positive_reward"] for row in generated], dtype=float)
    expected_positive = np.exp(np.clip(generated_rewards / args.reward_scale, -20.0, 20.0))
    exact_rewards = np.asarray(
        [reward_by_procedure[row["tokens"]] for row in generated],
        dtype=float,
    )
    ranks = np.asarray([rank_by_procedure[row["tokens"]] for row in generated], dtype=int)

    log_weights = np.clip(reference_rewards / args.reward_scale, -20.0, 20.0)
    weights = np.exp(log_weights - log_weights.max())
    target_probabilities = weights / weights.sum()
    exact_log_partition = float(log_weights.max() + np.log(weights.sum()))
    uniform_rates = category_rates(reference_rewards)
    generated_rates = category_rates(generated_rewards)
    target_rates = {
        "negative": float(target_probabilities[reference_rewards < 0].sum()),
        "zero": float(target_probabilities[reference_rewards == 0].sum()),
        "positive": float(target_probabilities[reference_rewards > 0].sum()),
    }
    top_k = top_k_metrics(
        ranks,
        target_probabilities,
        rank_order,
        len(reference_rewards),
    )
    target_inclusion_probabilities = -np.expm1(
        len(generated) * np.log1p(-target_probabilities)
    )
    for k in (10, 100, 1000):
        limit = min(k, len(reference_rewards))
        unique_count = len(
            {
                row["tokens"]
                for row, rank in zip(generated, ranks)
                if rank <= limit
            }
        )
        top_k[f"top_{k}"]["generated_unique_count"] = unique_count
        top_k[f"top_{k}"]["reference_coverage"] = float(unique_count / limit)
        top_k[f"top_{k}"]["target_expected_unique_count"] = float(
            target_inclusion_probabilities[rank_order[:limit]].sum()
        )

    best_index = int(np.argmax(generated_rewards))
    uniform_mean = float(reference_rewards.mean())
    target_mean = float(np.sum(target_probabilities * reference_rewards))
    generated_mean = float(generated_rewards.mean())
    target_gap = target_mean - uniform_mean
    progress = (generated_mean - uniform_mean) / target_gap if target_gap else float("nan")
    unique_procedures = {row["tokens"] for row in generated}
    procedure_counts = Counter(row["tokens"] for row in generated)
    most_repeated_count = max(procedure_counts.values())

    output_rows = []
    for row, exact_reward, rank in zip(generated, exact_rewards, ranks):
        output_rows.append(
            {
                "tokens": " ".join(str(token) for token in row["tokens"]),
                "raw_reward": row["raw_reward"],
                "positive_reward": row["positive_reward"],
                "exact_rank": int(rank),
            }
        )
    write_generated_csv(output_rows, output_dir / "generated_procedures.csv")

    summary = {
        "reference_csv": str(reference_path),
        "generated_db": str(generated_path),
        "reward_scale": args.reward_scale,
        "sample_count": len(generated),
        "unique_procedure_count": len(unique_procedures),
        "duplicate_count": len(generated) - len(unique_procedures),
        "concentration": {
            "duplicate_fraction": float(1 - len(unique_procedures) / len(generated)),
            "most_repeated_count": most_repeated_count,
            "most_repeated_fraction": float(most_repeated_count / len(generated)),
        },
        "valid_reference_match_count": len(generated) - len(missing),
        "reward_consistency": {
            "maximum_raw_reward_error": float(np.max(np.abs(generated_rewards - exact_rewards))),
            "maximum_positive_reward_error": float(
                np.max(np.abs(generated_positive - expected_positive))
            ),
        },
        "raw_reward": {
            "minimum": float(generated_rewards.min()),
            "mean": generated_mean,
            "median": float(np.median(generated_rewards)),
            "maximum": float(generated_rewards.max()),
            "standard_deviation": float(generated_rewards.std()),
        },
        "positive_reward_mean": float(generated_positive.mean()),
        "category_rates": {
            "uniform": uniform_rates,
            "generated": generated_rates,
            "target": target_rates,
        },
        "raw_reward_benchmarks": {
            "uniform_expected": uniform_mean,
            "generated": generated_mean,
            "exact_target_expected": target_mean,
            "generated_minus_uniform": generated_mean - uniform_mean,
            "fraction_of_uniform_to_target_gap": progress,
        },
        "exact_log_partition": exact_log_partition,
        "rank": {
            "best": int(ranks.min()),
            "median": float(np.median(ranks)),
            "mean": float(ranks.mean()),
        },
        "top_k": top_k,
        "exact_target_mass_covered_by_unique_samples": float(
            sum(target_probabilities[index_by_procedure[procedure]] for procedure in unique_procedures)
        ),
        "target_sampling_benchmarks": {
            "expected_unique_procedure_count": float(target_inclusion_probabilities.sum()),
            "expected_duplicate_fraction": float(
                1 - target_inclusion_probabilities.sum() / len(generated)
            ),
            "expected_target_mass_covered": float(
                np.sum(target_probabilities * target_inclusion_probabilities)
            ),
        },
        "best_generated_procedure": {
            "tokens": " ".join(str(token) for token in generated[best_index]["tokens"]),
            "raw_reward": float(generated_rewards[best_index]),
            "exact_rank": int(ranks[best_index]),
        },
    }

    with (output_dir / "analysis_summary.json").open("w", encoding="utf8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    plot_comparison(
        reference_rewards,
        generated_rewards,
        summary["category_rates"],
        top_k,
        output_dir / "reference_comparison.png",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
