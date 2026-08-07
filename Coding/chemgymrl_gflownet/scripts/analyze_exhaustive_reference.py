#!/usr/bin/env python3
"""Validate and analyse an exhaustive ChemGymRL procedure reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACTION_DIMENSION = 9
NUM_TOKEN_VALUES = 3
SEARCH_SPACE_SIZE = NUM_TOKEN_VALUES**ACTION_DIMENSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedures-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.5],
    )
    parser.add_argument("--selected-scale", type=float, default=0.05)
    parser.add_argument("--stored-scale", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.scales):
        parser.error("all scales must be positive")
    if args.selected_scale <= 0:
        parser.error("--selected-scale must be positive")
    if args.stored_scale <= 0:
        parser.error("--stored-scale must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("the procedures CSV is empty")
    return rows


def parse_tokens(value: str) -> tuple[int, ...]:
    return tuple(int(token) for token in value.split())


def validate_reference(rows: list[dict[str, str]]) -> dict[str, int]:
    tokens = [parse_tokens(row["tokens"]) for row in rows]
    if len(rows) != SEARCH_SPACE_SIZE:
        raise ValueError(f"expected {SEARCH_SPACE_SIZE} rows, found {len(rows)}")
    if any(len(procedure) != ACTION_DIMENSION for procedure in tokens):
        raise ValueError("every procedure must contain exactly nine tokens")
    if any(token < 0 or token >= NUM_TOKEN_VALUES for procedure in tokens for token in procedure):
        raise ValueError("procedure tokens must be in the range 0 to 2")
    unique_count = len(set(tokens))
    if unique_count != SEARCH_SPACE_SIZE:
        raise ValueError(f"expected {SEARCH_SPACE_SIZE} unique procedures, found {unique_count}")
    return {
        "row_count": len(rows),
        "unique_procedure_count": unique_count,
        "procedure_length": ACTION_DIMENSION,
        "token_value_count": NUM_TOKEN_VALUES,
    }


def count_for_mass(sorted_cumulative_probability: np.ndarray, target: float) -> int:
    return int(np.searchsorted(sorted_cumulative_probability, target) + 1)


def distribution_metrics(raw_rewards: np.ndarray, scale: float) -> dict[str, float | int]:
    unclipped_log_rewards = raw_rewards / scale
    log_rewards = np.clip(unclipped_log_rewards, -20.0, 20.0)
    unnormalised = np.exp(log_rewards - log_rewards.max())
    probabilities = unnormalised / unnormalised.sum()
    order = np.argsort(raw_rewards)[::-1]
    sorted_probabilities = probabilities[order]
    cumulative = np.cumsum(sorted_probabilities)
    entropy = -np.sum(probabilities * np.log(probabilities))
    top_10 = min(10, len(probabilities))
    top_100 = min(100, len(probabilities))
    return {
        "scale": scale,
        "minimum_positive_reward": float(np.exp(log_rewards.min())),
        "maximum_positive_reward": float(np.exp(log_rewards.max())),
        "top_1_probability": float(sorted_probabilities[0]),
        "top_10_probability": float(sorted_probabilities[:top_10].sum()),
        "top_100_probability": float(sorted_probabilities[:top_100].sum()),
        "positive_procedure_probability": float(probabilities[raw_rewards > 0].sum()),
        "zero_procedure_probability": float(probabilities[raw_rewards == 0].sum()),
        "negative_procedure_probability": float(probabilities[raw_rewards < 0].sum()),
        "expected_raw_reward": float(np.sum(probabilities * raw_rewards)),
        "effective_sample_size": float(1.0 / np.sum(probabilities**2)),
        "entropy_perplexity": float(np.exp(entropy)),
        "procedures_for_50_percent_mass": count_for_mass(cumulative, 0.5),
        "procedures_for_90_percent_mass": count_for_mass(cumulative, 0.9),
        "procedures_for_99_percent_mass": count_for_mass(cumulative, 0.99),
        "lower_clip_count": int((unclipped_log_rewards < -20.0).sum()),
        "upper_clip_count": int((unclipped_log_rewards > 20.0).sum()),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scale_metrics(
    metrics: list[dict[str, float | int]],
    selected_scale: float,
    path: Path,
) -> None:
    scales = np.asarray([row["scale"] for row in metrics], dtype=float)
    effective_sample_size = np.asarray([row["effective_sample_size"] for row in metrics], dtype=float)
    positive_mass = np.asarray([row["positive_procedure_probability"] for row in metrics], dtype=float)
    zero_mass = np.asarray([row["zero_procedure_probability"] for row in metrics], dtype=float)
    top_100_mass = np.asarray([row["top_100_probability"] for row in metrics], dtype=float)

    figure, (mass_axis, ess_axis) = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
    mass_axis.plot(scales, positive_mass, marker="o", label="Positive-procedure mass")
    mass_axis.plot(scales, zero_mass, marker="o", label="Zero-procedure mass")
    mass_axis.plot(scales, top_100_mass, marker="o", label="Top-100 mass")
    mass_axis.axvline(selected_scale, color="#555555", linestyle="--", label=f"Selected scale {selected_scale:g}")
    mass_axis.set_ylabel("Probability mass")
    mass_axis.grid(alpha=0.25)
    mass_axis.legend(frameon=False)

    ess_axis.plot(scales, effective_sample_size, color="#9C2F2F", marker="o")
    ess_axis.axvline(selected_scale, color="#555555", linestyle="--")
    ess_axis.set(xlabel="Reward scale", ylabel="Effective sample size")
    ess_axis.grid(alpha=0.25)
    figure.suptitle("ChemGymRL terminal-reward scale comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.procedures_csv.resolve())
    validation = validate_reference(rows)
    raw_rewards = np.asarray([float(row["raw_reward"]) for row in rows], dtype=float)
    if not np.isfinite(raw_rewards).all():
        raise ValueError("raw rewards must all be finite")

    scales = sorted(set(args.scales + [args.selected_scale]))
    metrics = [distribution_metrics(raw_rewards, scale) for scale in scales]
    selected = next(row for row in metrics if row["scale"] == args.selected_scale)
    ranked = sorted(rows, key=lambda row: float(row["raw_reward"]), reverse=True)
    top_rows = ranked[: min(args.top_k, len(ranked))]

    stored_positive = np.asarray([float(row["positive_reward"]) for row in rows], dtype=float)
    stored_scale = args.stored_scale
    expected_stored_positive = np.exp(np.clip(raw_rewards / stored_scale, -20.0, 20.0))
    maximum_transform_error = float(np.max(np.abs(stored_positive - expected_stored_positive)))

    best_procedure = dict(ranked[0])
    best_procedure["positive_reward_at_selected_scale"] = selected["maximum_positive_reward"]
    summary = {
        "source": str(args.procedures_csv.resolve()),
        "validation": validation,
        "raw_reward": {
            "minimum": float(raw_rewards.min()),
            "mean": float(raw_rewards.mean()),
            "standard_deviation": float(raw_rewards.std()),
            "median": float(np.median(raw_rewards)),
            "maximum": float(raw_rewards.max()),
            "negative_count": int((raw_rewards < 0).sum()),
            "zero_count": int((raw_rewards == 0).sum()),
            "positive_count": int((raw_rewards > 0).sum()),
            "unique_value_count": int(len(np.unique(raw_rewards))),
        },
        "stored_positive_reward_check": {
            "stored_scale": stored_scale,
            "maximum_absolute_error": maximum_transform_error,
        },
        "selected_scale": args.selected_scale,
        "selected_scale_metrics": selected,
        "best_procedure": best_procedure,
    }

    write_csv(metrics, output_dir / "reward_scale_metrics.csv")
    write_csv(top_rows, output_dir / f"top_{len(top_rows)}_procedures.csv")
    plot_scale_metrics(metrics, args.selected_scale, output_dir / "reward_scale_comparison.png")
    with (output_dir / "analysis_summary.json").open("w", encoding="utf8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
