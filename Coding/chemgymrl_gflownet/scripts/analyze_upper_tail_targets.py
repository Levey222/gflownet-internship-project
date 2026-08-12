#!/usr/bin/env python3
"""Analyse exact Wang-inspired terminal targets before GFlowNet training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from chemgymrl_gflownet.tail_reward import terminal_log_rewards  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--tail-threshold", type=float, default=0.1)
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--selected-strength", type=float, default=0.25)
    parser.add_argument("--sample-count", type=int, default=1024)
    args = parser.parse_args()
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if not np.isfinite(args.tail_threshold):
        parser.error("--tail-threshold must be finite")
    if any(not np.isfinite(value) or value < 0 for value in args.strengths):
        parser.error("all --strengths must be finite and non-negative")
    if not any(np.isclose(args.selected_strength, value) for value in args.strengths):
        parser.error("--selected-strength must appear in --strengths")
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    return args


def read_reference(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("the exhaustive reference is empty")
    raw_rewards = np.asarray([float(row["raw_reward"]) for row in rows], dtype=float)
    if not np.isfinite(raw_rewards).all():
        raise ValueError("the exhaustive reference contains non-finite rewards")
    return raw_rewards


def probabilities_from_log_rewards(log_rewards: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = float(log_rewards.max())
    weights = np.exp(log_rewards - maximum)
    partition = float(maximum + np.log(weights.sum()))
    return weights / weights.sum(), partition


def expected_inclusion(probabilities: np.ndarray, sample_count: int) -> np.ndarray:
    return -np.expm1(sample_count * np.log1p(-probabilities))


def target_metrics(
    raw_rewards: np.ndarray,
    rank_order: np.ndarray,
    baseline_probabilities: np.ndarray,
    reward_scale: float,
    threshold: float,
    strength: float,
    sample_count: int,
) -> dict[str, float | int]:
    log_rewards = np.asarray(
        terminal_log_rewards(
            raw_rewards,
            reward_scale,
            tail_threshold=threshold if strength > 0 else None,
            tail_strength=strength,
        )
    )
    probabilities, log_partition = probabilities_from_log_rewards(log_rewards)
    inclusion = expected_inclusion(probabilities, sample_count)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    tail_mask = raw_rewards > threshold
    kl_from_baseline = float(
        np.sum(probabilities * np.log(probabilities / baseline_probabilities))
    )
    return {
        "tail_strength": strength,
        "exact_log_partition": log_partition,
        "expected_raw_reward": float(np.sum(probabilities * raw_rewards)),
        "positive_reward_mass": float(probabilities[raw_rewards > 0].sum()),
        "defined_tail_mass": float(probabilities[tail_mask].sum()),
        "top_10_mass": float(probabilities[rank_order[:10]].sum()),
        "top_100_mass": float(probabilities[rank_order[:100]].sum()),
        "top_1000_mass": float(probabilities[rank_order[:1000]].sum()),
        "entropy": entropy,
        "effective_support": float(np.exp(entropy)),
        "kl_from_baseline": kl_from_baseline,
        "expected_unique_count": float(inclusion.sum()),
        "expected_duplicate_fraction": float(1 - inclusion.sum() / sample_count),
        "expected_unique_top_100": float(inclusion[rank_order[:100]].sum()),
    }


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: list[dict[str, float | int]], selected: float, path: Path) -> None:
    strengths = np.asarray([row["tail_strength"] for row in rows], dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    panels = [
        ("top_100_mass", "Exact top-100 probability", "Tail probability"),
        ("expected_raw_reward", "Expected raw reward", "Raw reward"),
        ("expected_unique_count", "Expected unique procedures", "Count in 1,024"),
        ("expected_unique_top_100", "Expected unique top-100", "Count in 1,024"),
    ]
    for axis, (field, title, ylabel) in zip(axes.flat, panels):
        values = [float(row[field]) for row in rows]
        axis.plot(strengths, values, marker="o", color="#0B6E4F")
        axis.axvline(selected, color="#C44536", linestyle="--", label=f"Selected {selected:g}")
        axis.set(xlabel="Tail strength lambda", ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Exact Wang-inspired upper-tail target sweep")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rewards = read_reference(args.reference_csv.resolve())
    rank_order = np.argsort(-raw_rewards, kind="stable")

    baseline_logs = np.asarray(terminal_log_rewards(raw_rewards, args.reward_scale))
    baseline_probabilities, _ = probabilities_from_log_rewards(baseline_logs)
    rows = [
        target_metrics(
            raw_rewards,
            rank_order,
            baseline_probabilities,
            args.reward_scale,
            args.tail_threshold,
            strength,
            args.sample_count,
        )
        for strength in args.strengths
    ]
    selected_row = next(
        row for row in rows if np.isclose(row["tail_strength"], args.selected_strength)
    )
    tail_count = int(np.sum(raw_rewards > args.tail_threshold))
    summary = {
        "reference_csv": str(args.reference_csv.resolve()),
        "procedure_count": len(raw_rewards),
        "reward_scale": args.reward_scale,
        "tail_definition": {
            "rule": "raw_reward > threshold",
            "threshold": args.tail_threshold,
            "procedure_count": tail_count,
            "procedure_fraction": float(tail_count / len(raw_rewards)),
            "reason": (
                "The fixed raw threshold avoids splitting the numerical reward tie at "
                "ranks 100 and 101."
            ),
        },
        "formula": (
            "log_R_tail = clip(raw_reward / reward_scale + lambda * "
            "max(raw_reward - threshold, 0) / reward_scale, -20, 20)"
        ),
        "sample_count": args.sample_count,
        "selected_strength": args.selected_strength,
        "selection_reason": (
            "Preselected as a restrained candidate: increase upper-tail mass without the "
            "strong concentration predicted at lambda >= 0.5."
        ),
        "selected_metrics": selected_row,
        "sweep": rows,
    }
    write_csv(rows, output_dir / "tail_target_sweep.csv")
    with (output_dir / "tail_target_sweep.json").open("w", encoding="utf8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    plot_sweep(rows, args.selected_strength, output_dir / "tail_target_sweep.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
