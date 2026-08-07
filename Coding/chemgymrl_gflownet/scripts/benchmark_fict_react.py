#!/usr/bin/env python3
"""Benchmark the discretised ChemGymRL reaction-procedure search space."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chemgymrl_gflownet import CONTROL_NAMES, DEFAULT_REWARD_SCALE, FictReactProcedureBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("random", "exhaustive"), default="random")
    parser.add_argument("--samples", type=int, default=2000, help="Unique procedures in random mode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=DEFAULT_REWARD_SCALE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    return args


def index_to_tokens(index: int, base: int, length: int) -> tuple[int, ...]:
    tokens = [0] * length
    for position in range(length - 1, -1, -1):
        index, tokens[position] = divmod(index, base)
    return tuple(tokens)


def procedures(mode: str, samples: int, seed: int, base: int, length: int):
    space_size = base**length
    if mode == "exhaustive":
        yield from itertools.product(range(base), repeat=length)
        return
    if samples > space_size:
        raise ValueError(f"Cannot draw {samples} unique procedures from a space of {space_size}")
    rng = np.random.default_rng(seed)
    for index in rng.choice(space_size, size=samples, replace=False):
        yield index_to_tokens(int(index), base, length)


def write_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rewards(rewards: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.hist(rewards, bins=50, color="#267365", alpha=0.82)
    axis.axvline(rewards.mean(), color="#9C2F2F", linestyle="--", label=f"Mean {rewards.mean():.4f}")
    axis.axvline(np.median(rewards), color="#CF8A2E", linestyle=":", label=f"Median {np.median(rewards):.4f}")
    axis.set(
        title="ChemGymRL fictitious-reaction procedure rewards",
        xlabel="Raw ChemGymRL terminal reward",
        ylabel="Procedure count",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    with FictReactProcedureBridge(seed=args.seed, reward_scale=args.reward_scale) as bridge:
        for tokens in procedures(
            args.mode,
            args.samples,
            args.seed,
            bridge.num_token_values,
            len(CONTROL_NAMES),
        ):
            evaluation = bridge.evaluate(tokens)
            row: dict[str, object] = {
                "tokens": " ".join(map(str, evaluation.tokens)),
                "raw_reward": evaluation.raw_reward,
                "positive_reward": evaluation.positive_reward,
            }
            row.update(dict(zip(CONTROL_NAMES, evaluation.action)))
            rows.append(row)
        simulator_calls = bridge.simulator_calls
        cache_hits = bridge.cache_hits
    elapsed = time.perf_counter() - start

    rewards = np.asarray([row["raw_reward"] for row in rows], dtype=float)
    ranked = sorted(rows, key=lambda row: float(row["raw_reward"]), reverse=True)
    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "control_bins": {
            name: bridge.control_bins[index].tolist() for index, name in enumerate(CONTROL_NAMES)
        },
        "search_space_size": bridge.num_token_values ** len(CONTROL_NAMES),
        "sample_count": len(rows),
        "simulator_calls": simulator_calls,
        "cache_hits": cache_hits,
        "elapsed_seconds": elapsed,
        "simulations_per_second": simulator_calls / elapsed,
        "reward_scale": args.reward_scale,
        "raw_reward": {
            "minimum": float(rewards.min()),
            "mean": float(rewards.mean()),
            "standard_deviation": float(rewards.std()),
            "median": float(np.median(rewards)),
            "quantile_05": float(np.quantile(rewards, 0.05)),
            "quantile_95": float(np.quantile(rewards, 0.95)),
            "maximum": float(rewards.max()),
            "negative_count": int((rewards < 0).sum()),
            "zero_count": int((rewards == 0).sum()),
            "positive_count": int((rewards > 0).sum()),
            "top_10_mean": float(np.mean([row["raw_reward"] for row in ranked[:10]])),
            "top_100_mean": float(np.mean([row["raw_reward"] for row in ranked[:100]])),
        },
        "best_procedure": ranked[0],
    }

    write_rows(rows, output_dir / "procedures.csv")
    plot_rewards(rewards, output_dir / "raw_reward_histogram.png")
    with (output_dir / "summary.json").open("w", encoding="utf8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
