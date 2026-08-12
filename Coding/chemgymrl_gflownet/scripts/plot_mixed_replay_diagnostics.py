#!/usr/bin/env python3
"""Plot mixed-replay cutoff and sampled replay composition across runs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELD_PATTERN = re.compile(r"([A-Za-z_]+):(-?[0-9]+(?:\.[0-9]+)?)")
ITERATION_PATTERN = re.compile(r"logger - iteration ([0-9]+) :")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        required=True,
        metavar=("LABEL", "TRAIN_LOG"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_log(path: Path) -> list[dict[str, float]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = ITERATION_PATTERN.search(line)
            if not match:
                continue
            fields = {name: float(value) for name, value in FIELD_PATTERN.findall(line)}
            required = {"replay_cutoff_raw_reward", "replay_positive_fraction"}
            if required.issubset(fields):
                records.append({"iteration": float(match.group(1)), **fields})
    if not records:
        raise ValueError(f"no mixed-replay diagnostics found in {path}")
    return records


def main() -> None:
    args = parse_args()
    figure, (cutoff_axis, composition_axis) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for label, path in args.run:
        records = parse_log(Path(path))
        iterations = [record["iteration"] for record in records]
        cutoff_axis.plot(
            iterations,
            [record["replay_cutoff_raw_reward"] for record in records],
            label=label,
        )
        composition_axis.plot(
            iterations,
            [record["replay_positive_fraction"] for record in records],
            label=label,
            alpha=0.85,
        )

    cutoff_axis.axhline(0, color="#333333", linewidth=0.8)
    cutoff_axis.set(
        xlabel="Training step",
        ylabel="Raw reward",
        title="Top-decile replay cutoff",
    )
    composition_axis.set(
        xlabel="Training step",
        ylabel="Positive fraction",
        title="Positive procedures in replay batch",
        ylim=(-0.03, 1.03),
    )
    for axis in (cutoff_axis, composition_axis):
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Mixed-replay diagnostics across matched seeds")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
