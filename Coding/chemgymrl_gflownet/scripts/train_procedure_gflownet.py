#!/usr/bin/env python3
"""Train a fixed-length procedure GFlowNet using a ChemGymRL terminal reward."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODING_ROOT = PACKAGE_ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gflownet-matplotlib")
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(CODING_ROOT / "recursionpharma-gflownet" / "src"))

from chemgymrl_gflownet import (  # noqa: E402
    ChemGymRLProcedureTrainer,
    DEFAULT_REWARD_SCALE,
    seed_process,
)
from gflownet.config import Config, init_empty  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reward-scale", type=float, default=DEFAULT_REWARD_SCALE)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--final-batches", type=int, default=2)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--positive-replay", action="store_true")
    parser.add_argument("--replay-capacity", type=int, default=512)
    parser.add_argument("--replay-warmup", type=int, default=1)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if args.validation_batches < 1:
        parser.error("--validation-batches must be at least 1")
    if args.final_batches < 0:
        parser.error("--final-batches cannot be negative")
    if args.replay_capacity < 1:
        parser.error("--replay-capacity must be at least 1")
    if args.replay_warmup < 1 or args.replay_warmup > args.replay_capacity:
        parser.error("--replay-warmup must be between 1 and --replay-capacity")
    if args.replay_batch_size < 1:
        parser.error("--replay-batch-size must be at least 1")
    return args


def default_log_dir(seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return PACKAGE_ROOT / "logs" / f"procedure_tb_seed{seed}_{timestamp}"


def main() -> None:
    args = parse_args()
    log_dir = (args.log_dir or default_log_dir(args.seed)).resolve()
    seed_process(args.seed)

    config = init_empty(Config())
    config.desc = "Fixed-length procedure GFlowNet with a ChemGymRL terminal reward"
    config.log_dir = str(log_dir)
    config.device = args.device
    config.seed = args.seed
    config.num_workers = 0
    config.num_training_steps = args.steps
    config.print_every = args.print_every
    config.validate_every = args.validate_every
    config.num_validation_gen_steps = args.validation_batches
    config.num_final_gen_steps = args.final_batches
    config.overwrite_existing_exp = args.overwrite
    config.algo.method = "TB"
    config.algo.num_from_policy = args.batch_size
    config.algo.valid_num_from_policy = args.batch_size
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [1.0]
    if args.positive_replay:
        config.replay.use = True
        config.replay.capacity = args.replay_capacity
        config.replay.warmup = args.replay_warmup
        config.replay.num_from_replay = args.replay_batch_size
        config.replay.num_new_samples = args.batch_size

    print(f"Writing experiment outputs to: {log_dir}")
    trainer = ChemGymRLProcedureTrainer(
        config,
        reward_scale=args.reward_scale,
        positive_replay=args.positive_replay,
    )
    try:
        trainer.run()
    finally:
        trainer.terminate()


if __name__ == "__main__":
    main()
