"""Positive terminal-reward transforms for upper-tail experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


LOG_REWARD_CLIP = 20.0


def terminal_log_rewards(
    raw_rewards: float | Sequence[float] | np.ndarray,
    reward_scale: float,
    tail_threshold: float | None = None,
    tail_strength: float = 0.0,
) -> float | np.ndarray:
    """Convert raw outcomes to baseline or upper-tail-shaped log rewards.

    The dimensionless tail bonus is
    ``tail_strength * max(raw_reward - tail_threshold, 0) / reward_scale``.
    A zero strength recovers the original reward transform exactly.
    """

    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("reward_scale must be finite and positive")
    if not np.isfinite(tail_strength) or tail_strength < 0:
        raise ValueError("tail_strength must be finite and non-negative")
    if tail_strength > 0 and tail_threshold is None:
        raise ValueError("tail_threshold is required when tail_strength is positive")
    if tail_threshold is not None and not np.isfinite(tail_threshold):
        raise ValueError("tail_threshold must be finite")

    values = np.asarray(raw_rewards, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("raw rewards must be finite")

    log_rewards = values / reward_scale
    if tail_strength > 0:
        excess = np.maximum(values - float(tail_threshold), 0.0) / reward_scale
        log_rewards = log_rewards + tail_strength * excess
    log_rewards = np.clip(log_rewards, -LOG_REWARD_CLIP, LOG_REWARD_CLIP)

    if values.ndim == 0:
        return float(log_rewards)
    return log_rewards


def terminal_positive_rewards(
    raw_rewards: float | Sequence[float] | np.ndarray,
    reward_scale: float,
    tail_threshold: float | None = None,
    tail_strength: float = 0.0,
) -> float | np.ndarray:
    """Return strictly positive rewards for the requested terminal target."""

    log_rewards = terminal_log_rewards(
        raw_rewards,
        reward_scale,
        tail_threshold=tail_threshold,
        tail_strength=tail_strength,
    )
    positive = np.exp(log_rewards)
    if np.ndim(log_rewards) == 0:
        return float(positive)
    return positive
