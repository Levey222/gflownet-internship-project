"""Decode discrete procedures and evaluate them in ChemGymRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import chemistrylab  # noqa: F401  # Registers ChemGymRL environments with Gymnasium.
import gymnasium as gym
import numpy as np

from .tail_reward import terminal_positive_rewards


ENVIRONMENT_ID = "FictReactBandit-v1"
ACTION_DIMENSION = 9
CONTROL_NAMES = (
    "temperature",
    "fict_A_amount",
    "fict_B_amount",
    "fict_C_amount",
    "fict_D_amount",
    "fict_A_addition_time",
    "fict_B_addition_time",
    "fict_C_addition_time",
    "fict_D_addition_time",
)
DEFAULT_STANDARD_BINS = (0.0, 0.5, 1.0)
DEFAULT_ADDITION_TIME_BINS = (0.0, 0.5, 0.9)
DEFAULT_CONTROL_BINS = (
    DEFAULT_STANDARD_BINS,
    DEFAULT_STANDARD_BINS,
    DEFAULT_STANDARD_BINS,
    DEFAULT_STANDARD_BINS,
    DEFAULT_STANDARD_BINS,
    DEFAULT_ADDITION_TIME_BINS,
    DEFAULT_ADDITION_TIME_BINS,
    DEFAULT_ADDITION_TIME_BINS,
    DEFAULT_ADDITION_TIME_BINS,
)
DEFAULT_REWARD_SCALE = 0.05


@dataclass(frozen=True)
class ProcedureEvaluation:
    """One decoded procedure and its raw and positive rewards."""

    tokens: tuple[int, ...]
    action: tuple[float, ...]
    raw_reward: float
    positive_reward: float


class FictReactProcedureBridge:
    """Evaluate fixed-length, discretised procedures in FictReactBandit-v1."""

    def __init__(
        self,
        bins: Sequence[float] | None = None,
        control_bins: Sequence[Sequence[float]] | None = None,
        seed: int = 0,
        reward_scale: float = DEFAULT_REWARD_SCALE,
        tail_threshold: float | None = None,
        tail_strength: float = 0.0,
    ) -> None:
        if bins is not None and control_bins is not None:
            raise ValueError("provide either bins or control_bins, not both")
        if control_bins is None:
            control_bins = DEFAULT_CONTROL_BINS if bins is None else (tuple(bins),) * ACTION_DIMENSION

        bin_array = np.asarray(control_bins, dtype=np.float32)
        if bin_array.ndim != 2 or bin_array.shape[0] != ACTION_DIMENSION or bin_array.shape[1] < 2:
            raise ValueError(
                f"control_bins must have shape ({ACTION_DIMENSION}, n) with at least two values per control"
            )
        if not np.isfinite(bin_array).all() or np.any(bin_array < 0) or np.any(bin_array > 1):
            raise ValueError("every bin value must be finite and within [0, 1]")
        if np.any(np.diff(bin_array, axis=1) <= 0):
            raise ValueError("bin values for every control must be strictly increasing")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")

        self.control_bins = bin_array
        self.num_token_values = int(bin_array.shape[1])
        self.seed = seed
        self.reward_scale = float(reward_scale)
        self.tail_threshold = tail_threshold
        self.tail_strength = float(tail_strength)
        terminal_positive_rewards(
            0.0,
            self.reward_scale,
            tail_threshold=self.tail_threshold,
            tail_strength=self.tail_strength,
        )
        self.environment = gym.make(ENVIRONMENT_ID)
        if self.environment.action_space.shape != (ACTION_DIMENSION,):
            raise ValueError(
                f"{ENVIRONMENT_ID} action shape changed from {(ACTION_DIMENSION,)} "
                f"to {self.environment.action_space.shape}"
            )
        targets = list(self.environment.unwrapped.targets)
        if targets != ["fict_I"]:
            raise ValueError(f"{ENVIRONMENT_ID} target changed from ['fict_I'] to {targets}")

        self._raw_reward_cache: dict[tuple[int, ...], float] = {}
        self.simulator_calls = 0
        self.cache_hits = 0

    def validate_tokens(self, tokens: Iterable[int]) -> tuple[int, ...]:
        token_tuple = tuple(tokens)
        if len(token_tuple) != ACTION_DIMENSION:
            raise ValueError(f"a procedure must contain exactly {ACTION_DIMENSION} tokens")
        if any(not isinstance(token, (int, np.integer)) for token in token_tuple):
            raise TypeError("procedure tokens must be integers")
        if any(token < 0 or token >= self.num_token_values for token in token_tuple):
            raise ValueError(f"procedure tokens must be between 0 and {self.num_token_values - 1}")
        return tuple(int(token) for token in token_tuple)

    def decode(self, tokens: Iterable[int]) -> np.ndarray:
        token_tuple = self.validate_tokens(tokens)
        token_indices = np.asarray(token_tuple, dtype=np.int64)
        return self.control_bins[np.arange(ACTION_DIMENSION), token_indices].copy()

    def simulate_action(self, action: Sequence[float]) -> float:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (ACTION_DIMENSION,):
            raise ValueError(f"action must have shape {(ACTION_DIMENSION,)}")
        if not np.isfinite(action_array).all() or np.any(action_array < 0) or np.any(action_array > 1):
            raise ValueError("all action values must be finite and within [0, 1]")

        self.environment.reset(seed=self.seed)
        _, reward, terminated, truncated, _ = self.environment.step(action_array)
        if not terminated or truncated:
            raise RuntimeError(f"{ENVIRONMENT_ID} did not finish as a one-step bandit")
        self.simulator_calls += 1
        return float(reward)

    def raw_reward(self, tokens: Iterable[int]) -> float:
        token_tuple = self.validate_tokens(tokens)
        if token_tuple in self._raw_reward_cache:
            self.cache_hits += 1
            return self._raw_reward_cache[token_tuple]
        reward = self.simulate_action(self.decode(token_tuple))
        self._raw_reward_cache[token_tuple] = reward
        return reward

    def to_positive_reward(self, raw_reward: float) -> float:
        return float(
            terminal_positive_rewards(
                raw_reward,
                self.reward_scale,
                tail_threshold=self.tail_threshold,
                tail_strength=self.tail_strength,
            )
        )

    def evaluate(self, tokens: Iterable[int]) -> ProcedureEvaluation:
        token_tuple = self.validate_tokens(tokens)
        action = self.decode(token_tuple)
        raw_reward = self.raw_reward(token_tuple)
        return ProcedureEvaluation(
            tokens=token_tuple,
            action=tuple(float(value) for value in action),
            raw_reward=raw_reward,
            positive_reward=self.to_positive_reward(raw_reward),
        )

    @property
    def cache_size(self) -> int:
        return len(self._raw_reward_cache)

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> FictReactProcedureBridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
