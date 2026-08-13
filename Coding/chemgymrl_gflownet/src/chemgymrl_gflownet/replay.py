"""Replay strategies for the ChemGymRL procedure task."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from gflownet.config import Config
from gflownet.data.data_source import DataSource
from gflownet.data.replay_buffer import ReplayBuffer, detach_and_cpu
from gflownet.utils.misc import get_worker_rng


def _format_sampled_entries(selected: list[tuple]):
    output = list(zip(*selected))
    for index in range(len(output)):
        if all(isinstance(value, np.ndarray) for value in output[index]):
            output[index] = np.stack(output[index], axis=0)
        elif all(isinstance(value, torch.Tensor) for value in output[index]):
            output[index] = torch.stack(output[index], dim=0)
        else:
            output[index] = list(output[index])
    for trajectory in output[0]:
        trajectory["is_online"] = False
    return output


class PositiveRewardReplayBuffer(ReplayBuffer):
    """Retain positive trajectories and use recent data only as an empty-buffer fallback."""

    def __init__(self, cfg: Config, raw_reward_threshold: float = 0.0) -> None:
        super().__init__(cfg)
        self.raw_reward_threshold = raw_reward_threshold
        self.fallback_buffer: list[tuple] = []
        self.fallback_position = 0
        self.total_seen = 0
        self.total_retained = 0

    def push(self, *args) -> None:
        if len(args) != 5:
            raise ValueError("replay entries must contain trajectory, reward, properties, conditioning, and validity")
        detached = detach_and_cpu(args)
        self.total_seen += 1
        self._push_fallback(detached)

        obj_props = detached[2]
        is_valid = bool(detached[4])
        raw_reward = float(obj_props[1])
        if is_valid and raw_reward > self.raw_reward_threshold:
            super().push(*detached)
            self.total_retained += 1

    def sample(self, batch_size: int):
        source = self.buffer if self.buffer else self.fallback_buffer
        if not source:
            raise ValueError("cannot sample from an empty replay buffer")
        indexes = get_worker_rng().choice(len(source), batch_size)
        selected = [deepcopy(source[index]) for index in indexes]
        return _format_sampled_entries(selected)

    def _push_fallback(self, entry: tuple) -> None:
        if len(self.fallback_buffer) < self.capacity:
            self.fallback_buffer.append(entry)
        else:
            self.fallback_buffer[self.fallback_position] = entry
        self.fallback_position = (self.fallback_position + 1) % self.capacity


class PercentileMixedReplayBuffer(ReplayBuffer):
    """Sample equally from the highest-reward rank group and its complement."""

    def __init__(
        self,
        cfg: Config,
        top_fraction: float = 0.1,
        top_sample_fraction: float = 0.5,
    ) -> None:
        super().__init__(cfg)
        if not 0 < top_fraction < 1:
            raise ValueError("top_fraction must be between 0 and 1")
        if not 0 < top_sample_fraction < 1:
            raise ValueError("top_sample_fraction must be between 0 and 1")
        self.top_fraction = top_fraction
        self.top_sample_fraction = top_sample_fraction
        self.total_seen = 0
        self.total_retained = 0
        self.last_top_pool_size = 0
        self.last_remainder_pool_size = 0
        self.last_top_sample_count = 0
        self.last_remainder_sample_count = 0
        self.last_cutoff_raw_reward = float("nan")

    def push(self, *args) -> None:
        if len(args) != 5:
            raise ValueError("replay entries must contain trajectory, reward, properties, conditioning, and validity")
        self.total_seen += 1
        if not bool(args[4]):
            return
        super().push(*args)
        self.total_retained += 1

    def sample(self, batch_size: int):
        if not self.buffer:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        rng = get_worker_rng()
        rewards = np.asarray([float(entry[2][1]) for entry in self.buffer])
        shuffled_indexes = rng.permutation(len(self.buffer))
        ranked_indexes = shuffled_indexes[
            np.argsort(-rewards[shuffled_indexes], kind="stable")
        ]
        top_pool_size = max(1, int(np.ceil(len(self.buffer) * self.top_fraction)))
        top_pool = ranked_indexes[:top_pool_size]
        remainder_pool = ranked_indexes[top_pool_size:]

        top_sample_count = int(round(batch_size * self.top_sample_fraction))
        top_sample_count = min(max(top_sample_count, 1), batch_size)
        remainder_sample_count = batch_size - top_sample_count
        if not len(remainder_pool):
            top_sample_count = batch_size
            remainder_sample_count = 0

        selected_indexes = list(rng.choice(top_pool, top_sample_count))
        if remainder_sample_count:
            selected_indexes.extend(
                rng.choice(remainder_pool, remainder_sample_count).tolist()
            )
        selected = [deepcopy(self.buffer[index]) for index in selected_indexes]

        self.last_top_pool_size = len(top_pool)
        self.last_remainder_pool_size = len(remainder_pool)
        self.last_top_sample_count = top_sample_count
        self.last_remainder_sample_count = remainder_sample_count
        self.last_cutoff_raw_reward = float(rewards[top_pool].min())
        return _format_sampled_entries(selected)


class FixedTailReplayBuffer(ReplayBuffer):
    """Replay a fixed mixture from either side of an absolute reward threshold."""

    def __init__(
        self,
        cfg: Config,
        raw_reward_threshold: float,
        tail_sample_fraction: float = 0.5,
    ) -> None:
        super().__init__(cfg)
        if not np.isfinite(raw_reward_threshold):
            raise ValueError("raw_reward_threshold must be finite")
        if not 0 < tail_sample_fraction < 1:
            raise ValueError("tail_sample_fraction must be between 0 and 1")
        self.raw_reward_threshold = raw_reward_threshold
        self.tail_sample_fraction = tail_sample_fraction
        self.tail_capacity = max(1, int(round(self.capacity * tail_sample_fraction)))
        self.non_tail_capacity = self.capacity - self.tail_capacity
        if self.non_tail_capacity < 1:
            raise ValueError("fixed-tail replay capacity must leave room for both pools")
        self.tail_buffer: list[tuple] = []
        self.non_tail_buffer: list[tuple] = []
        self.tail_position = 0
        self.non_tail_position = 0
        self.total_seen = 0
        self.total_retained = 0
        self.last_tail_pool_size = 0
        self.last_non_tail_pool_size = 0
        self.last_tail_sample_count = 0
        self.last_non_tail_sample_count = 0
        self.last_replay_active = False

    def push(self, *args) -> None:
        if len(args) != 5:
            raise ValueError("replay entries must contain trajectory, reward, properties, conditioning, and validity")
        self.total_seen += 1
        if not bool(args[4]):
            return
        detached = detach_and_cpu(args)
        raw_reward = float(detached[2][1])
        if self._is_tail(raw_reward):
            self.tail_position = self._push_pool(
                self.tail_buffer,
                self.tail_position,
                self.tail_capacity,
                detached,
            )
        else:
            self.non_tail_position = self._push_pool(
                self.non_tail_buffer,
                self.non_tail_position,
                self.non_tail_capacity,
                detached,
            )
        self.total_retained += 1

    def can_sample(self, batch_size: int) -> bool:
        tail_sample_count, non_tail_sample_count = self._sample_counts(batch_size)
        tail_pool, non_tail_pool = self._pool_indexes()
        self.last_tail_pool_size = len(tail_pool)
        self.last_non_tail_pool_size = len(non_tail_pool)
        self.last_replay_active = (
            len(self) >= self.warmup
            and len(tail_pool) >= tail_sample_count
            and len(non_tail_pool) >= non_tail_sample_count
        )
        return self.last_replay_active

    def sample(self, batch_size: int):
        tail_sample_count, non_tail_sample_count = self._sample_counts(batch_size)
        tail_pool, non_tail_pool = self._pool_indexes()
        if not self.can_sample(batch_size):
            raise ValueError(
                "fixed-tail replay is not ready: the buffer needs enough entries "
                "on both sides of the threshold"
            )

        rng = get_worker_rng()
        selected_indexes = rng.choice(
            tail_pool,
            tail_sample_count,
            replace=False,
        ).tolist()
        selected_indexes.extend(
            rng.choice(
                non_tail_pool,
                non_tail_sample_count,
                replace=False,
            ).tolist()
        )
        combined_buffer = self.tail_buffer + self.non_tail_buffer
        selected = [deepcopy(combined_buffer[index]) for index in selected_indexes]
        self.last_tail_sample_count = tail_sample_count
        self.last_non_tail_sample_count = non_tail_sample_count
        return _format_sampled_entries(selected)

    def _pool_indexes(self) -> tuple[np.ndarray, np.ndarray]:
        tail_pool = np.arange(len(self.tail_buffer))
        non_tail_pool = np.arange(
            len(self.tail_buffer),
            len(self.tail_buffer) + len(self.non_tail_buffer),
        )
        return tail_pool, non_tail_pool

    def _sample_counts(self, batch_size: int) -> tuple[int, int]:
        if batch_size < 2:
            raise ValueError("fixed-tail replay batch size must be at least 2")
        tail_count = int(round(batch_size * self.tail_sample_fraction))
        tail_count = min(max(tail_count, 1), batch_size - 1)
        return tail_count, batch_size - tail_count

    def _is_tail(self, raw_reward: float) -> bool:
        return bool(
            np.float32(raw_reward) > np.float32(self.raw_reward_threshold)
        )

    @staticmethod
    def _push_pool(
        pool: list[tuple],
        position: int,
        capacity: int,
        entry: tuple,
    ) -> int:
        if len(pool) < capacity:
            pool.append(entry)
        else:
            pool[position] = entry
        return (position + 1) % capacity

    def __len__(self) -> int:
        return len(self.tail_buffer) + len(self.non_tail_buffer)


class OfflineFirstDataSource(DataSource):
    """Keep replay trajectories first for the upstream TB loss diagnostics."""

    def do_sample_replay(self, num_samples):
        def iterator():
            while self.active:
                can_sample = getattr(self.replay_buffer, "can_sample", None)
                if can_sample is not None and not can_sample(num_samples):
                    yield [], {}
                    continue
                trajs, *_ = self.replay_buffer.sample(num_samples)
                self.relabel_in_hindsight(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def create_batch(self, trajs, batch_info):
        ordered = sorted(trajs, key=lambda trajectory: bool(trajectory.get("is_online", False)))
        return super().create_batch(ordered, batch_info)
