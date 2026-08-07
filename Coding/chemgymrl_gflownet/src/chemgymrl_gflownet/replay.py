"""Positive-reward replay support for the ChemGymRL procedure task."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from gflownet.config import Config
from gflownet.data.data_source import DataSource
from gflownet.data.replay_buffer import ReplayBuffer, detach_and_cpu
from gflownet.utils.misc import get_worker_rng


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

    def _push_fallback(self, entry: tuple) -> None:
        if len(self.fallback_buffer) < self.capacity:
            self.fallback_buffer.append(entry)
        else:
            self.fallback_buffer[self.fallback_position] = entry
        self.fallback_position = (self.fallback_position + 1) % self.capacity


class OfflineFirstDataSource(DataSource):
    """Keep replay trajectories first for the upstream TB loss diagnostics."""

    def create_batch(self, trajs, batch_info):
        ordered = sorted(trajs, key=lambda trajectory: bool(trajectory.get("is_online", False)))
        return super().create_batch(ordered, batch_info)
