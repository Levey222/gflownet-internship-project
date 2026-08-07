from types import SimpleNamespace

import torch

from chemgymrl_gflownet import PositiveRewardReplayBuffer
from gflownet.config import Config


def replay_config() -> Config:
    config = Config()
    config.replay.capacity = 4
    config.replay.warmup = 1
    return config


def replay_entry(raw_reward: float):
    trajectory = {"result": SimpleNamespace(seq=[0] * 9), "is_online": True}
    return (
        trajectory,
        torch.tensor(raw_reward),
        torch.tensor([1.0, raw_reward]),
        {"encoding": torch.tensor([1.0])},
        True,
    )


def test_positive_replay_uses_recent_fallback_until_positive_reward_exists():
    replay = PositiveRewardReplayBuffer(replay_config())
    replay.push(*replay_entry(0.0))

    trajectories, _, properties, _, _ = replay.sample(2)

    assert len(replay) == 0
    assert properties[:, 1].tolist() == [0.0, 0.0]
    assert all(not trajectory["is_online"] for trajectory in trajectories)


def test_positive_replay_retains_and_samples_only_positive_rewards():
    replay = PositiveRewardReplayBuffer(replay_config())
    replay.push(*replay_entry(-0.1))
    replay.push(*replay_entry(0.2))
    replay.push(*replay_entry(0.4))

    _, _, properties, _, _ = replay.sample(20)

    assert len(replay) == 2
    assert replay.total_seen == 3
    assert replay.total_retained == 2
    assert torch.all(properties[:, 1] > 0)
