from types import SimpleNamespace

import torch

from chemgymrl_gflownet import PercentileMixedReplayBuffer, PositiveRewardReplayBuffer
from gflownet.config import Config


def replay_config(capacity: int = 4) -> Config:
    config = Config()
    config.replay.capacity = capacity
    config.replay.warmup = 1
    return config


def replay_entry(raw_reward: float, valid: bool = True):
    trajectory = {"result": SimpleNamespace(seq=[0] * 9), "is_online": True}
    return (
        trajectory,
        torch.tensor(raw_reward),
        torch.tensor([1.0, raw_reward]),
        {"encoding": torch.tensor([1.0])},
        valid,
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


def test_mixed_replay_samples_equally_from_top_decile_and_remainder():
    replay = PercentileMixedReplayBuffer(replay_config(capacity=100))
    for raw_reward in range(100):
        replay.push(*replay_entry(float(raw_reward)))

    _, _, properties, _, _ = replay.sample(20)

    assert replay.last_top_pool_size == 10
    assert replay.last_remainder_pool_size == 90
    assert replay.last_top_sample_count == 10
    assert replay.last_remainder_sample_count == 10
    assert replay.last_cutoff_raw_reward == 90.0
    assert torch.all(properties[:10, 1] >= 90)
    assert torch.all(properties[10:, 1] < 90)


def test_mixed_replay_retains_valid_entries_across_reward_categories():
    replay = PercentileMixedReplayBuffer(replay_config())
    replay.push(*replay_entry(-0.2))
    replay.push(*replay_entry(0.0))
    replay.push(*replay_entry(0.4))
    replay.push(*replay_entry(1.0, valid=False))

    assert len(replay) == 3
    assert replay.total_seen == 4
    assert replay.total_retained == 3
