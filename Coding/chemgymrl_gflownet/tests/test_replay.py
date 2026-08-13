from types import SimpleNamespace

import torch

from chemgymrl_gflownet import (
    FixedTailReplayBuffer,
    OfflineFirstDataSource,
    PercentileMixedReplayBuffer,
    PositiveRewardReplayBuffer,
)
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


def test_fixed_tail_replay_waits_for_both_sides_of_threshold():
    replay = FixedTailReplayBuffer(
        replay_config(capacity=16),
        raw_reward_threshold=0.1,
    )
    for raw_reward in (-0.1, 0.0, 0.05, 0.1, 0.11, 0.12, 0.13):
        replay.push(*replay_entry(raw_reward))

    assert not replay.can_sample(8)
    assert replay.last_tail_pool_size == 3
    assert replay.last_non_tail_pool_size == 4

    replay.push(*replay_entry(0.14))

    assert replay.can_sample(8)


def test_fixed_tail_replay_samples_four_from_each_pool_without_replacement():
    replay = FixedTailReplayBuffer(
        replay_config(capacity=16),
        raw_reward_threshold=0.1,
    )
    rewards = (-0.2, -0.1, 0.0, 0.1, 0.11, 0.12, 0.13, 0.14)
    for raw_reward in rewards:
        replay.push(*replay_entry(raw_reward))

    trajectories, _, properties, _, _ = replay.sample(8)

    assert replay.last_tail_sample_count == 4
    assert replay.last_non_tail_sample_count == 4
    assert torch.all(properties[:4, 1] > 0.1)
    assert torch.all(properties[4:, 1] <= 0.1)
    assert len(set(properties[:, 1].tolist())) == 8
    assert all(not trajectory["is_online"] for trajectory in trajectories)


def test_fixed_tail_replay_does_not_evict_rare_tail_with_non_tail_entries():
    replay = FixedTailReplayBuffer(
        replay_config(capacity=8),
        raw_reward_threshold=0.1,
    )
    for raw_reward in (0.11, 0.12, 0.13, 0.14):
        replay.push(*replay_entry(raw_reward))
    for raw_reward in range(20):
        replay.push(*replay_entry(-float(raw_reward)))

    _, _, properties, _, _ = replay.sample(8)

    assert len(replay) == 8
    assert torch.all(properties[:4, 1] > 0.1)
    assert torch.all(properties[4:, 1] <= 0.1)


def test_offline_first_data_source_skips_fixed_tail_replay_during_warmup():
    config = replay_config(capacity=16)
    replay = FixedTailReplayBuffer(config, raw_reward_threshold=0.1)
    for raw_reward in (-0.1, 0.0, 0.05, 0.1, 0.11, 0.12, 0.13):
        replay.push(*replay_entry(raw_reward))
    source = object.__new__(OfflineFirstDataSource)
    source.active = True
    source.cfg = config
    source.replay_buffer = replay
    source.iterators = []
    source.do_sample_replay(8)
    iterator = source.iterators[0]()

    trajectories, batch_info = next(iterator)

    assert trajectories == []
    assert batch_info == {}
