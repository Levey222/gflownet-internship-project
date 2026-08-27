import numpy as np
import pytest

from chemgymrl_gflownet import ACTION_DIMENSION, FictReactProcedureBridge


def test_decode_maps_tokens_to_control_values():
    with FictReactProcedureBridge() as bridge:
        action = bridge.decode((0, 1, 2, 0, 1, 2, 0, 1, 2))
    np.testing.assert_array_equal(
        action,
        np.array([0, 0.5, 1, 0, 0.5, 0.9, 0, 0.5, 0.9], dtype=np.float32),
    )


def test_default_addition_times_all_occur_before_episode_end():
    with FictReactProcedureBridge() as bridge:
        assert np.all(bridge.control_bins[5:] < 1)
        assert bridge.num_token_values == 3


@pytest.mark.parametrize(
    "tokens, error",
    [
        ((0,) * (ACTION_DIMENSION - 1), ValueError),
        ((0,) * (ACTION_DIMENSION - 1) + (3,), ValueError),
        ((0,) * (ACTION_DIMENSION - 1) + (0.5,), TypeError),
    ],
)
def test_invalid_token_sequences_are_rejected(tokens, error):
    with FictReactProcedureBridge() as bridge:
        with pytest.raises(error):
            bridge.decode(tokens)


def test_bandit_reward_is_deterministic_for_the_same_action():
    action = np.full(ACTION_DIMENSION, 0.5, dtype=np.float32)
    with FictReactProcedureBridge() as bridge:
        first = bridge.simulate_action(action)
        second = bridge.simulate_action(action)
    assert first == second


def test_cached_procedure_avoids_repeated_simulation():
    with FictReactProcedureBridge() as bridge:
        first = bridge.raw_reward((1,) * ACTION_DIMENSION)
        second = bridge.raw_reward((1,) * ACTION_DIMENSION)
        assert bridge.simulator_calls == 1
        assert bridge.cache_hits == 1
    assert first == second


def test_positive_reward_is_finite_positive_and_order_preserving():
    with FictReactProcedureBridge(reward_scale=0.1) as bridge:
        transformed = [bridge.to_positive_reward(reward) for reward in (-0.5, 0.0, 0.5)]
    assert np.isfinite(transformed).all()
    assert all(reward > 0 for reward in transformed)
    assert transformed[0] < transformed[1] < transformed[2]


def test_upper_tail_reward_changes_only_values_above_the_threshold():
    with FictReactProcedureBridge(
        reward_scale=0.05,
        tail_threshold=0.1,
        tail_strength=0.25,
    ) as bridge:
        below = bridge.to_positive_reward(0.05)
        at_threshold = bridge.to_positive_reward(0.1)
        above = bridge.to_positive_reward(0.15)

    assert below == pytest.approx(np.exp(1.0))
    assert at_threshold == pytest.approx(np.exp(2.0))
    assert above == pytest.approx(np.exp(3.25))


def test_zero_tail_strength_recovers_baseline_with_replay_threshold():
    raw_rewards = (-0.05, 0.05, 0.15)
    with FictReactProcedureBridge(reward_scale=0.05) as baseline_bridge:
        baseline = [baseline_bridge.to_positive_reward(value) for value in raw_rewards]
    with FictReactProcedureBridge(
        reward_scale=0.05,
        tail_threshold=0.1,
        tail_strength=0.0,
    ) as replay_control_bridge:
        replay_control = [
            replay_control_bridge.to_positive_reward(value) for value in raw_rewards
        ]

    np.testing.assert_array_equal(replay_control, baseline)


def test_upper_tail_reward_requires_a_threshold():
    with pytest.raises(ValueError, match="tail_threshold"):
        FictReactProcedureBridge(tail_strength=0.25)


def test_legacy_shared_bins_are_still_supported():
    with FictReactProcedureBridge(bins=(0.0, 0.25, 1.0)) as bridge:
        action = bridge.decode((1,) * ACTION_DIMENSION)
    np.testing.assert_array_equal(action, np.full(ACTION_DIMENSION, 0.25, dtype=np.float32))
