from types import SimpleNamespace

import numpy as np
import torch

from chemgymrl_gflownet import (
    ACTION_DIMENSION,
    ChemGymRLProcedureTask,
    ChemGymRLProcedureTrainer,
    FixedLengthSeqTransformerGFN,
    ProcedureSequenceContext,
)
from gflownet.config import Config, init_empty
from gflownet.envs.graph_building_env import GraphActionType
from gflownet.envs.seq_building_env import Seq


class RecordingBridge:
    num_token_values = 3

    def __init__(self):
        self.calls = []
        self.closed = False

    def evaluate(self, procedure):
        self.calls.append(tuple(procedure))
        raw_reward = float(sum(procedure))
        return SimpleNamespace(positive_reward=raw_reward + 1, raw_reward=raw_reward)

    def close(self):
        self.closed = True


def task_config() -> Config:
    config = Config()
    config.device = "cpu"
    config.seed = 0
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [1.0]
    config.cond.temperature.num_thermometer_dim = 1
    return config


def test_context_accepts_only_complete_nine_token_procedures():
    context = ProcedureSequenceContext(num_token_values=3, num_cond_dim=1)
    complete = context.obj_to_graph((0, 1, 2) * 3)
    partial = Seq()
    partial.seq.extend((0, 1, 2))

    assert context.is_sane(complete)
    assert not context.is_sane(partial)
    assert context.graph_to_obj(complete) == (0, 1, 2) * 3


def test_terminal_task_does_not_evaluate_partial_procedures():
    bridge = RecordingBridge()
    task = ChemGymRLProcedureTask(task_config(), bridge=bridge)

    properties, valid = task.compute_obj_properties([(0,) * ACTION_DIMENSION, (0,) * 8])

    assert valid.tolist() == [True, False]
    assert properties.tolist() == [[1.0, 0.0]]
    assert bridge.calls == [(0,) * ACTION_DIMENSION]


def test_terminal_task_returns_positive_rewards_and_log_rewards():
    bridge = RecordingBridge()
    task = ChemGymRLProcedureTask(task_config(), bridge=bridge)
    procedures = [(0,) * ACTION_DIMENSION, (1,) * ACTION_DIMENSION]

    properties, valid = task.compute_obj_properties(procedures)
    conditional = task.sample_conditional_information(len(procedures), train_it=0)
    log_rewards = task.cond_info_to_logreward(conditional, properties)

    assert valid.all()
    np.testing.assert_allclose(properties.numpy(), [[1.0, 0.0], [10.0, 9.0]])
    np.testing.assert_allclose(log_rewards.numpy(), np.log([1.0, 10.0]), rtol=1e-6)

    single_properties, _ = task.compute_obj_properties([procedures[0]])
    single_conditional = task.sample_conditional_information(1, train_it=0)
    single_log_reward = task.cond_info_to_logreward(single_conditional, single_properties)
    assert single_log_reward.shape == (1,)


def test_fixed_length_model_masks_stop_and_add_actions():
    config = task_config()
    config.algo.max_len = ACTION_DIMENSION + 1
    config.model.num_emb = 16
    config.model.num_layers = 1
    config.model.seq_transformer.num_heads = 2
    context = ProcedureSequenceContext(num_token_values=3, num_cond_dim=1)
    model = FixedLengthSeqTransformerGFN(context, config)

    empty = Seq()
    complete = context.obj_to_graph((0,) * ACTION_DIMENSION)
    batch = context.collate([context.graph_to_Data(empty), context.graph_to_Data(complete)])
    distribution, _ = model(batch, torch.zeros((2, 1)))

    stop_mask, add_mask = distribution.action_masks
    assert stop_mask.reshape(-1).tolist() == [0.0, 1.0]
    assert add_mask.reshape(-1).tolist() == [1.0, 0.0]


def test_sampler_produces_exactly_nine_tokens_then_stop(tmp_path):
    config = init_empty(Config())
    config.log_dir = str(tmp_path / "sample")
    config.device = "cpu"
    config.seed = 0
    config.num_workers = 0
    config.algo.num_from_policy = 4
    config.algo.valid_num_from_policy = 4
    trainer = ChemGymRLProcedureTrainer(config, print_config=False)
    try:
        conditional = trainer.task.sample_conditional_information(4, train_it=0)
        trajectories = trainer.algo.create_training_data_from_own_samples(
            trainer.model,
            4,
            conditional["encoding"],
            random_action_prob=0.0,
        )
    finally:
        trainer.terminate()

    assert all(len(trajectory["result"].seq) == ACTION_DIMENSION for trajectory in trajectories)
    assert all(len(trajectory["traj"]) == ACTION_DIMENSION + 1 for trajectory in trajectories)
    assert all(trajectory["traj"][-1][1].action is GraphActionType.Stop for trajectory in trajectories)
    assert all(trajectory["is_valid"] for trajectory in trajectories)
