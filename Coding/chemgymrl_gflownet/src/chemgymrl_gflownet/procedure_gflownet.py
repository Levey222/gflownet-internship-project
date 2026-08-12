"""Fixed-length GFlowNet task using a ChemGymRL terminal reward."""

from __future__ import annotations

import json
import pathlib
import socket
from numbers import Integral
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config
from gflownet.utils.sqlite_log import SQLiteLogHook
from gflownet.envs.seq_building_env import (
    AutoregressiveSeqBuildingContext,
    Seq,
    SeqBatch,
    SeqBuildingEnv,
)
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional

from .bridge import ACTION_DIMENSION, DEFAULT_REWARD_SCALE, FictReactProcedureBridge
from .replay import (
    OfflineFirstDataSource,
    PercentileMixedReplayBuffer,
    PositiveRewardReplayBuffer,
)


class ProcedureSequenceContext(AutoregressiveSeqBuildingContext):
    """Represent a complete ChemGymRL procedure as a fixed-length token tuple."""

    def __init__(self, num_token_values: int, num_cond_dim: int, target_length: int = ACTION_DIMENSION):
        if num_token_values < 2:
            raise ValueError("num_token_values must be at least 2")
        if target_length < 1:
            raise ValueError("target_length must be positive")
        super().__init__(tuple(str(index) for index in range(num_token_values)), num_cond_dim)
        self.target_length = target_length

    def graph_to_obj(self, g: Seq) -> tuple[int, ...]:
        return tuple(int(token) for token in g.seq)

    def obj_to_graph(self, obj: Iterable[int]) -> Seq:
        tokens = tuple(obj)
        if not self._is_valid_tokens(tokens):
            raise ValueError("procedure object contains invalid tokens")
        sequence = Seq()
        sequence.seq.extend(int(token) for token in tokens)
        return sequence

    def is_sane(self, g: Seq) -> bool:
        return self._is_valid_tokens(g.seq)

    def object_to_log_repr(self, g: Seq) -> str:
        return json.dumps({"tokens": list(self.graph_to_obj(g))}, separators=(",", ":"))

    def _is_valid_tokens(self, tokens: Sequence[int]) -> bool:
        return len(tokens) == self.target_length and all(
            isinstance(token, Integral) and 0 <= int(token) < len(self.alphabet) for token in tokens
        )


class FixedLengthSeqTransformerGFN(SeqTransformerGFN):
    """Apply the fixed-length action constraints to the upstream sequence model."""

    def __init__(self, env_ctx: ProcedureSequenceContext, cfg: Config):
        super().__init__(env_ctx, cfg)
        self.target_length = env_ctx.target_length

    def forward(self, xs: SeqBatch, cond: Tensor, batched: bool = False):
        action_distribution, state_predictions = super().forward(xs, cond, batched=batched)
        token_lengths = self._token_lengths(xs, batched)
        stop_mask = token_lengths.eq(self.target_length).float().unsqueeze(1)
        add_token_mask = token_lengths.lt(self.target_length).float().unsqueeze(1)
        action_distribution.action_masks = [stop_mask, add_token_mask]
        return action_distribution, state_predictions

    @staticmethod
    def _token_lengths(xs: SeqBatch, batched: bool) -> Tensor:
        if not batched:
            return xs.lens.to(xs.x.device) - 1
        return torch.cat(
            [torch.arange(int(sequence_length), device=xs.x.device) for sequence_length in xs.lens.tolist()]
        )


class ChemGymRLProcedureTask(GFNTask):
    """Evaluate complete generated procedures using the ChemGymRL bridge."""

    def __init__(self, cfg: Config, bridge: FictReactProcedureBridge | None = None) -> None:
        self.bridge = bridge if bridge is not None else FictReactProcedureBridge(seed=cfg.seed)
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(
        self,
        cond_info: Dict[str, Tensor],
        obj_props: ObjectProperties,
    ) -> LogScalar:
        log_rewards = LogScalar(obj_props[:, 0].clamp(min=1e-30).log())
        return LogScalar(self.temperature_conditional.transform(cond_info, log_rewards))

    def compute_obj_properties(
        self,
        procedures: List[tuple[int, ...]],
    ) -> Tuple[ObjectProperties, Tensor]:
        is_valid = torch.tensor([self._is_valid_procedure(procedure) for procedure in procedures], dtype=torch.bool)
        evaluations = [
            self.bridge.evaluate(procedure)
            for procedure, valid in zip(procedures, is_valid.tolist())
            if valid
        ]
        properties = torch.tensor(
            [[evaluation.positive_reward, evaluation.raw_reward] for evaluation in evaluations],
            dtype=torch.float32,
        ).reshape((-1, 2))
        return ObjectProperties(properties), is_valid

    def _is_valid_procedure(self, procedure: object) -> bool:
        if not isinstance(procedure, (tuple, list)) or len(procedure) != ACTION_DIMENSION:
            return False
        return all(
            isinstance(token, Integral) and 0 <= int(token) < self.bridge.num_token_values
            for token in procedure
        )

    def close(self) -> None:
        self.bridge.close()


class ChemGymRLProcedureTrainer(StandardOnlineTrainer):
    """Train a sequence GFlowNet against complete ChemGymRL procedures."""

    task: ChemGymRLProcedureTask
    ctx: ProcedureSequenceContext

    def __init__(
        self,
        config: Config,
        reward_scale: float = DEFAULT_REWARD_SCALE,
        control_bins: Sequence[Sequence[float]] | None = None,
        positive_replay: bool = False,
        mixed_replay: bool = False,
        replay_raw_reward_threshold: float = 0.0,
        mixed_replay_top_fraction: float = 0.1,
        mixed_replay_top_sample_fraction: float = 0.5,
        print_config: bool = True,
    ) -> None:
        if positive_replay and mixed_replay:
            raise ValueError("positive replay and mixed replay are mutually exclusive")
        self.reward_scale = reward_scale
        self.control_bins = control_bins
        self.positive_replay = positive_replay
        self.mixed_replay = mixed_replay
        self.replay_raw_reward_threshold = replay_raw_reward_threshold
        self.mixed_replay_top_fraction = mixed_replay_top_fraction
        self.mixed_replay_top_sample_fraction = mixed_replay_top_sample_fraction
        super().__init__(config, print_config=print_config)

    def set_default_hps(self, cfg: Config) -> None:
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 0
        cfg.num_validation_gen_steps = 1
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.method = "TB"
        cfg.algo.num_from_policy = 32
        cfg.algo.valid_num_from_policy = 32
        cfg.algo.max_nodes = ACTION_DIMENSION
        cfg.algo.max_len = ACTION_DIMENSION + 1
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.05
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4
        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [1.0]
        cfg.cond.temperature.num_thermometer_dim = 1

    def setup_task(self) -> None:
        bridge = FictReactProcedureBridge(
            control_bins=self.control_bins,
            seed=self.cfg.seed,
            reward_scale=self.reward_scale,
        )
        self.task = ChemGymRLProcedureTask(self.cfg, bridge)

    def setup_env_context(self) -> None:
        self.env = SeqBuildingEnv(None)
        self.ctx = ProcedureSequenceContext(
            num_token_values=self.task.bridge.num_token_values,
            num_cond_dim=self.task.num_cond_dim,
        )

    def setup_model(self) -> None:
        self.model = FixedLengthSeqTransformerGFN(self.ctx, self.cfg)

    def setup_algo(self) -> None:
        super().setup_algo()
        self.algo.model_is_autoregressive = True

    def setup(self) -> None:
        super().setup()
        if self.positive_replay:
            if not self.cfg.replay.use:
                raise ValueError("positive replay requires config.replay.use")
            self.replay_buffer = PositiveRewardReplayBuffer(
                self.cfg,
                raw_reward_threshold=self.replay_raw_reward_threshold,
            )
        elif self.mixed_replay:
            if not self.cfg.replay.use:
                raise ValueError("mixed replay requires config.replay.use")
            self.replay_buffer = PercentileMixedReplayBuffer(
                self.cfg,
                top_fraction=self.mixed_replay_top_fraction,
                top_sample_fraction=self.mixed_replay_top_sample_fraction,
            )

    def build_training_data_loader(self):
        if not (self.positive_replay or self.mixed_replay):
            return super().build_training_data_loader()

        model = self._wrap_for_mp(self.sampling_model)
        replay_buffer = self._wrap_for_mp(self.replay_buffer)
        num_from_policy = self.cfg.algo.num_from_policy
        num_from_replay = self.cfg.replay.num_from_replay or num_from_policy
        num_new_samples = self.cfg.replay.num_new_samples or num_from_policy

        source = OfflineFirstDataSource(
            self.cfg,
            self.ctx,
            self.algo,
            self.task,
            replay_buffer=replay_buffer,
        )
        source.do_sample_model(model, num_from_policy, num_new_samples)
        source.do_sample_replay(num_from_replay)
        if self.cfg.log_dir:
            source.add_sampling_hook(
                SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.ctx)
            )
        for hook in self.sampling_hooks:
            source.add_sampling_hook(hook)
        return self._make_data_loader(source)

    def train_batch(self, batch, epoch_idx: int, batch_idx: int, train_it: int):
        info = super().train_batch(batch, epoch_idx, batch_idx, train_it)
        if isinstance(
            self.replay_buffer,
            (PositiveRewardReplayBuffer, PercentileMixedReplayBuffer),
        ):
            offline_raw_rewards = batch.obj_props[: batch.num_offline, 1]
            info.update(
                {
                    "replay_size": len(self.replay_buffer),
                    "replay_retained": self.replay_buffer.total_retained,
                    "replay_raw_reward_avg": offline_raw_rewards.mean().item(),
                    "replay_positive_fraction": (offline_raw_rewards > 0).float().mean().item(),
                }
            )
        if isinstance(self.replay_buffer, PercentileMixedReplayBuffer):
            info.update(
                {
                    "replay_top_pool_size": self.replay_buffer.last_top_pool_size,
                    "replay_remainder_pool_size": self.replay_buffer.last_remainder_pool_size,
                    "replay_top_sample_count": self.replay_buffer.last_top_sample_count,
                    "replay_remainder_sample_count": self.replay_buffer.last_remainder_sample_count,
                    "replay_cutoff_raw_reward": self.replay_buffer.last_cutoff_raw_reward,
                }
            )
        return info

    def terminate(self) -> None:
        self.task.close()
        super().terminate()
