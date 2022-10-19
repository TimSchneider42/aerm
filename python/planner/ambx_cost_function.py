from typing import Tuple, Dict, Any, Optional, Union, Literal

import torch

from filter import BaseFilter
from logger import logger
from .planner import CostFunction
from util.normalizer import BaseNormalizer


class ConstantModule(torch.nn.Module):
    def __init__(self, constant: float):
        super(ConstantModule, self).__init__()
        self._constant = constant

    def forward(self, input: float):
        return self._constant


class AMBXCostFunction(CostFunction):
    __constants__ = ["_imagined_observations_per_action", "_information_method",
                     "_use_intrinsic_reward", "_use_extrinsic_reward", "_adaptive_weight_strategy",
                     "_adaptive_weight_reward_scale", "_reward_moving_average_gain", "_ignore_model_weights",
                     "_ig_include", "_mi_exclude_outer_sample_from_inner"]

    def __init__(self, filter: BaseFilter, reward_normalizer: BaseNormalizer,
                 imagined_observations_per_action: int = 10, information_method: str = "lautum_information",
                 intrinsic_clamp_bounds: Optional[Tuple[float, float]] = None,
                 extrinsic_weight_schedule: Union[float, torch.nn.Module] = 1.0,
                 min_ll_per_step_and_dim: Optional[float] = None, use_intrinsic_reward: bool = True,
                 use_extrinsic_reward: bool = True, adaptive_weight_strategy: Literal["none", "max", "avg"] = "none",
                 adaptive_weight_reward_scale: float = 1.0, reward_moving_average_gain: float = 1e-3,
                 ignore_model_weights: bool = False, mi_exclude_outer_sample_from_inner: bool = True,
                 ig_include: str = "both", device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self._imagined_observations_per_action = imagined_observations_per_action
        self._information_method = information_method
        self._reward_normalizer = reward_normalizer
        self._filter = filter
        self._intrinsic_clamp_bounds = intrinsic_clamp_bounds
        self._min_ll_per_step_and_dim = min_ll_per_step_and_dim
        if not isinstance(extrinsic_weight_schedule, torch.nn.Module):
            extrinsic_weight_schedule = ConstantModule(extrinsic_weight_schedule)
        self._extrinsic_weight_schedule = extrinsic_weight_schedule
        self._use_intrinsic_reward = use_intrinsic_reward
        self._use_extrinsic_reward = use_extrinsic_reward
        self._adaptive_weight_strategy = adaptive_weight_strategy
        self._adaptive_weight_reward_scale = adaptive_weight_reward_scale
        self._reward_moving_average_gain = reward_moving_average_gain
        self._adaptive_extrinsic_weight = torch.tensor(self._extrinsic_weight_schedule(0.0), device=device)
        self._reward_moving_average = torch.tensor(0.0, device=device)
        self._reward_max = torch.tensor(0.0, device=device)
        self._ignore_model_weights = ignore_model_weights
        self._mi_exclude_outer_sample_from_inner = mi_exclude_outer_sample_from_inner
        self._ig_include = ig_include
        assert self._use_extrinsic_reward or self._use_intrinsic_reward
        self._state_fields = [
            "_adaptive_extrinsic_weight",
            "_reward_moving_average",
            "_reward_max"
        ]

    def forward(self, actions_normalized: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        expected_information, reward_means_normalized, reward_std_devs, model_weights, filter_info = \
            self._filter.compute_expected_information_gain_and_rewards(
                actions_normalized, mode=self._information_method,
                imagined_observations_per_action_and_model=self._imagined_observations_per_action,
                compute_rewards_only=self.evaluation_mode or not self._use_intrinsic_reward,
                min_ll_per_step_and_dim=self._min_ll_per_step_and_dim, ignore_model_weights=self._ignore_model_weights,
                mi_exclude_outer_sample_from_inner=self._mi_exclude_outer_sample_from_inner,
                ig_include=self._ig_include)

        # Simply use mean reward mean
        reward_means = self._reward_normalizer.denormalize(reward_means_normalized)
        mean_reward_means = reward_means.mean(3)
        weighted_extrinsic_term = (mean_reward_means * model_weights[:, None, None]).sum(0)
        extrinsic_term = weighted_extrinsic_term.sum(0)

        if self._intrinsic_clamp_bounds is not None:
            intrinsic_term_clamped = torch.clip(
                expected_information, self._intrinsic_clamp_bounds[0], self._intrinsic_clamp_bounds[1])
        else:
            intrinsic_term_clamped = expected_information

        weight = self._get_extrinsic_weight()
        extrinsic_term_scaled = extrinsic_term * weight

        reward_components = {
            "intrinsic": intrinsic_term_clamped,
            "extrinsic": extrinsic_term_scaled
        }

        if self.evaluation_mode or not self._use_intrinsic_reward:
            return extrinsic_term, reward_components, filter_info
        else:
            return intrinsic_term_clamped + extrinsic_term_scaled, reward_components, filter_info

    def _get_extrinsic_weight(self) -> float:
        if self._use_extrinsic_reward:
            if self._adaptive_weight_strategy != "none":
                weight = self._adaptive_extrinsic_weight
            else:
                weight = torch.tensor(self._extrinsic_weight_schedule(self._progress),
                                      device=self._adaptive_extrinsic_weight.device)
        else:
            weight = 0.0
        return weight

    @torch.jit.export
    def on_step_end(self, collected_reward: float):
        if not self.evaluation_mode:
            self._reward_moving_average = (1 - self._reward_moving_average_gain) * self._reward_moving_average \
                                          + self._reward_moving_average_gain * collected_reward
            self._reward_max = torch.maximum(
                self._reward_max, torch.tensor(collected_reward, device=self._reward_max.device))

    @torch.jit.export
    def extrinsic_weight(self) -> float:
        return self._get_extrinsic_weight()

    @torch.jit.export
    def reward_moving_average(self) -> torch.Tensor:
        return self._reward_moving_average

    @torch.jit.export
    def update_weights(self):
        if self._adaptive_weight_strategy == "avg":
            self._adaptive_extrinsic_weight = self._extrinsic_weight_schedule(self._progress) \
                                              + self._adaptive_weight_reward_scale * self._reward_moving_average
        elif self._adaptive_weight_strategy == "max":
            self._adaptive_extrinsic_weight = self._extrinsic_weight_schedule(self._progress) \
                                              + self._adaptive_weight_reward_scale * self._reward_max

    @torch.jit.export
    def get_current_state_estimate(self) -> torch.Tensor:
        return self._filter.get_current_state_estimate()

    @torch.jit.ignore
    def custom_state_dict(self):
        return {f: getattr(self, f) for f in self._state_fields}

    @torch.jit.ignore
    def custom_load_state_dict(self, state_dict):
        for f in self._state_fields:
            if f in state_dict:
                setattr(self, f, state_dict[f])
            else:
                logger.warning("Key {} not found in state_dict when loading AMBXCostFunction.".format(f))
