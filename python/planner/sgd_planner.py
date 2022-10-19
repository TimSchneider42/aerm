from typing import Iterable, Dict, Tuple

import torch
from torch.autograd import Variable
from torch.optim.adam import Adam

from filter import BaseFilter
from util.normalizer import BaseNormalizer
from .planner import Planner, CostFunction


class SGDPlanner(Planner):
    __constants__ = ["_planning_horizon", "_optimization_iters"]

    def __init__(self, cost_function: CostFunction, filter_params: Iterable[torch.nn.Parameter],
                 action_normalizer: BaseNormalizer, min_action: torch.Tensor, max_action: torch.Tensor,
                 planning_horizon: int = 12, optimization_iters: int = 10):
        super().__init__()
        self._cost_function = cost_function
        assert min_action.shape == max_action.shape
        self._planning_horizon = planning_horizon
        self._optimization_iters = optimization_iters
        self._filter_parameters = list(filter_params)
        self._act_center = (min_action + max_action) / 2
        self._act_range = max_action - min_action
        self._action_normalizer = action_normalizer

    def plan(self, evaluation_mode: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Disable gradient computation for the filter model parameters
        for p in self._filter_parameters:
            p.requires_grad = False

        action_size, = self._act_center.shape
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        actions_var = Variable(
            torch.randn((self._planning_horizon, 1, action_size), device=self._act_center.device), requires_grad=True)
        optimizer = Adam([actions_var])

        self._cost_function.init_session()

        for _ in range(self._optimization_iters):
            optimizer.zero_grad()

            actions_normalized = torch.tanh(actions_var)
            actions_unnormalized = (actions_normalized + self._act_center[None, None]) * self._act_range[None, None] / 2
            actions_mean_var_normalized = self._action_normalizer.normalize(actions_unnormalized)

            returns, info = self._cost_function.forward(actions_mean_var_normalized, evaluation_mode=evaluation_mode)

            loss = -returns
            loss.backward()
            optimizer.step()

        # Enable gradient computation for the filter model parameters again
        for p in self._filter_parameters:
            p.requires_grad = True

        info: Dict[str, torch.Tensor] = {}

        return (torch.tanh(actions_var.detach())[:, 0] + self._act_center) * self._act_range / 2, info
