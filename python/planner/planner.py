from abc import abstractmethod
from typing import Tuple, Dict, Any, List, Union

import torch.nn


class CostFunction(torch.nn.Module):
    def __init__(self):
        super(CostFunction, self).__init__()
        self._progress = 0.0
        self.evaluation_mode: bool = False

    @abstractmethod
    def forward(self, actions_normalized: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError()

    @torch.jit.export
    def on_step_start(self):
        pass

    @torch.jit.export
    def on_step_end(self, collected_reward: float):
        pass

    @torch.jit.export
    def on_episode_start(self):
        pass

    @torch.jit.export
    def on_episode_end(self):
        pass

    @abstractmethod
    def get_current_state_estimate(self) -> torch.Tensor:
        pass

    def custom_state_dict(self):
        return {}

    def custom_load_state_dict(self, state_dict):
        pass

    @torch.jit.export
    def update_progress(self, new_progress: float):
        self._progress = new_progress


class PlannerOptimizer:
    def optimize(self):
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


class Planner:
    @abstractmethod
    def plan(self, evaluation_mode: bool = False) \
            -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        raise NotImplementedError()

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def on_step_start(self):
        pass

    def on_step_end(self, collected_reward: float):
        pass

    def make_planner_optimizer(self) -> PlannerOptimizer:
        return PlannerOptimizer()

    def custom_state_dict(self):
        return {}

    def custom_load_state_dict(self, state_dict):
        pass

    def clear_policy_cache(self):
        pass

    @property
    @abstractmethod
    def evaluation_mode(self) -> bool:
        pass

    @evaluation_mode.setter
    @abstractmethod
    def evaluation_mode(self, value: bool):
        pass
