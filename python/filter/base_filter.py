from abc import abstractmethod
from typing import Union, Optional, Tuple, Dict

import torch


class BaseFilter(torch.nn.Module):
    __constants__ = ["_device"]

    def __init__(self, device: Union[str, torch.device]):
        super(BaseFilter, self).__init__()
        self._device = torch.device(device)

    @abstractmethod
    def compute_expected_information_gain_and_rewards(
            self, actions: torch.Tensor, imagined_observations_per_action_and_model: int = 1,
            mode: str = "mutual_information", compute_rewards_only: bool = False,
            min_ll_per_step_and_dim: Optional[float] = None, ignore_model_weights: bool = False,
            mi_exclude_outer_sample_from_inner: bool = True, ig_include: str = "both") \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def get_current_state_estimate(self) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self, observation: torch.Tensor):
        pass

    @abstractmethod
    def step(self, action: torch.Tensor, observation: torch.Tensor, reward: Optional[torch.Tensor] = None):
        pass

    @torch.jit.export
    def get_device(self) -> torch.device:
        # properties make problems with the JIT compiler
        return self._device
