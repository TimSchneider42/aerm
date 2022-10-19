from abc import abstractmethod
from itertools import chain
from typing import Tuple, Any, Dict, Optional

import torch

from util.normalizer import BaseNormalizer
from .ensemble_model import EnsembleModelComponent, EnsembleModel


class MDPTransitionModelEnsemble(EnsembleModelComponent):
    __constants__ = ["state_size", "action_size"]

    def __init__(self, ensemble_size: int, state_size: int, action_size: int,
                 state_limit_lower: Optional[torch.Tensor] = None,
                 state_limit_upper: Optional[torch.Tensor] = None, ):
        super(MDPTransitionModelEnsemble, self).__init__(ensemble_size)
        self.state_size = state_size
        self.action_size = action_size
        if state_limit_upper is not None:
            state_limit_lower = state_limit_lower
        else:
            state_limit_lower = torch.full((state_size,), -1e4)
        if state_limit_upper is not None:
            state_limit_upper = state_limit_upper
        else:
            state_limit_upper = torch.full((state_size,), 1e4)
        self.register_buffer("state_limit_lower", state_limit_lower)
        self.register_buffer("state_limit_upper", state_limit_upper)
        self.state_limit_lower: torch.Tensor
        self.state_limit_upper: torch.Tensor

    @abstractmethod
    def forward(self, prev_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @torch.jit.export
    def update_state_limits(self, lower: torch.Tensor, upper: torch.Tensor):
        # Perform the operation in place to avoid future problems with a divergent JIT version as much as possible
        self.state_limit_lower[:] = lower
        self.state_limit_upper[:] = upper

    @torch.jit.export
    def disable_learned_std_devs(self):
        pass

    @torch.jit.export
    def enable_learned_std_devs(self):
        pass


class MDPRewardModelEnsemble(EnsembleModelComponent):
    __constants__ = ["action_size", "state_size"]

    def __init__(self, ensemble_size: int, action_size: int, state_size: int):
        super(MDPRewardModelEnsemble, self).__init__(ensemble_size)
        self.state_size = state_size
        self.action_size = action_size

    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def disable_learned_std_devs(self):
        pass

    def enable_learned_std_devs(self):
        pass


class EnsembleModelMDP(EnsembleModel):
    def __init__(self, transition_model: MDPTransitionModelEnsemble, reward_model: MDPRewardModelEnsemble,
                 action_normalizer: BaseNormalizer, observation_normalizer: BaseNormalizer,
                 reward_normalizer: BaseNormalizer):
        self._transition_model = transition_model
        self._reward_model = reward_model
        super(EnsembleModelMDP, self).__init__(
            action_normalizer, observation_normalizer, reward_normalizer, transition_model.ensemble_size,
            transition_model.action_size, transition_model.state_size)

    def state_dict(self) -> Dict[str, Any]:
        output = super(EnsembleModelMDP, self).state_dict()
        output.update({
            "transition": self._transition_model.state_dict(),
            "reward": self._reward_model.state_dict()
        })
        return output

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, load_transition_model: bool = True,
                        load_reward_model: bool = True):
        super(EnsembleModelMDP, self).load_state_dict(state_dict, strict=strict)
        if load_transition_model:
            self._transition_model.load_state_dict(state_dict["transition"], strict=strict)
        if load_reward_model:
            self._reward_model.load_state_dict(state_dict["reward"], strict=strict)

    def parameters(self):
        return chain(
            super(EnsembleModelMDP, self).parameters(), self._transition_model.parameters(),
            self._reward_model.parameters())

    def refresh_state_limits(self):
        self._transition_model.update_state_limits(
            self.observation_normalizer.observed_lower_bound_normalized(),
            self.observation_normalizer.observed_upper_bound_normalized())

    @property
    def transition(self) -> MDPTransitionModelEnsemble:
        return self._transition_model

    @property
    def reward(self) -> MDPRewardModelEnsemble:
        return self._reward_model

    def reset_parameters(self):
        self._transition_model.reset_parameters()
        self._reward_model.reset_parameters()

    def disable_learned_std_devs(self):
        self._transition_model.disable_learned_std_devs()
        self._reward_model.disable_learned_std_devs()
        super(EnsembleModelMDP, self).disable_learned_std_devs()

    def enable_learned_std_devs(self):
        self._transition_model.enable_learned_std_devs()
        self._reward_model.enable_learned_std_devs()
        super(EnsembleModelMDP, self).enable_learned_std_devs()
