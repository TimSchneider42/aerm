from pathlib import Path
from typing import Dict, Any, Union, Optional

import torch.nn

from util.normalizer import BaseNormalizer


class EnsembleModelComponent(torch.nn.Module):
    __constants__ = ["ensemble_size"]

    def __init__(self, ensemble_size: int):
        super(EnsembleModelComponent, self).__init__()
        self.ensemble_size = ensemble_size

    def reset_parameters(self):
        pass


class EnsembleModel:
    def __init__(self, action_normalizer: BaseNormalizer, observation_normalizer: BaseNormalizer,
                 reward_normalizer: BaseNormalizer, ensemble_size: int, action_size: int, observation_size: int):
        self._action_normalizer = action_normalizer
        self._observation_normalizer = observation_normalizer
        self._reward_normalizer = reward_normalizer
        self._ensemble_size = ensemble_size
        self._action_size = action_size
        self._observation_size = observation_size
        self._std_dev_learning_enabled = True
        self.refresh_state_limits()

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    def load(self, path: Path, strict: bool = True, map_location: Optional[Union[str, torch.device]] = None):
        self.load_state_dict(torch.load(path, map_location=map_location), strict=strict)

    def parameters(self):
        return []

    def state_dict(self) -> Dict[str, Any]:
        return {
            "action_normalizer": self._action_normalizer.state_dict(),
            "observation_normalizer": self._observation_normalizer.state_dict(),
            "reward_normalizer": self._reward_normalizer.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, load_transition_model: bool = True,
                        load_reward_model: bool = True):
        self._action_normalizer.load_state_dict(state_dict["action_normalizer"], strict=strict)
        self._observation_normalizer.load_state_dict(state_dict["observation_normalizer"], strict=strict)
        self._reward_normalizer.load_state_dict(state_dict["reward_normalizer"], strict=strict)
        self.refresh_state_limits()

    def refresh_state_limits(self):
        # Implement in child class
        pass

    def reset_parameters(self):
        pass

    def disable_learned_std_devs(self):
        self._std_dev_learning_enabled = False

    def enable_learned_std_devs(self):
        self._std_dev_learning_enabled = True

    @property
    def action_normalizer(self) -> BaseNormalizer:
        return self._action_normalizer

    @property
    def observation_normalizer(self) -> BaseNormalizer:
        return self._observation_normalizer

    @property
    def reward_normalizer(self) -> BaseNormalizer:
        return self._reward_normalizer

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    @property
    def std_dev_learning_enabled(self):
        return self._std_dev_learning_enabled
