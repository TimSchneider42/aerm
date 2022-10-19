from typing import Optional, Tuple

import torch

from .ensemble_model_mdp import MDPRewardModelEnsemble
from .gaussian_mlp_ensemble import GaussianMLPEnsemble


class MDPRewardModelEnsNN(MDPRewardModelEnsemble):
    def __init__(self, state_size: int, action_size: int, hidden_size: int, ensemble_size: int,
                 num_hidden_layers: int = 2, activation_function: str = "relu", min_std_dev: float = 0.01,
                 max_std_dev: Optional[float] = None, constant_std_dev: float = 0.001,
                 create_std_dev_head: bool = True):
        super().__init__(ensemble_size, state_size, action_size)
        self._network = GaussianMLPEnsemble(
            state_size + action_size, [hidden_size] * num_hidden_layers, 1, ensemble_size,
            activation_function=activation_function, min_std_dev=min_std_dev, max_std_dev=max_std_dev,
            constant_std_dev=constant_std_dev, name="reward", create_std_dev_head=create_std_dev_head)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._network.forward(torch.cat([state, action], dim=-1))

    @torch.jit.export
    def reset_parameters(self):
        self._network.reset_parameters()

    @torch.jit.export
    def disable_learned_std_devs(self):
        self._network.std_dev_disabled = True

    @torch.jit.export
    def enable_learned_std_devs(self):
        self._network.std_dev_disabled = False
