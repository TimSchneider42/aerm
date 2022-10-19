import torch
from torch.nn import functional

from .linear_ensemble import LinearEnsemble
from .encoder_ens import EncoderEns


class SymbolicEncoderEnsNN(EncoderEns):
    def __init__(self, observation_size: int, embedding_size: int, ensemble_size: int,
                 activation_function: str = 'relu'):
        super().__init__(observation_size, embedding_size)
        self._act_fn = getattr(functional, activation_function)
        self._fc1 = LinearEnsemble(observation_size, embedding_size, ensemble_size)
        self._fc2 = LinearEnsemble(embedding_size, embedding_size, ensemble_size)
        self._fc3 = LinearEnsemble(embedding_size, embedding_size, ensemble_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self._act_fn(self._fc1(observation))
        hidden = self._act_fn(self._fc2(hidden))
        hidden = self._fc3(hidden)
        return hidden
