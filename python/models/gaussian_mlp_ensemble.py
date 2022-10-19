from typing import Optional, Tuple, Sequence

import torch
from torch.nn import functional

from .linear_ensemble import LinearEnsemble


class _DummyModule(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def reset_parameters(self):
        pass


class GaussianMLPEnsemble(torch.nn.Module):
    __constants__ = ["_min_std_dev", "_max_std_dev", "_constant_std_dev"]

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_size: int, ensemble_size: int,
                 activation_function: str = "relu", min_std_dev: float = 0.01, max_std_dev: Optional[float] = None,
                 constant_std_dev: Optional[float] = 0.001, name: str = "gaussian_mlp",
                 create_std_dev_head: bool = True):
        super(GaussianMLPEnsemble, self).__init__()
        assert max_std_dev is None or min_std_dev <= max_std_dev
        self._act_fn = getattr(functional, activation_function)
        layer_sizes = [input_size] + list(hidden_sizes)
        self._layers = torch.nn.ModuleList([
            LinearEnsemble(lp, ln, ensemble_size)
            for i, (lp, ln) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        ])
        self._output_layer_mean = LinearEnsemble(layer_sizes[-1], output_size, ensemble_size)
        if create_std_dev_head:
            self._output_layer_std_dev = LinearEnsemble(layer_sizes[-1], output_size, ensemble_size)
        else:
            self._output_layer_std_dev = _DummyModule()
        self._std_dev_head_present = create_std_dev_head
        self._min_std_dev = min_std_dev
        self._max_std_dev = max_std_dev
        self._constant_std_dev = constant_std_dev
        self.std_dev_disabled = not self._std_dev_head_present

    def reset_parameters(self):
        self._output_layer_std_dev.reset_parameters()
        self._output_layer_mean.reset_parameters()
        for l in self._layers:
            l.reset_parameters()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = input
        for l in self._layers:
            hidden = self._act_fn(l(hidden))
        output_mean = self._output_layer_mean(hidden)
        if self.std_dev_disabled:
            output_std_dev = torch.full_like(output_mean, self._constant_std_dev)
        else:
            assert self._std_dev_head_present
            std_dev_hidden = self._output_layer_std_dev(hidden)
            if self._max_std_dev is None:
                output_std_dev = torch.nn.functional.softplus(std_dev_hidden) + self._min_std_dev
            else:
                output_std_dev = self._min_std_dev + \
                                 torch.sigmoid(std_dev_hidden) * (self._max_std_dev - self._min_std_dev)
        return output_mean, output_std_dev
