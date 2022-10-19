import math
from typing import Optional

import torch
from torch import Tensor, baddbmm
from torch.nn import Parameter
import torch.nn.init as init


class LinearEnsemble(torch.nn.Module):
    __constants__ = ["in_features", "out_features", "ensemble_size", "init_bound_weight", "init_bound_bias"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, bias: bool = True,
                 init_bound_weight: Optional[float] = None, init_bound_bias: Optional[float] = None):
        super(LinearEnsemble, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        if init_bound_weight is None:
            gain = init.calculate_gain("leaky_relu", math.sqrt(5))
            std = gain / math.sqrt(self.in_features)
            self.init_bound_weight = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        else:
            self.init_bound_weight = init_bound_weight
        if init_bound_bias is None:
            self.init_bound_bias = 1 / math.sqrt(self.in_features)
        else:
            self.init_bound_bias = init_bound_bias
        self.weight = Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(ensemble_size, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.uniform_(-self.init_bound_weight, self.init_bound_weight)
            if self.bias is not None:
                self.bias.uniform_(-self.init_bound_bias, self.init_bound_bias)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input:   E x B x P x DI
                        where E is the ensemble size, B is the batch size, P is the particle count and D is the data
                        input dimension
        :return:        E x B x P x DO
                        where DO is the data output dimension
        """
        ensemble_size, batch_size, particle_count, _ = input.shape  # Might be using broadcasting
        input_flat = input.view((ensemble_size, batch_size * particle_count, self.in_features))
        input_flat_exp = input_flat.expand((self.ensemble_size, batch_size * particle_count, self.in_features))
        output_flat = baddbmm(self.bias, input_flat_exp, self.weight)
        output = output_flat.view((self.ensemble_size, batch_size, particle_count, self.out_features))
        # output = self.bias + input @ self.weight
        return output

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, ensemble_size={}, bias={}".format(
            self.in_features, self.out_features, self.ensemble_size, self.bias is not None)
