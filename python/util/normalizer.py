from math import prod
from typing import Union, OrderedDict, Optional

import numpy as np
import torch
from torch import Tensor


class BaseNormalizer(torch.nn.Module):
    __constants__ = ["_scalar_values"]

    def __init__(self, dimensions: int, scalar_values: bool = False, initial_lower_bound: Optional[torch.Tensor] = None,
                 initial_upper_bound: Optional[torch.Tensor] = None, device: Union[str, torch.device] = "cpu"):
        super(BaseNormalizer, self).__init__()
        if initial_lower_bound is not None:
            observed_lower_bound = initial_lower_bound.to(device)
        else:
            observed_lower_bound = torch.full((dimensions,), np.inf, device=device)
        if initial_upper_bound is not None:
            observed_upper_bound = initial_upper_bound.to(device)
        else:
            observed_upper_bound = torch.full((dimensions,), -np.inf, device=device)
        self.register_buffer("observed_lower_bound_unnormalized", observed_lower_bound)
        self.register_buffer("observed_upper_bound_unnormalized", observed_upper_bound)
        self.observed_lower_bound_unnormalized: torch.Tensor
        self.observed_upper_bound_unnormalized: torch.Tensor
        self._scalar_values = scalar_values

    def normalize(self, unnormalized_values: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def normalize_std_dev(self, unnormalized_std_dev: torch.Tensor, indices: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        raise NotImplementedError()

    def denormalize(self, normalized_values: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def denormalize_std_dev(self, normalized_std_dev: torch.Tensor, indices: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        raise NotImplementedError()

    @torch.jit.export
    def add_sample(self, unnormalized_sample: torch.Tensor):
        if self._scalar_values:
            unnormalized_sample = unnormalized_sample[..., None]
        reduction_dims = list(range(len(unnormalized_sample.shape) - 1))
        min_val = torch.amin(unnormalized_sample, dim=reduction_dims)
        max_val = torch.amax(unnormalized_sample, dim=reduction_dims)
        self.observed_lower_bound_unnormalized = torch.minimum(self.observed_lower_bound_unnormalized, min_val)
        self.observed_upper_bound_unnormalized = torch.maximum(self.observed_upper_bound_unnormalized, max_val)
        self._add_sample(unnormalized_sample)

    def _add_sample(self, unnormalized_sample: torch.Tensor):
        pass

    @torch.jit.export
    def observed_lower_bound_normalized(self) -> torch.Tensor:
        return self.normalize(self.observed_lower_bound_unnormalized, indices=None)

    @torch.jit.export
    def observed_upper_bound_normalized(self) -> torch.Tensor:
        return self.normalize(self.observed_upper_bound_unnormalized, indices=None)


class MeanVarNormalizer(BaseNormalizer):
    def __init__(self, dimensions: int, min_std_dev: float = 0.01, scalar_values: bool = False,
                 initial_lower_bound: Optional[torch.Tensor] = None,
                 initial_upper_bound: Optional[torch.Tensor] = None, device: Union[str, torch.device] = "cpu"):
        super(MeanVarNormalizer, self).__init__(
            dimensions, scalar_values, initial_lower_bound, initial_upper_bound, device=device)
        self.register_buffer("_sample_count", torch.tensor(0, device=device))
        self.register_buffer("_mean", torch.zeros(dimensions, device=device))
        self.register_buffer("_var_times_n", torch.zeros(dimensions, device=device))
        self._sample_count: torch.Tensor
        self._mean: torch.Tensor
        self._var_times_n: torch.Tensor
        self._min_std_dev = torch.tensor(min_std_dev, device=device)

    @torch.jit.export
    def normalize(self, unnormalized_values: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._process(unnormalized_values, normalize=True, use_mean=True, indices=indices)

    @torch.jit.export
    def normalize_std_dev(self, unnormalized_std_devs: torch.Tensor, indices: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        return self._process(unnormalized_std_devs, normalize=True, use_mean=False, indices=indices)

    @torch.jit.export
    def denormalize(self, normalized_values: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._process(normalized_values, normalize=False, use_mean=True, indices=indices)

    @torch.jit.export
    def denormalize_std_dev(self, normalized_std_devs: torch.Tensor, indices: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        return self._process(normalized_std_devs, normalize=False, use_mean=False, indices=indices)

    def _process(self, values: torch.Tensor, normalize: bool = True, use_mean: bool = True,
                 indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._sample_count == 0:
            return values
        if self._scalar_values:
            values = values[..., None]
        broadcast_dims = len(values.shape) - 1
        shape = (1,) * broadcast_dims + (-1,)
        if indices is None:
            mean = self._mean
            std_dev = self.std_dev
        else:
            mean = self._mean[indices]
            std_dev = self.std_dev[indices]
        if normalize:
            if use_mean:
                values = values - mean.reshape(shape)
            values = values / std_dev.reshape(shape)
        else:
            values = values * std_dev.reshape(shape)
            if use_mean:
                values = values + mean.reshape(shape)
        if self._scalar_values:
            return values[..., 0]
        else:
            return values

    @torch.jit.export
    def _add_sample(self, unnormalized_sample: torch.Tensor):
        # Use the parallel version of Welford's algorithm
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        with torch.no_grad():
            if self._scalar_values:
                unnormalized_sample = unnormalized_sample[..., None]
            sample_flat = unnormalized_sample.view((-1, unnormalized_sample.shape[-1]))
            # Leave this as is due to the JIT compiler
            new_samples = sample_flat.shape[0]
            input_mean = sample_flat.mean(0)
            input_var_times_nb = ((sample_flat - input_mean) ** 2).sum(0)
            new_count = self._sample_count + new_samples
            delta = input_mean - self._mean
            self._var_times_n += input_var_times_nb + delta ** 2 * self._sample_count * new_samples / new_count
            self._mean += new_samples / new_count * delta
            self._sample_count += new_samples

    @property
    def mean(self):
        return self._mean

    @property
    def std_dev(self):
        return torch.maximum(torch.clip(torch.sqrt(self._var_times_n / self._sample_count), 1e-6), self._min_std_dev)

    @property
    def sample_count(self):
        return self._sample_count


class IdNormalizer(BaseNormalizer):
    def normalize(self, unnormalized_values: torch.Tensor) -> torch.Tensor:
        return unnormalized_values

    def normalize_std_dev(self, unnormalized_std_dev: torch.Tensor) -> torch.Tensor:
        return unnormalized_std_dev

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        return normalized_values

    def denormalize_std_dev(self, normalized_std_dev: torch.Tensor) -> torch.Tensor:
        return normalized_std_dev


def MeanVarNormalizerJIT(
        dimensions: int, min_std_dev: float = 0.01, scalar_values: bool = False,
        initial_lower_bound: Optional[torch.Tensor] = None,
        initial_upper_bound: Optional[torch.Tensor] = None, device: Union[str, torch.device] = "cpu"):
    return torch.jit.script(
        MeanVarNormalizer(dimensions, min_std_dev, scalar_values, initial_lower_bound, initial_upper_bound,
                          device=device))
