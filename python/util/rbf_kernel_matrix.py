import math
from typing import Optional

import torch


@torch.jit.script
def rbf_kernel_matrix(x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
    n = x.shape[0]
    pairwise_distances = torch.pdist(x)

    if h is None:
        # See Stein Variational Gradient Descent Paper
        median_distance = torch.median(pairwise_distances).detach()
        h = median_distance ** 2 / math.log(n)

    kernel_matrix = torch.ones((n, n), device=x.device)
    idx = torch.triu_indices(n, n, offset=1)
    kernel_matrix[idx[0], idx[1]] = torch.exp(-pairwise_distances / h)
    kernel_matrix[idx[1], idx[0]] = kernel_matrix[idx[0], idx[1]]
    return kernel_matrix