import math
import torch


@torch.jit.script
def log_prob_normal(loc: torch.Tensor, scale: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # compute the variance
    var = scale ** 2
    log_scale = scale.log()
    return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


def reparametrized_normal_sample(loc: torch.Tensor, scale: torch.Tensor):
    eps = torch.normal(torch.zeros_like(loc), torch.ones_like(loc))
    return loc + eps * scale


def kl_divergence_normal(mean1: torch.Tensor, std_dev1: torch.Tensor, mean2: torch.Tensor,
                         std_dev2: torch.Tensor) -> torch.Tensor:
    return torch.log(std_dev2 / std_dev1) + (std_dev1 ** 2 + (mean1 - mean2) ** 2) / (2 * std_dev2 ** 2) - 0.5
