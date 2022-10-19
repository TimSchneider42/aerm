import numpy as np
import torch

from typing import Optional, Union

from util.normalizer import MeanVarNormalizerJIT, MeanVarNormalizer
from .ensemble_model_mdp import EnsembleModelMDP
from .mdp_reward_model_ens_nn import MDPRewardModelEnsNN
from .mdp_transition_model_ens_nn import MDPTransitionModelEnsNN


def create_ensemble_model_mdp(
        symbolic: bool, ensemble_size: int, state_size: int, action_size: int, transition_hidden_size: int,
        reward_hidden_size: int, reward_hidden_layers: int = 2, transition_hidden_layers: int = 3,
        activation_function: str = "relu", transition_min_std_dev: float = 0.01,
        transition_max_std_dev: Optional[float] = None, reward_min_std_dev: float = 0.01,
        reward_max_std_dev: Optional[float] = None, transition_constant_std_dev: float = 0.001,
        reward_constant_std_dev: float = 0.001, obs_lower_bound: Optional[np.ndarray] = None,
        obs_upper_bound: Optional[np.ndarray] = None, device: Optional[Union[torch.device, str]] = None,
        no_jit: bool = False, create_std_dev_heads: bool = True) \
        -> EnsembleModelMDP:
    if symbolic:
        transition_model = MDPTransitionModelEnsNN(
            state_size, action_size, transition_hidden_size, ensemble_size, activation_function=activation_function,
            min_std_dev=transition_min_std_dev, max_std_dev=transition_max_std_dev,
            constant_std_dev=transition_constant_std_dev, num_hidden_layers=transition_hidden_layers,
            create_std_dev_head=create_std_dev_heads)
        if device is not None:
            transition_model = transition_model.to(device)
        if not no_jit:
            transition_model = torch.jit.script(transition_model)
    else:
        raise NotImplementedError()

    if obs_lower_bound is not None:
        obs_lower_bound_torch = torch.from_numpy(obs_lower_bound).float().to(device)
    else:
        obs_lower_bound_torch = None

    if obs_upper_bound is not None:
        obs_upper_bound_torch = torch.from_numpy(obs_upper_bound).float().to(device)
    else:
        obs_upper_bound_torch = None

    Norm = MeanVarNormalizerJIT if not no_jit else MeanVarNormalizer

    reward_model = MDPRewardModelEnsNN(
        action_size, state_size, reward_hidden_size, ensemble_size,
        activation_function=activation_function, min_std_dev=reward_min_std_dev,
        max_std_dev=reward_max_std_dev, constant_std_dev=reward_constant_std_dev,
        num_hidden_layers=reward_hidden_layers, create_std_dev_head=create_std_dev_heads)

    if device is not None:
        reward_model = reward_model.to(device)
    if not no_jit:
        reward_model = torch.jit.script(reward_model)

    model = EnsembleModelMDP(
        transition_model,
        reward_model,
        action_normalizer=Norm(action_size, device=device),
        observation_normalizer=Norm(
            state_size, initial_lower_bound=obs_lower_bound_torch, initial_upper_bound=obs_upper_bound_torch,
            device=device),
        reward_normalizer=Norm(1, scalar_values=True, min_std_dev=0.1, device=device))

    return model
