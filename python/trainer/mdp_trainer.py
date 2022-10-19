from typing import Union, Sequence, List

import torch
from torch.optim import Adam

from filter.mdp_filter import MDPFilter
from models import EnsembleModelMDP
from models.ensemble_model_mdp import MDPTransitionModelEnsemble, MDPRewardModelEnsemble
from util.normal import reparametrized_normal_sample, log_prob_normal
from .base_trainer import BaseTrainer


class MDPTrainer(BaseTrainer[MDPFilter]):
    def __init__(self, filter: MDPFilter, model: EnsembleModelMDP, time_steps_per_episode: int, batch_size: int,
                 use_stein_gradient: bool = True, training_device: Union[str, torch.device] = "cpu",
                 storage_device: Union[str, torch.device] = "cpu", replay_buffer_max_size: int = 100000,
                 multi_step_prediction_length: int = 10, use_jit: bool = True, individual_episodes: bool = True,
                 replan_interval_steps: int = 5, use_multi_step_prediction_reward: bool = False,
                 fixed_transition_model: bool = True, fixed_reward_model: bool = True,
                 train_input_state_noise: float = 0.0):
        parameters = []
        if not fixed_transition_model:
            parameters += list(filter._transition_model_ens.parameters())
        if not fixed_reward_model:
            parameters += list(filter._reward_model_ens.parameters())
        optimizer = Adam(parameters)
        if multi_step_prediction_length > 1:
            multi_step_prediction_weights = \
                [0.5] + [0.5 / (multi_step_prediction_length - 1)] * (multi_step_prediction_length - 1)
            if use_multi_step_prediction_reward:
                multi_step_prediction_weights_reward = \
                    [0.5] + [0.5 / multi_step_prediction_length] * multi_step_prediction_length
            else:
                multi_step_prediction_weights_reward = \
                    [1.0] + [0.0] * multi_step_prediction_length
        else:
            multi_step_prediction_weights = [1.0]
            multi_step_prediction_weights_reward = [1.0, 0.0]
        self._loss_computer = _MDPLossComputer(
            multi_step_prediction_weights, multi_step_prediction_weights_reward, model.transition,
            model.reward, time_steps_per_episode, device=training_device)

        if use_jit:
            self._loss_computer = torch.jit.script(self._loss_computer)
        super(MDPTrainer, self).__init__(
            filter=filter, model=model,
            time_steps_per_episode=time_steps_per_episode, batch_size=batch_size,
            optimizer=optimizer, use_stein_gradient=use_stein_gradient, training_device=training_device,
            storage_device=storage_device, replay_buffer_max_size=replay_buffer_max_size,
            individual_episodes=individual_episodes, replan_interval_steps=replan_interval_steps,
            update_action_normalizer=not fixed_transition_model and not fixed_reward_model,
            update_observation_normalizer=not fixed_transition_model and not fixed_reward_model,
            update_reward_normalizer=not fixed_reward_model, train_input_state_noise=train_input_state_noise)

    def _compute_loss(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, reward_batch: torch.Tensor) \
            -> torch.Tensor:
        return self._loss_computer.forward(obs_batch, act_batch, reward_batch)


class _MDPLossComputer(torch.nn.Module):
    def __init__(self, multi_step_prediction_weights_transition: Sequence[float],
                 multi_step_prediction_weights_reward: Sequence[float],
                 transition_model_ens: MDPTransitionModelEnsemble, reward_model_ens: MDPRewardModelEnsemble,
                 time_steps_per_episode: int, device: Union[str, torch.device] = "cpu"):
        super(_MDPLossComputer, self).__init__()
        assert len(multi_step_prediction_weights_transition) + 1 == len(multi_step_prediction_weights_reward), \
            "Multi-step prediction weights of the reward model must be one step longer than of the transition model."
        # The reasoning behind this is as follows: if we set a planning horizon of n, we evaluate the transition model
        # and the reward model each n times. However, we never evaluate the reward model on the current (real) state,
        # but always on predicted states. Hence if we want the entire planning horizon to be included in the multi-step
        # prediction loss, we need to incorporate a reward model evaluation after the final transition model evaluation.
        # If we also want to train on the actual observed data, we end up with n + 1 evaluations of the reward model.
        # Technically, we don't need to train on the real observed data as the reward model is never evaluated on it.
        transition_weights_single = torch.tensor(multi_step_prediction_weights_transition, device=device)
        reward_weights_single = torch.tensor(multi_step_prediction_weights_reward, device=device)
        msl = transition_weights_single.shape[0]
        repeats_transition = time_steps_per_episode - torch.arange(0, msl, device=device)
        repeats_reward = torch.cat([torch.tensor([time_steps_per_episode], device=device), repeats_transition])
        self._transition_weights = torch.repeat_interleave(transition_weights_single, repeats_transition)[None, :, None]
        reward_weights = torch.repeat_interleave(reward_weights_single, repeats_reward)
        self._reward_weights = reward_weights[None, :, None]
        self._transition_model_ens = transition_model_ens
        self._reward_model_ens = reward_model_ens
        self._reward_sel_indices = \
            torch.cat(
                [torch.arange(time_steps_per_episode, device=device)] +
                [torch.arange(i, time_steps_per_episode, device=device) for i in range(msl)])
        self._transition_target_sel_indices = \
            torch.cat(
                [torch.arange(i + 1, time_steps_per_episode + 1, device=device) for i in range(msl)])
        self._reward_weights_nonzero = reward_weights != 0
        self._multi_step_prediction_length = len(multi_step_prediction_weights_transition)

    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, reward_batch: torch.Tensor) \
            -> torch.Tensor:
        lower = self._transition_model_ens.state_limit_lower
        upper = self._transition_model_ens.state_limit_upper

        new_state_mean_lst: List[torch.Tensor] = []
        new_state_std_dev_lst: List[torch.Tensor] = []
        state_lst = [obs_batch]

        for i in range(self._multi_step_prediction_length):
            new_state_mean, new_state_std_dev = self._transition_model_ens.forward(
                state_lst[-1][:, :-1], act_batch[:, i:])
            new_state_mean_lst.append(new_state_mean)
            new_state_std_dev_lst.append(new_state_std_dev)
            new_states_unclipped = reparametrized_normal_sample(new_state_mean, new_state_std_dev).detach()
            new_states = torch.clamp(new_states_unclipped, lower, upper)
            state_lst.append(new_states)
        states = torch.cat(state_lst, dim=1)
        new_state_mean = torch.cat(new_state_mean_lst, dim=1)
        new_state_std_dev = torch.cat(new_state_std_dev_lst, dim=1)
        target_states = obs_batch[:, self._transition_target_sel_indices]
        state_log_prob = log_prob_normal(new_state_mean, new_state_std_dev, target_states)
        state_loss = -(state_log_prob.sum(3) * self._transition_weights).sum(1)

        input_actions = act_batch[:, self._reward_sel_indices]
        input_states = states[:, 1:]
        target_rewards = reward_batch[:, self._reward_sel_indices, :, None]
        reward_mean, reward_std_dev = self._reward_model_ens.forward(
            input_states[:, self._reward_weights_nonzero], input_actions[:, self._reward_weights_nonzero])
        reward_log_prob = log_prob_normal(reward_mean, reward_std_dev, target_rewards[:, self._reward_weights_nonzero])
        reward_loss = -(reward_log_prob[:, :, :, 0] * self._reward_weights[:, self._reward_weights_nonzero]).sum(1)
        return reward_loss + state_loss
