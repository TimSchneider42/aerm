from typing import Union, Optional, Tuple, Dict

import torch

from filter import BaseFilter
from models import EnsembleModelMDP
from util.normal import log_prob_normal, reparametrized_normal_sample


class MDPFilter(BaseFilter):
    def __init__(self, model_ensemble: EnsembleModelMDP, device: Union[str, torch.device]):
        super(MDPFilter, self).__init__(device)
        self._current_state = torch.empty(0)
        self._transition_model_ens = model_ensemble.transition
        self._reward_model_ens = model_ensemble.reward
        self._model_log_weights_unnormalized = torch.empty(0)

    @torch.jit.export
    def compute_expected_information_gain_and_rewards(
            self, actions: torch.Tensor, imagined_observations_per_action_and_model: int = 1,
            mode: str = "mutual_information", compute_rewards_only: bool = False,
            mi_exclude_outer_sample_from_inner: bool = True, min_ll_per_step_and_dim: Optional[float] = None,
            ignore_model_weights: bool = False, ig_include: str = "both") \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        img = imagined_observations_per_action_and_model
        ts, bs, act_size = actions.shape
        es = self._transition_model_ens.ensemble_size
        ss = self._transition_model_ens.state_size
        # batches and imaginations are merged into one dimension
        actions_expanded = actions[None, :, :, None].expand((es, ts, bs, img, -1))
        imagined_states_lst = [self._current_state[None, :, None].expand((es, bs, img, -1))]
        imagined_reward_means_lst = []
        imagined_reward_std_devs_lst = []
        for i in range(actions.shape[0]):
            act = actions_expanded[:, i, :]
            new_state_mean, new_state_std_dev = self._transition_model_ens.forward(imagined_states_lst[-1], act)
            new_state = reparametrized_normal_sample(new_state_mean, new_state_std_dev)
            new_state_clipped = torch.clip(
                new_state, self._transition_model_ens.state_limit_lower, self._transition_model_ens.state_limit_upper)
            new_reward_mean, new_reward_std_dev = self._reward_model_ens(new_state_clipped, act)
            imagined_states_lst.append(new_state_clipped)
            imagined_reward_means_lst.append(new_reward_mean[..., 0])
            imagined_reward_std_devs_lst.append(new_reward_std_dev[..., 0])

        imagined_reward_means = torch.stack(imagined_reward_means_lst, dim=1)
        imagined_reward_std_devs = torch.stack(imagined_reward_std_devs_lst, dim=1)
        imagined_rewards = reparametrized_normal_sample(imagined_reward_means, imagined_reward_std_devs)

        # Note that this is assuming that all parameters are equally likely in the prior
        if ignore_model_weights:
            prior_model_probs_current = torch.full((es,), 1 / es, device=self._model_log_weights_unnormalized.device)
        else:
            prior_model_probs_current = torch.softmax(self._model_log_weights_unnormalized, dim=0)

        imagined_states = torch.stack(imagined_states_lst, dim=1)
        if compute_rewards_only:
            expected_measure = torch.zeros(bs, device=actions.device)
            model_log_likelihood_structured = torch.empty(0, device=actions.device)
        else:
            start_states_flat = imagined_states[:, :-1].reshape((1, es * ts * bs * img, ss))
            end_states_flat = imagined_states[:, 1:].reshape((1, es * ts * bs * img, ss))
            actions_flat = actions_expanded.reshape((1, es * ts * bs * img, act_size))
            if ig_include == "states":
                rewards_flat = None
            else:
                rewards_flat = imagined_rewards.reshape((1, es * ts * bs * img))

            model_log_likelihood_flat = self.compute_log_likelihood(
                start_states_flat, actions_flat, end_states_flat, rewards_flat,
                min_ll_per_step_and_dim=min_ll_per_step_and_dim, rewards_only=ig_include == "rewards")
            model_log_likelihood_structured = model_log_likelihood_flat.reshape((es, es, ts, bs, img))
            model_log_likelihood_future = model_log_likelihood_structured.sum(2)
            model_log_likelihood_past = self._model_log_weights_unnormalized

            model_log_likelihood = model_log_likelihood_future + model_log_likelihood_past[:, None, None, None]

            if mode == "mutual_information":
                if mi_exclude_outer_sample_from_inner:
                    min_likelihoods = model_log_likelihood.min(0)[0]
                    ll_diag = torch.diagonal(model_log_likelihood).clone().permute((2, 0, 1))
                    # Warning: this overwrites the model_log_likelihood
                    # Ensure that the large value on the diagonal does not make any numerical problems by overwriting
                    # it with a small value (it is discarded later anyway)
                    torch.diagonal(model_log_likelihood).permute((2, 0, 1))[:] = min_likelihoods
                    max_likelihoods = model_log_likelihood.max(0, keepdim=True)[0]
                    model_ll_stabilized = model_log_likelihood - max_likelihoods
                    ll_exp = torch.exp(model_ll_stabilized)
                    torch.diagonal(ll_exp)[:] = 0  # Set diagonal to zero
                    ll_logsum = max_likelihoods[0] + ll_exp.sum(0).log()
                    measure = (ll_diag - ll_logsum).mean(2)
                else:
                    measure = torch.diagonal(torch.log_softmax(model_log_likelihood, dim=0)).mean(1).T
            elif mode == "lautum_information":
                ll_normalized = torch.log_softmax(model_log_likelihood, dim=0)
                ll_normalized_weighted = (ll_normalized * prior_model_probs_current[:, None, None, None]).sum(0)
                measure = -ll_normalized_weighted.mean(-1)
            else:
                raise NotImplementedError()
            expected_measure = measure.T @ prior_model_probs_current

        info = {
            "predicted_states": imagined_states,
            "log_likelihood": model_log_likelihood_structured
        }

        return expected_measure, imagined_reward_means, imagined_reward_std_devs, prior_model_probs_current, info

    @torch.jit.export
    def reset(self, observation: torch.Tensor):
        self._current_state = observation
        self._model_log_weights_unnormalized = torch.zeros(
            self._transition_model_ens.ensemble_size, device=self._device)

    @torch.jit.export
    def step(self, action: torch.Tensor, observation: torch.Tensor, reward: Optional[torch.Tensor] = None):
        if reward is None:
            reward_reshaped = None
        else:
            reward_reshaped = reward[None]
        log_prob_per_model = self.compute_log_likelihood(
            self._current_state[None], action[None], observation[None], reward_reshaped, min_ll_per_step_and_dim=None,
            rewards_only=False)
        self._model_log_weights_unnormalized += log_prob_per_model[:, 0]
        self._model_log_weights_unnormalized -= self._model_log_weights_unnormalized.max()
        self._current_state = observation

    @torch.jit.export
    def compute_log_likelihood(
            self, start_states: torch.Tensor, actions: torch.Tensor, end_states: torch.Tensor,
            rewards: Optional[torch.Tensor] = None, min_ll_per_step_and_dim: Optional[float] = None,
            rewards_only: bool = False) \
            -> torch.Tensor:
        if rewards_only:
            assert rewards is not None
        ensemble_size, batch_size, state_size = start_states.shape
        act_size = actions.shape[-1]
        actions_reshaped = actions.reshape((ensemble_size, batch_size, 1, act_size))
        start_states_reshaped = start_states.reshape((ensemble_size, batch_size, 1, state_size))
        end_states_reshaped = end_states.reshape((ensemble_size, batch_size, 1, state_size))
        if not rewards_only:
            state_prediction_mean, state_prediction_std_dev = self._transition_model_ens(
                start_states_reshaped, actions_reshaped)
            log_prob_per_dim = log_prob_normal(
                state_prediction_mean[:, :, 0], state_prediction_std_dev[:, :, 0], end_states_reshaped[:, :, 0])
            if min_ll_per_step_and_dim is not None:
                log_prob_per_dim = torch.clamp_min(log_prob_per_dim, min_ll_per_step_and_dim)
            log_prob_trans = log_prob_per_dim.sum(-1)
        else:
            log_prob_trans = torch.zeros(1).to(self._device)
        if rewards is not None:
            rewards_reshaped = rewards.reshape((ensemble_size, batch_size))
            reward_prediction_mean, reward_prediction_std_dev = self._reward_model_ens(
                end_states_reshaped, actions_reshaped)
            # For some reason this function is 10 times slower when it is called the second time with real data
            log_prob_rewards = log_prob_normal(
                reward_prediction_mean[:, :, 0, 0], reward_prediction_std_dev[:, :, 0, 0], rewards_reshaped)
            if min_ll_per_step_and_dim is not None:
                log_prob_rewards = torch.clamp_min(log_prob_rewards, min_ll_per_step_and_dim)
        else:
            log_prob_rewards = torch.zeros(1).to(self._device)

        return log_prob_trans + log_prob_rewards

    @torch.jit.export
    def get_transition_model_ens(self):
        return self._transition_model_ens

    @torch.jit.export
    def get_reward_model_ens(self):
        return self._reward_model_ens

    @torch.jit.export
    def current_model_weights(self) -> torch.Tensor:
        return torch.softmax(self._model_log_weights_unnormalized, dim=0)

    def get_current_state_estimate(self) -> torch.Tensor:
        return self._current_state[0]
