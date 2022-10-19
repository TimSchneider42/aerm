import math
from abc import ABC, abstractmethod
from os import PathLike
from time import time
from typing import Optional, Union, TypeVar, Generic, Tuple, List, NamedTuple, Dict, Any, BinaryIO, IO, Literal

import numpy as np

import gym
import torch
from torch.optim import Optimizer

from robot_gym.core.base_task import EpisodeInvalidException
from filter import BaseFilter
from logger import logger
from models import EnsembleModel
from planner import Planner
from util.event import Event
from util.rbf_kernel_matrix import rbf_kernel_matrix

FilterType = TypeVar("FilterType", bound=BaseFilter)

StepMetaInfo = NamedTuple("StepMetaInfo", (
    ("obs", np.ndarray),
    ("reward", Optional[np.ndarray]),
    ("done", bool),
    ("env_info", Optional[Dict[str, Any]]),
    ("action", Optional[np.ndarray]),
    ("planner_info", Optional[Dict[str, Any]])))


class BaseTrainer(ABC, Generic[FilterType]):
    def __init__(self, filter: FilterType, model: EnsembleModel,
                 time_steps_per_episode: int, batch_size: int, optimizer: Optional[Optimizer],
                 use_stein_gradient: bool = True, training_device: Union[str, torch.device] = "cpu",
                 storage_device: Union[str, torch.device] = "cpu", replay_buffer_max_size: int = 100000,
                 replan_interval_steps: int = 5, new_sample_weight: float = 0.2, individual_episodes: bool = True,
                 update_action_normalizer: bool = True, update_reward_normalizer: bool = True,
                 update_observation_normalizer: bool = True, train_input_state_noise: float = 0.0):
        self._use_stein_gradient = use_stein_gradient
        self._training_device = torch.device(training_device)
        self._replay_buffer_max_size = replay_buffer_max_size
        self._replay_buffer_size = 0
        self._replay_buffer_index = 0
        self._optimizer = optimizer
        self._parameters_cat = torch.cat(
            [p.reshape(model.ensemble_size, -1) for g in optimizer.param_groups for p in g["params"]], dim=1)
        self._batch_size = batch_size
        self._individual_episodes = individual_episodes
        rp_ens_size = model.ensemble_size if individual_episodes else 1
        self.replay_buffer_obs = torch.full(
            (self._replay_buffer_max_size, rp_ens_size, time_steps_per_episode + 1, model.observation_size),
            np.nan, dtype=torch.float32, device=storage_device)
        self.replay_buffer_act = torch.full(
            (self._replay_buffer_max_size, rp_ens_size, time_steps_per_episode, model.action_size),
            np.nan, dtype=torch.float32, device=storage_device)
        self.replay_buffer_reward = torch.full(
            (self._replay_buffer_max_size, rp_ens_size, time_steps_per_episode),
            np.nan, dtype=torch.float32, device=storage_device)
        self._filter = filter
        self._replan_interval_steps = replan_interval_steps
        self._pre_reset_event = Event()
        self._post_reset_event = Event()
        self._post_step_event = Event()
        self._episode_done_event = Event()
        self._episode_invalid_event = Event()
        self._collected_episodes_since_last_training = 0
        self._new_sample_min_weight = new_sample_weight
        self._model = model
        self._update_action_normalizer = update_action_normalizer
        self._update_reward_normalizer = update_reward_normalizer
        self._update_observation_normalizer = update_observation_normalizer
        self._train_input_state_noise = train_input_state_noise

    def collect_rollouts(self, env: gym.Env, planner: Optional[Planner] = None) -> List[List[StepMetaInfo]]:
        r = self._replay_buffer_index % self._replay_buffer_max_size
        meta_info = []
        for m in range(self._model.ensemble_size if self._individual_episodes else 1):
            storage_location = (r, m)
            meta_info.append(self._collect_rollout(env, planner, storage_location=storage_location))
        self._replay_buffer_size = min(self._replay_buffer_size + 1, self._replay_buffer_max_size)
        self._replay_buffer_index += 1
        self._collected_episodes_since_last_training += 1

        if self._update_reward_normalizer:
            self._model.reward_normalizer.add_sample(self.replay_buffer_reward[r].to(self._training_device))
        if self._update_action_normalizer:
            self._model.action_normalizer.add_sample(self.replay_buffer_act[r].to(self._training_device))
        if self._update_observation_normalizer:
            self._model.observation_normalizer.add_sample(self.replay_buffer_obs[r].to(self._training_device))

        self._model.refresh_state_limits()
        return meta_info

    def perform_evaluation_rollout(self, env: gym.Env, planner: Planner) -> List[StepMetaInfo]:
        return self._collect_rollout(env, planner, evaluation_mode=True)

    def _collect_rollout(self, env: gym.Env, planner: Optional[Planner] = None, evaluation_mode: bool = False,
                         storage_location: Optional[Tuple[int, int]] = None) -> List[StepMetaInfo]:
        while True:
            try:
                self._pre_reset_event()
                if planner is not None:
                    planner.evaluation_mode = evaluation_mode
                    planner.on_episode_start()
                obs = env.reset()
                meta_info = [StepMetaInfo(obs, None, False, None, None, None)]
                obs_torch = torch.from_numpy(obs).float()
                if planner is not None:
                    with torch.no_grad():
                        obs_norm = self._model.observation_normalizer.normalize(
                            obs_torch.to(self._filter.get_device())).unsqueeze(0)
                        self._filter.reset(obs_norm)
                step = 0
                if storage_location is not None:
                    r, m = storage_location
                    self.replay_buffer_obs[r, m, 0] = obs_torch.to(self.replay_buffer_obs.device)
                action_plan = []
                self._post_reset_event(obs)
                while step < self.replay_buffer_reward.shape[2]:
                    if planner is not None:
                        planner.on_step_start()
                    planner_info = None
                    if len(action_plan) == 0:
                        if planner is not None:
                            act_plan_torch, planner_info, cf_info = planner.plan()
                            action_plan = list(act_plan_torch.cpu().numpy())[:self._replan_interval_steps]
                        else:
                            action_plan.append(env.action_space.sample())
                    action = action_plan[0]
                    action_plan[0:1] = []
                    # TODO: is it okay to only clamp the action here?
                    action_clamped = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, done, env_info = env.step(action_clamped)
                    act_torch = torch.from_numpy(action).float()
                    obs_torch = torch.from_numpy(obs).float()
                    if planner is not None:
                        with torch.no_grad():
                            act_norm = self._model.action_normalizer.normalize(
                                act_torch.to(self._filter.get_device())).unsqueeze(0)
                            obs_norm = self._model.observation_normalizer.normalize(
                                obs_torch.to(self._filter.get_device())).unsqueeze(0)
                            rew_norm = self._model.reward_normalizer.normalize(
                                torch.tensor([reward]).to(self._filter.get_device()))
                            self._filter.step(act_norm, obs_norm, rew_norm)
                    if storage_location is not None:
                        r, m = storage_location
                        self.replay_buffer_obs[r, m, step + 1] = obs_torch.to(self.replay_buffer_obs.device)
                        self.replay_buffer_act[r, m, step] = act_torch.to(self.replay_buffer_act.device)
                        self.replay_buffer_reward[r, m, step] = reward
                    if planner is not None:
                        planner.on_step_end(reward)
                    self._post_step_event(obs, reward, done, env_info, action, planner_info)
                    step += 1
                    meta_info.append(StepMetaInfo(obs, reward, done, env_info, action, planner_info))
                self._episode_done_event()
                if planner is not None:
                    planner.on_episode_end()
                return meta_info
            except EpisodeInvalidException as ex:
                self._episode_invalid_event()
                logger.info("Caught EpisodeInvalidException: {}".format(ex))
            finally:
                if planner is not None:
                    planner.evaluation_mode = False

    def train_model(self, steps: int = 1, evaluate_only: bool = False,
                    mode: Literal["random_sample", "prioritize_new", "full_iteration"] = "random_sample",
                    print_losses: bool = True) \
            -> Dict[str, torch.Tensor]:
        with torch.set_grad_enabled(self._optimizer is not None and not evaluate_only):
            if self._individual_episodes:
                model_indices = torch.arange(self._model.ensemble_size, device=self.replay_buffer_act.device)[:, None]
            else:
                model_indices = torch.zeros((self._model.ensemble_size, 1), dtype=torch.long,
                                            device=self.replay_buffer_act.device)
            current_batch_size = min(self._batch_size, self._replay_buffer_size)
            used_new_episodes = min(current_batch_size // 2, self._collected_episodes_since_last_training)
            if used_new_episodes != 0 and mode == "prioritize_new":
                new_episode_indices = (self._replay_buffer_index - 1 - torch.arange(
                    self._collected_episodes_since_last_training,
                    device=self.replay_buffer_act.device)) % self._replay_buffer_max_size
                selection_weights = torch.ones((self._model.ensemble_size, self._replay_buffer_size),
                                               device=self.replay_buffer_act.device)
                selection_weights[:, new_episode_indices[:used_new_episodes]] = 0
                new_sample_weight = max(used_new_episodes / current_batch_size, self._new_sample_min_weight)
                training_weights = torch.empty((current_batch_size,), device=self._training_device)
                training_weights[:used_new_episodes] = new_sample_weight / used_new_episodes
                training_weights[used_new_episodes:] = (1 - new_sample_weight) / (
                        current_batch_size - used_new_episodes)
            else:
                training_weights = torch.ones((current_batch_size,),
                                              device=self._training_device) / current_batch_size
                selection_weights = new_episode_indices = None
                if mode == "full_iteration":
                    steps = int(math.ceil(self.replay_buffer_size / self._batch_size))

            if print_losses:
                print_steps = set(int(i) for i in np.ceil(np.linspace(0, steps - 1, 5)))
            else:
                print_steps = set()

            start_time = time()

            model_losses = torch.zeros((steps, self._model.ensemble_size), device=self._training_device)
            batch_sizes = torch.zeros((steps,), dtype=torch.int, device=self._training_device)

            for j in range(steps):
                if mode == "prioritize_new":
                    model_samples_old = torch.multinomial(
                        selection_weights, num_samples=current_batch_size - used_new_episodes)
                    model_samples_new = new_episode_indices[None, :used_new_episodes].expand(
                        (self._model.ensemble_size, -1))
                    model_samples = torch.cat([model_samples_old, model_samples_new], dim=1)
                elif mode == "full_iteration":
                    model_samples = torch.arange(
                        j * self._batch_size, min((j + 1) * self._batch_size, self._replay_buffer_size),
                        device=self.replay_buffer_act.device)
                    training_weights = torch.full(
                        (model_samples.shape[0],), 1 / model_samples.shape[0], device=self._training_device)
                else:
                    model_samples = torch.multinomial(
                        torch.ones(self._replay_buffer_size, device=self.replay_buffer_act.device),
                        num_samples=current_batch_size, replacement=False)

                act = self.replay_buffer_act[model_samples, model_indices].to(
                    self._training_device).permute((0, 2, 1, 3))
                obs = self.replay_buffer_obs[model_samples, model_indices].to(
                    self._training_device).permute((0, 2, 1, 3))
                rew = self.replay_buffer_reward[model_samples, model_indices].to(
                    self._training_device).permute((0, 2, 1))

                assert not torch.any(torch.isnan(act))
                assert not torch.any(torch.isnan(obs))
                assert not torch.any(torch.isnan(rew))

                if self._optimizer is not None:
                    self._optimizer.zero_grad()

                if self._train_input_state_noise > 0:
                    obs += torch.randn_like(obs) * self._train_input_state_noise

                act_norm = self._model.action_normalizer.normalize(act)
                obs_norm = self._model.observation_normalizer.normalize(obs)
                rew_norm = self._model.reward_normalizer.normalize(rew)

                model_loss_per_batch = self._compute_loss(obs_norm, act_norm, rew_norm)
                model_loss = (model_loss_per_batch * training_weights[None]).sum(1)

                if self._use_stein_gradient and not evaluate_only:
                    # Compute kernel matrix
                    k = rbf_kernel_matrix(self._parameters_cat)[:, :, None]
                    stein_loss_per_model_and_batch = (k.detach() * model_loss_per_batch[:, None, :] + k).mean(0)
                    loss_per_batch = stein_loss_per_model_and_batch.sum(0)
                    loss = (loss_per_batch * training_weights).sum()
                else:
                    # Mean over ensembles
                    loss = model_loss.mean()

                if self._optimizer is not None and not evaluate_only:
                    loss.backward()
                    self._optimizer.step()

                if j in print_steps:
                    time_delta = time() - start_time
                    logger.info("Step {: 5d}: loss = {:0.4f} ({} steps/s)".format(
                        j, model_loss.mean().item(), (j + 1) / time_delta))

                model_losses[j] = model_loss.detach()
                batch_sizes[j] = model_samples.shape[0]
            self._collected_episodes_since_last_training = 0

            return {
                "model_losses": model_losses,
                "steps_per_s": torch.tensor(steps / (time() - start_time), device=self._training_device),
                "batch_sizes": batch_sizes
            }

    @abstractmethod
    def _compute_loss(
            self, obs_batch: torch.Tensor, act_batch: torch.Tensor, reward_batch: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the given batch.
        The symbols in the dimensions denote the following:
            - T: number of time steps of an episode
            - E: number of models in an ensemble
            - B: batch size
            - O: observation dimensions
            - A: action dimensions

        :param obs_batch:       E x T x B x O tensor of observations.
        :param act_batch:       E x T x B x A tensor of actions.
        :param reward_batch:    E x T x B tensor of rewards.
        :return: E x B tensor of losses.
        """
        pass

    @property
    def pre_reset_event(self) -> Event:
        return self._pre_reset_event

    @property
    def post_reset_event(self) -> Event:
        return self._post_reset_event

    @property
    def post_step_event(self) -> Event:
        return self._post_step_event

    @property
    def episode_done_event(self) -> Event:
        return self._episode_done_event

    @property
    def episode_invalid_event(self) -> Event:
        return self._episode_invalid_event

    def save_replay_buffer(self, f: Union[str, PathLike, BinaryIO, IO[bytes]]):
        state = {
            "obs": self.replay_buffer_obs[:self._replay_buffer_size],
            "act": self.replay_buffer_act[:self._replay_buffer_size],
            "reward": self.replay_buffer_reward[:self._replay_buffer_size],
            "index": self._replay_buffer_index
        }
        torch.save(state, f)

    def load_replay_buffer(self, f: Union[str, PathLike, BinaryIO, IO[bytes]]):
        state = torch.load(f, map_location=self.replay_buffer_act.device)
        replay_buffer_size = state["obs"].shape[0]
        assert replay_buffer_size <= self._replay_buffer_max_size
        self.replay_buffer_obs[:replay_buffer_size] = state["obs"]
        self.replay_buffer_act[:replay_buffer_size] = state["act"]
        self.replay_buffer_reward[:replay_buffer_size] = state["reward"]
        self._replay_buffer_size = replay_buffer_size
        if "index" in state:
            self._replay_buffer_index = state["index"]
        else:
            self._replay_buffer_index = replay_buffer_size % self._replay_buffer_max_size

    def state_dict(self):
        if self._optimizer is not None:
            return self._optimizer.state_dict()
        else:
            return {}

    def load_state_dict(self, state_dict):
        if self._optimizer is not None:
            try:
                self._optimizer.load_state_dict(state_dict)
            except ValueError:
                logger.warning("Failed to load optimizer state.")

    @property
    def replay_buffer_size(self) -> int:
        return self._replay_buffer_size

    def clear_replay_buffer(self):
        self._replay_buffer_index = self._replay_buffer_size = 0
