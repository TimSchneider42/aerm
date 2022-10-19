from typing import Tuple, Any

import gym
import numpy as np

from util.event import Event


class CallbackWrapperEnv(gym.Env):
    def __init__(self, env: gym.Env):
        super(CallbackWrapperEnv, self).__init__()
        self._env = env
        self._post_step_event = Event()
        self._pre_step_event = Event()
        self._reset_event = Event()
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action) -> Tuple[np.ndarray, float, bool, Any]:
        self._pre_step_event(action)
        obs, reward, done, info = self._env.step(action)
        self._post_step_event(obs, reward, done, info)
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        self._reset_event(obs)
        return obs

    def render(self, mode="human"):
        return self._env.render(mode)

    def seed(self, seed=None):
        return self._env.seed(seed)

    def close(self):
        return self._env.close()

    @property
    def inner_env(self) -> gym.Env:
        return self._env

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def pre_step_event(self) -> Event:
        return self._pre_step_event

    @property
    def post_step_event(self) -> Event:
        return self._post_step_event

    @property
    def reset_event(self) -> Event:
        return self._reset_event
