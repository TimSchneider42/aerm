import gym
import numpy as np

from sisyphus_env.base_env import BaseEnv


class WarmUpEnv(BaseEnv):
    def __init__(self, imitated_env: gym.Env):
        assert isinstance(imitated_env.observation_space, gym.spaces.Box)
        self.__obs_shape = imitated_env.observation_space.shape
        self.action_space = imitated_env.action_space
        self.observation_space = imitated_env.observation_space
        super(WarmUpEnv, self).__init__()

    def reset(self, *args, **kwargs):
        return np.zeros(self.__obs_shape)

    def step(self, action):
        return np.zeros(self.__obs_shape), 0.0, False, {}
