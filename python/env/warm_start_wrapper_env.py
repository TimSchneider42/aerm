import gym
import numpy as np

from sisyphus_env.base_env import BaseEnv


class WarmStartWrapperEnv(gym.Wrapper):
    def __init__(self, env: BaseEnv):
        super(WarmStartWrapperEnv, self).__init__(env)
        self.__initial_state_buffer = None
        self.__reset_counter = None

    def set_initial_state_buffer(self, new_initial_state_buffer: np.ndarray):
        self.__reset_counter = 0
        self.__initial_state_buffer = new_initial_state_buffer

    def reset(self) -> np.ndarray:
        env = self.env
        assert isinstance(env, BaseEnv)
        self.__reset_counter += 1
        return env.reset_to_state(self.__initial_state_buffer[self.__reset_counter - 1])
