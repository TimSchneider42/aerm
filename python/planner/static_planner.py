from typing import Tuple, Dict, List, Union

import torch

from .planner import Planner


class StaticPlanner(Planner):
    def __init__(self):
        super(StaticPlanner, self).__init__()
        self.__current_step = -1
        self.__action_buffer = torch.empty(0)

    @torch.jit.export
    def reset(self, new_action_buffer: torch.Tensor):
        self.__current_step = 0
        self.__action_buffer = new_action_buffer

    def forward(self, evaluation_mode: bool = False) \
            -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        self.__current_step += 1
        return self.__action_buffer[self.__current_step - 1][None], {}, []
