import torch


class EncoderEns(torch.nn.Module):
    __constants__ = ["_observation_size", "_embedding_size"]

    def __init__(self, observation_size: int, embedding_size: int):
        super(EncoderEns, self).__init__()
        self._observation_size = observation_size
        self._embedding_size = embedding_size

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def observation_size(self) -> int:
        return self._observation_size
