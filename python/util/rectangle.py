import numpy as np

from typing import Union, Tuple, NamedTuple


class Rectangle(NamedTuple("_RectangleFields", (("min_coords", np.ndarray), ("max_coords", np.ndarray)))):
    def intersects(self, other: "Rectangle", strict: bool = False) -> bool:
        if strict:
            return not (np.any(self.max_coords <= other.min_coords) or np.any(self.min_coords >= other.max_coords))
        else:
            return not (np.any(self.max_coords < other.min_coords) or np.any(self.min_coords > other.max_coords))

    def grow(self, amount: Union[np.ndarray, float]) -> "Rectangle":
        return Rectangle(self.min_coords - amount, self.max_coords + amount)

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.logical_and(np.all(self.min_coords <= points, axis=1), np.all(self.max_coords >= points, axis=1))

    @property
    def size(self) -> Tuple[float, float]:
        return tuple(self.max_coords - self.min_coords)
