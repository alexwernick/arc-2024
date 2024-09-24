from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.colour import Color


class Shape:
    colour: Optional[Color]
    width: int
    height: int
    position: Tuple[int, int]
    mask: NDArray[np.int16]

    def __init__(
        self,
        colour: Optional[Color],
        position: Tuple[int, int],
        mask: NDArray[np.int16],
    ):
        if mask.ndim != 2:
            raise ValueError("Array must be 2D")

        height, width = mask.shape

        self.colour = colour
        self.width = width
        self.height = height
        self.position = position
        self.mask = mask
