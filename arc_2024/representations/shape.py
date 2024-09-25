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
    num_of_coloured_pixels: int

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
        self.num_of_coloured_pixels = np.count_nonzero(mask)

    def __eq__(self, other):
        print(type(other))
        if isinstance(other, Shape):
            array_equal = np.array_equal(self.mask, other.mask)
            colour_equal = self.colour == other.colour
            position_equal = self.position == other.position
            return array_equal and colour_equal and position_equal
        return False
