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
    centre: Tuple[float, float]

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

        # The below calculation ensures that for a 2*2 grid with position (0,0)
        # the center would be (0.5, 0.5). For a 3*3 grid with position (0,0)
        # the center would be (1, 1)
        centre_height = self.position[0] + self.height / 2 - 0.5
        centre_width = self.position[1] + self.width / 2 - 0.5
        self.centre = (centre_height, centre_width)

    def __eq__(self, other):
        print(type(other))
        if isinstance(other, Shape):
            array_equal = np.array_equal(self.mask, other.mask)
            colour_equal = self.colour == other.colour
            position_equal = self.position == other.position
            return array_equal and colour_equal and position_equal
        return False

    def is_above(self, other: "Shape") -> bool:
        """
        Returns True if self is above other
        """
        return self.centre[0] < other.centre[0]

    def is_below(self, other: "Shape") -> bool:
        """
        Returns True if self is below other
        """
        return self.centre[0] < other.centre[0]

    def is_left_of(self, other: "Shape") -> bool:
        """
        Returns True if self is left of other
        """
        return self.centre[1] < other.centre[1]

    def is_right_of(self, other: "Shape") -> bool:
        """
        Returns True if self is right of other
        """
        return self.centre[1] > other.centre[1]

    def is_inline_horizontally_above_right(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other horizontally and above and to the right
        """
        above_by = self.centre[0] - other.centre[0]
        right_by = other.centre[1] - self.centre[1]
        return above_by == right_by and above_by > 0

    def is_inline_horizontally_above_left(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other horizontally and above and to the left
        """
        above_by = self.centre[0] - other.centre[0]
        left_by = self.centre[1] - other.centre[1]
        return above_by == left_by and above_by > 0

    def is_inline_horizontally_below_right(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other horizontally and below and to the right
        """
        below_by = other.centre[0] - self.centre[0]
        right_by = other.centre[1] - self.centre[1]
        return below_by == right_by and below_by > 0

    def is_inline_horizontally_below_left(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other horizontally and below and to the left
        """
        below_by = other.centre[0] - self.centre[0]
        left_by = self.centre[1] - other.centre[1]
        return below_by == left_by and below_by > 0

    def is_inline_above_vertically(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other vertically and above
        """
        return self.is_above(other) and not (
            self.is_left_of(other) or self.is_right_of(other)
        )

    def is_inline_below_vertically(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other vertically and below
        """
        return self.is_below(other) and not (
            self.is_left_of(other) or self.is_right_of(other)
        )

    def is_inline_left_horizontally(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other horizontally and left
        """
        return self.is_left_of(other) and not (
            self.is_above(other) or self.is_below(other)
        )

    def is_inline_right_horizontally(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other horizontally and right
        """
        return self.is_right_of(other) and not (
            self.is_above(other) or self.is_below(other)
        )

    def is_same_colour(self, other: "Shape") -> bool:
        """
        Returns True if self and other have the same colour
        """
        return self.colour == other.colour
