from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.colour import Color

RelationshipType = Callable[["Shape"], bool]


class Shape:
    colour: Optional[Color]
    width: int
    height: int
    position: Tuple[int, int]
    mask: NDArray[np.int16]
    num_of_coloured_pixels: int
    centre: Tuple[float, float]
    relationships: dict[str, RelationshipType]

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
        self.right_most = self.position[1] + self.width - 1
        self.left_most = self.position[1]
        self.bottom_most = self.position[0] + self.height - 1
        self.top_most = self.position[0]

        # The below calculation ensures that for a 2*2 grid with position (0,0)
        # the center would be (0.5, 0.5). For a 3*3 grid with position (0,0)
        # the center would be (1, 1)
        centre_height = self.position[0] + self.height / 2 - 0.5
        centre_width = self.position[1] + self.width / 2 - 0.5
        self.centre = (centre_height, centre_width)

        self.relationships = {
            "is_exact_match": self.is_exact_match,
            "is_above": self.is_above,
            "is_below": self.is_below,
            "is_left_of": self.is_left_of,
            "is_right_of": self.is_right_of,
            "is_inline_horizontally_above_right": self.is_inline_horizontally_above_right,  # noqa: E501
            "is_inline_horizontally_above_left": self.is_inline_horizontally_above_left,
            "is_inline_horizontally_below_right": self.is_inline_horizontally_below_right,  # noqa: E501
            "is_inline_horizontally_below_left": self.is_inline_horizontally_below_left,
            "is_inline_above_vertically": self.is_inline_above_vertically,
            "is_inline_below_vertically": self.is_inline_below_vertically,
            "is_inline_left_horizontally": self.is_inline_left_horizontally,
            "is_inline_right_horizontally": self.is_inline_right_horizontally,
            "is_mask_overlapping": self.is_mask_overlapping,
            "is_inside": self.is_inside,
            "is_same_colour": self.is_same_colour,
        }

    def __eq__(self, other):
        if isinstance(other, Shape):
            array_equal = np.array_equal(self.mask, other.mask)
            colour_equal = self.colour == other.colour
            position_equal = self.position == other.position
            return array_equal and colour_equal and position_equal
        return False

    def is_exact_match(self, other: "Shape") -> bool:
        """
        Returns True if self is an exact match of other
        """
        return self == other

    def is_above(self, other: "Shape") -> bool:
        """
        Returns True if self is above other
        """
        return self.centre[0] < other.centre[0]

    def is_below(self, other: "Shape") -> bool:
        """
        Returns True if self is below other
        """
        return self.centre[0] > other.centre[0]

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
        above_by = other.centre[0] - self.centre[0]
        right_by = self.centre[1] - other.centre[1]
        return above_by == right_by and above_by > 0

    def is_inline_horizontally_above_left(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other horizontally and above and to the left
        """
        above_by = other.centre[0] - self.centre[0]
        left_by = other.centre[1] - self.centre[1]
        return above_by == left_by and above_by > 0

    def is_inline_horizontally_below_right(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other horizontally and below and to the right
        """
        below_by = self.centre[0] - other.centre[0]
        right_by = self.centre[1] - other.centre[1]
        return below_by == right_by and below_by > 0

    def is_inline_horizontally_below_left(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other horizontally and below and to the left
        """
        below_by = self.centre[0] - other.centre[0]
        left_by = other.centre[1] - self.centre[1]
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

    def is_mask_overlapping(self, other: "Shape") -> bool:
        """
        Returns True if self's mask overlaps with other's mask
        """
        # Define the larger grid size
        max_height = max(
            self.position[0] + self.height, other.position[0] + other.height
        )
        max_width = max(self.position[1] + self.width, other.position[1] + other.width)
        grid_size = (max_height, max_width)

        # Create the larger grid initialized to zeros
        self_relative_mask = np.zeros(grid_size, dtype=bool)
        other_relative_mask = np.zeros(grid_size, dtype=bool)

        # Place self.mask onto self_relative_mask
        self_relative_mask[
            self.position[0] : self.position[0] + self.height,
            self.position[1] : self.position[1] + self.width,
        ] = self.mask

        # Place other.mask onto other_relative_mask
        other_relative_mask[
            other.position[0] : other.position[0] + other.height,
            other.position[1] : other.position[1] + other.width,
        ] = other.mask

        # Check for overlap using logical_and
        return np.any(np.logical_and(self_relative_mask, other_relative_mask))

    def is_inside(self, other: "Shape") -> bool:
        """
        Returns True if self is inside other
        """
        return (
            self.top_most >= other.top_most
            and self.bottom_most <= other.bottom_most
            and self.left_most >= other.left_most
            and self.right_most <= other.right_most
        )

    def is_same_colour(self, other: "Shape") -> bool:
        """
        Returns True if self and other have the same colour
        """
        return self.colour == other.colour
