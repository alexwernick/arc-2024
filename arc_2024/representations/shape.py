from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.colour import Colour
from arc_2024.representations.helper import (
    find_smallest_indices_equal_to_q,
    remove_rows_and_cols_with_value_x,
    surrounding_coordinates,
)
from arc_2024.representations.shape_type import ShapeType

RelationshipType = Callable[["Shape"], bool]


class Shape:
    colour: Optional[Colour]
    width: int
    height: int
    position: Tuple[int, int]
    mask: NDArray[np.int16]
    num_of_coloured_pixels: int
    centre: Tuple[float, float]
    shape_type: ShapeType
    colours: set[Colour]
    colour_count: int
    shape_groups: set[str]
    _internal_blank_spaces_in_mask: list[tuple[tuple[int, int], NDArray[np.int16]]]

    def __init__(
        self,
        position: Tuple[int, int],
        mask: NDArray[np.int16],
        shape_type: ShapeType,
    ):
        if mask.ndim != 2:
            raise ValueError("Array must be 2D")

        self.colour = None

        height, width = mask.shape

        unique_colours = np.unique(mask)
        unique_colours_non_zero = unique_colours[unique_colours != 0]
        colours: set[Colour] = {Colour(value) for value in unique_colours_non_zero}

        if len(colours) == 0:
            raise ValueError("Shapes must have at least one colour")

        if len(colours) == 1:
            self.colour = list(colours)[0]
            if shape_type == ShapeType.MIXED_COLOUR:
                raise ValueError("Single colour shapes must have only one colour")

        if (
            shape_type == ShapeType.SINGLE_COLOUR or shape_type == ShapeType.PIXEL
        ) and len(colours) > 1:
            raise ValueError("Single colour shapes must have only one colour")

        self.width = width
        self.height = height
        self.position = position
        self.mask = mask
        self.num_of_coloured_pixels = np.count_nonzero(mask)
        self.right_most = self.position[1] + self.width - 1
        self.left_most = self.position[1]
        self.bottom_most = self.position[0] + self.height - 1
        self.top_most = self.position[0]
        self.shape_type = shape_type

        if colours is None:
            # code should never reach here due to above validation
            raise ValueError("colours can not be None")

        self.colours = colours
        self.colour_count = len(colours)

        # The below calculation ensures that for a 2*2 grid with position (0,0)
        # the center would be (0.5, 0.5). For a 3*3 grid with position (0,0)
        # the center would be (1, 1)
        centre_height = self.position[0] + self.height / 2 - 0.5
        centre_width = self.position[1] + self.width / 2 - 0.5
        self.centre = (centre_height, centre_width)
        self.shape_groups = set()
        self._internal_blank_spaces_in_mask = self._find_internal_blank_spaces_in_mask()

    def __eq__(self, other):
        if isinstance(other, Shape):
            array_equal = np.array_equal(self.mask, other.mask)
            colours_equal = self.colours == other.colours
            position_equal = self.position == other.position
            type_equal = self.shape_type == other.shape_type
            return array_equal and colours_equal and position_equal and type_equal
        return False

    def __hash__(self):
        return hash(
            (
                self.position,
                self.mask.tobytes(),
                self.shape_type,
                tuple(sorted(self.colours, key=lambda colour: colour.value)),
            )
        )

    def __repr__(self):
        return f"Shape({self.colour}, {self.position}, {self.mask})"

    def is_exact_match(self, other: "Shape") -> bool:
        """
        Returns True if self is an exact match of other
        """
        return self == other

    def is_above(self, other: "Shape") -> bool:
        """
        Returns True if self is above other
        """
        return self.is_above_ij(other.centre[0], other.centre[1])

    def is_above_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is above i
        """
        return self.centre[0] < i

    def is_below(self, other: "Shape") -> bool:
        """
        Returns True if self is below other
        """
        return self.is_below_ij(other.centre[0], other.centre[1])

    def is_below_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is below i
        """
        return self.centre[0] > i

    def is_left_of(self, other: "Shape") -> bool:
        """
        Returns True if self is left of other
        """
        return self.is_left_of_ij(other.centre[0], other.centre[1])

    def is_left_of_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is left of j
        """
        return self.centre[1] < j

    def is_right_of(self, other: "Shape") -> bool:
        """
        Returns True if self is right of other
        """
        return self.is_right_of_ij(other.centre[0], other.centre[1])

    def is_right_of_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is right of j
        """
        return self.centre[1] > j

    def is_inline_diagonally_above_right(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other diagonally and above and to the right
        """
        return self.is_inline_diagonally_above_right_ij(
            other.centre[0], other.centre[1]
        )

    def is_inline_diagonally_above_right_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and above and to the right
        """
        above_by = i - self.centre[0]
        right_by = self.centre[1] - j
        return above_by == right_by and above_by > 0

    def is_inline_diagonally_above_left(self, other: "Shape") -> bool:
        """
        Returns True if self inline with i,j diagonally and above and to the left
        """
        return self.is_inline_diagonally_above_left_ij(other.centre[0], other.centre[1])

    def is_inline_diagonally_above_left_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and above and to the left
        """
        above_by = i - self.centre[0]
        left_by = j - self.centre[1]
        return above_by == left_by and above_by > 0

    def is_inline_diagonally_below_right(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other diagonally and below and to the right
        """
        return self.is_inline_diagonally_below_right_ij(
            other.centre[0], other.centre[1]
        )

    def is_inline_diagonally_below_right_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and below and to the right
        """
        below_by = self.centre[0] - i
        right_by = self.centre[1] - j
        return below_by == right_by and below_by > 0

    def is_inline_diagonally_below_left(self, other: "Shape") -> bool:
        """
        Returns True if self inline with other diagonally and below and to the left
        """
        return self.is_inline_diagonally_below_left_ij(other.centre[0], other.centre[1])

    def is_inline_diagonally_below_left_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and below and to the left
        """
        below_by = self.centre[0] - i
        left_by = j - self.centre[1]
        return below_by == left_by and below_by > 0

    def is_inline_above_vertically(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other vertically and above
        """
        return self.is_above(other) and not (
            self.is_left_of(other) or self.is_right_of(other)
        )

    def is_inline_above_vertically_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self is inline with i,j vertically and above
        """
        return self.is_above_ij(i, j) and not (
            self.is_left_of_ij(i, j) or self.is_right_of_ij(i, j)
        )

    def is_inline_below_vertically(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other vertically and below
        """
        return self.is_below(other) and not (
            self.is_left_of(other) or self.is_right_of(other)
        )

    def is_inline_below_vertically_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self is inline with i,j vertically and below
        """
        return self.is_below_ij(i, j) and not (
            self.is_left_of_ij(i, j) or self.is_right_of_ij(i, j)
        )

    def is_inline_left_horizontally(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other horizontally and left
        """
        return self.is_left_of(other) and not (
            self.is_above(other) or self.is_below(other)
        )

    def is_inline_left_horizontally_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self is inline with i,j horizontally and left
        """
        return self.is_left_of_ij(i, j) and not (
            self.is_above_ij(i, j) or self.is_below_ij(i, j)
        )

    def is_inline_right_horizontally(self, other: "Shape") -> bool:
        """
        Returns True if self is inline with other horizontally and right
        """
        return self.is_right_of(other) and not (
            self.is_above(other) or self.is_below(other)
        )

    def is_inline_right_horizontally_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self is inline with i,j horizontally and right
        """
        return self.is_right_of_ij(i, j) and not (
            self.is_above_ij(i, j) or self.is_below_ij(i, j)
        )

    def is_mask_overlapping(self, other: "Shape") -> bool:
        """
        Returns True if self's mask overlaps with other's mask
        """
        return self._is_mask_overlapping(
            other.height, other.width, other.position, other.mask
        )

    def is_mask_overlapping_ij(self, i: int, j: int) -> bool:
        """
        Returns True if self's mask overlaps with i,j
        """
        return self._is_mask_overlapping(1, 1, (i, j), np.array([[1]]))

    def is_mask_overlapping_and_colour_ij(self, i: int, j: int, colour: Colour) -> bool:
        """
        Returns True if self's mask overlaps with i,j and colour
        """
        if i < self.position[0]:
            return False

        if j < self.position[1]:
            return False

        if i > self.position[0] + self.mask.shape[0] - 1:
            return False

        if j > self.position[1] + self.mask.shape[1] - 1:
            return False

        i_in_mask = i - self.position[0]
        j_in_mask = j - self.position[1]

        return self.mask[i_in_mask, j_in_mask] == colour.value

    def _is_mask_overlapping(
        self,
        other_height: int,
        other_width: int,
        other_position: Tuple[int, int],
        other_mask: NDArray[np.int16],
    ) -> bool:
        """
        Returns True if masks overlap
        """
        if other_position[0] < 0 or other_position[1] < 0:
            return False

        # Define the larger grid size
        max_height = max(
            self.position[0] + self.height, other_position[0] + other_height
        )
        max_width = max(self.position[1] + self.width, other_position[1] + other_width)
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
            other_position[0] : other_position[0] + other_height,
            other_position[1] : other_position[1] + other_width,
        ] = other_mask

        # Check for overlap using logical_and
        return np.any(np.logical_and(self_relative_mask, other_relative_mask))

    def is_inside_mask(self, other: "Shape") -> bool:
        """
        Returns True if self is inside other
        """
        return (
            self.top_most >= other.top_most
            and self.bottom_most <= other.bottom_most
            and self.left_most >= other.left_most
            and self.right_most <= other.right_most
        )

    def is_ij_inside_blank_space(self, i: int, j: int) -> bool:
        for blank_space in self._internal_blank_spaces_in_mask:
            position = blank_space[0]
            mask = blank_space[1]

            if i < position[0]:
                continue

            if j < position[1]:
                continue

            if i > position[0] + mask.shape[0] - 1:
                continue

            if j > position[1] + mask.shape[1] - 1:
                continue

            i_in_mask = i - position[0]
            j_in_mask = j - position[1]

            if mask[i_in_mask, j_in_mask]:
                return True
        return False

    def is_ij_inside_mask(self, i: int, j: int) -> bool:
        """
        Returns True if i,j inside self
        """
        return (
            self.top_most <= i
            and self.bottom_most >= i
            and self.left_most <= j
            and self.right_most >= j
        )

    def is_inside_mask_not_overlapping(self, other: "Shape") -> bool:
        """
        Returns True if self is inside other and not overlapping
        """
        return self.is_inside_mask(other) and not self.is_mask_overlapping(other)

    def is_ij_inside_mask_not_overlapping(self, i: int, j: int) -> bool:
        """
        Returns True if i,j inside self and not overlapping
        """
        return self.is_ij_inside_mask(i, j) and not self.is_mask_overlapping_ij(i, j)

    def is_same_colour(self, other: "Shape") -> bool:
        """
        Returns True if self and other have the same colour
        """
        return self.is_same_colour_as(other.colour)

    def is_same_colour_as(self, colour: Optional[Colour]) -> bool:
        """
        Returns True if self has the same colour as colour
        """
        if self.colour is None:
            return False

        return self.colour == colour

    def horizontal_distance_from_center(self, other: "Shape") -> Union[int, float]:
        """
        The horizontal distance between the centers of the two shapes
        """
        return self.horizontal_distance_from_center_ij(other.centre[0], other.centre[1])

    def horizontal_distance_from_center_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The horizontal distance between the center and the point j
        """
        return abs(self.centre[1] - j)

    def vertical_distance_from_center(self, other: "Shape") -> Union[int, float]:
        """
        The vertical distance between the centers of the two shapes
        """
        return self.vertical_distance_from_center_ij(other.centre[0], other.centre[1])

    def vertical_distance_from_center_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The vertical distance between the center and the point i
        """
        return abs(self.centre[0] - i)

    def horizontal_distance_from_edge_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The horizontal distance between the edge and the point j
        """
        if Shape.is_left_of_ij(
            self, i, j
        ):  # call Shape here to make sure we don't rotate again for RotatableMaskShape
            return abs(self.right_most - j)
        elif Shape.is_right_of_ij(self, i, j):
            return abs(self.left_most - j)
        else:
            return 0

    def vertical_distance_from_edge_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The vertical distance between the edge and the point i
        """
        if Shape.is_above_ij(
            self, i, j
        ):  # call Shape here to make sure we don't rotate again for RotatableMaskShape
            return abs(self.bottom_most - i)
        elif Shape.is_below_ij(self, i, j):
            return abs(self.top_most - i)
        else:
            return 0

    def is_vertical_line(self, grid: NDArray) -> bool:
        return (
            self.height == grid.shape[0]
            and self.width == 1
            and self.num_of_coloured_pixels == self.height
            and grid.shape[1] > 2
            and self.shape_type == ShapeType.SINGLE_COLOUR
        )

    def is_horizontal_line(self, grid: NDArray) -> bool:
        return (
            self.width == grid.shape[1]
            and self.height == 1
            and self.num_of_coloured_pixels == self.width
            and grid.shape[0] > 2
            and self.shape_type == ShapeType.SINGLE_COLOUR
        )

    def all_pixels(self) -> list[Tuple[int, int]]:
        """
        Returns all pixels in the shape
        """
        pixels = []
        for i in range(self.height):
            for j in range(self.width):
                if self.mask[i, j]:
                    pixels.append((self.position[0] + i, self.position[1] + j))
        return pixels

    def add_group(self, group: str) -> None:
        """
        Adds a group to the shape
        """
        self.shape_groups.add(group)

    @staticmethod
    def is_mask_rotation_of(
        self_mask: NDArray[np.int16], other_mask: NDArray[np.int16]
    ) -> bool:
        """
        Returns if rotation of self's mask is equal to other's mask
        Does not care about colour
        """
        rot0: NDArray[np.bool_] = self_mask.astype(bool)
        rot90 = np.rot90(rot0)
        rot180 = np.rot90(rot90)
        rot270 = np.rot90(rot180)

        other_bool: NDArray[np.bool_] = other_mask.astype(bool)

        return (
            np.array_equal(rot0, other_bool)
            or np.array_equal(rot90, other_bool)
            or np.array_equal(rot180, other_bool)
            or np.array_equal(rot270, other_bool)
        )

    @staticmethod
    def is_mask_rotationally_symmetric(self_mask: NDArray[np.int16]) -> bool:
        """
        Returns if the mask is rotationally symmetric
        """
        rot0: NDArray[np.bool_] = self_mask.astype(bool)
        rot90 = np.rot90(rot0)
        return np.array_equal(rot0, rot90)

    def _find_internal_blank_spaces_in_mask(
        self,
    ) -> list[tuple[tuple[int, int], NDArray[np.int16]]]:
        searched_space: set = set()
        frontier: set = set()
        spaces: list[tuple[tuple[int, int], NDArray[np.int16]]] = []

        if self.mask.shape[0] < 3 or self.mask.shape[1] < 3:
            return []  # not possible to have hollow insides

        for j in range(self.mask.shape[0]):
            for k in range(self.mask.shape[1]):
                if (j, k) in searched_space:
                    continue

                # isn't blank
                if self.mask[j, k] != 0:
                    searched_space.add((j, k))
                    continue

                discovered_space = self._find_blank_space_by_search(
                    j, k, searched_space, frontier
                )

                if not self._is_mask_touching_side(
                    discovered_space[1], discovered_space[0]
                ):
                    spaces.append(discovered_space)
        return spaces

    def _find_blank_space_by_search(
        self, starting_j: int, starting_k: int, searched_space: set, frontier: set
    ) -> tuple[tuple[int, int], NDArray[np.int16]]:
        grid = self.mask
        if grid.ndim != 2:
            raise ValueError("grid Array must be 2D")

        # we make a mask the size of the
        # whole grid and then chop it down later
        mask = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.bool_)
        mask[starting_j, starting_k] = True

        # loop over entire area surrounding the shape recursively
        # if the shape is not already in the searched_space
        frontier = surrounding_coordinates(
            starting_j,
            starting_k,
            grid.shape[0],
            grid.shape[1],
            include_diagonals=False,
        )
        frontier = frontier - searched_space
        searched_space.add((starting_j, starting_k))

        while frontier:
            explore_j, explore_k = frontier.pop()
            if grid[explore_j, explore_k] != 0:
                searched_space.add((explore_j, explore_k))
                continue

            frontier = frontier.union(
                surrounding_coordinates(
                    explore_j,
                    explore_k,
                    grid.shape[0],
                    grid.shape[1],
                    include_diagonals=False,
                )
            )
            frontier = frontier - searched_space
            mask[explore_j, explore_k] = True
            searched_space.add((explore_j, explore_k))

        position = find_smallest_indices_equal_to_q(mask, True)
        mask = remove_rows_and_cols_with_value_x(mask, False)
        return ((position[0] + self.position[0], position[1] + self.position[1]), mask)

    def _is_mask_touching_side(
        self, mask: NDArray[np.bool_], position: tuple[int, int]
    ) -> bool:
        touching_top = position[0] == self.top_most
        touching_left = position[1] == self.left_most
        touching_bottom = position[0] + mask.shape[0] - 1 == self.bottom_most
        touching_right = position[1] + mask.shape[1] - 1 == self.right_most
        return touching_top or touching_left or touching_bottom or touching_right
