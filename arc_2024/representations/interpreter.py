from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.colour import Colour
from arc_2024.representations.shape import Shape, ShapeType


class Interpreter:
    """
    This class is responsible for interpreting the input data and
    creating a list of representations in the form of Shape objects.
    """

    inputs: List[NDArray[np.int16]]
    outputs: List[NDArray[np.int16]]
    test_inputs: List[NDArray[np.int16]]

    def __init__(
        self,
        inputs: List[NDArray[np.int16]],
        outputs: List[NDArray[np.int16]],
        test_inputs: List[NDArray[np.int16]],
    ):
        for arr in inputs + outputs + test_inputs:
            if arr.ndim != 2:
                raise ValueError(
                    "Arrays in lists inputs, outputs & test_inputs must be 2D"
                )

        self.inputs = inputs
        self.outputs = outputs
        self.test_inputs = test_inputs

    from typing import NamedTuple

    class InterpretedShapes(NamedTuple):
        inputs: List[List[Shape]]
        outputs: List[List[Shape]]
        test_inputs: List[List[Shape]]

    def interpret_shapes(self) -> InterpretedShapes:
        """
        This function interprets the task data and creates
        a list of representations in the form of Shape objects.
        """
        inputs: List[List[Shape]] = self._interpret_shapes_from_grids(self.inputs)
        outputs: List[List[Shape]] = self._interpret_shapes_from_grids(self.outputs)
        test_inputs: List[List[Shape]] = self._interpret_shapes_from_grids(
            self.test_inputs
        )

        # shape_types: self._interpret_shape_types()

        return self.InterpretedShapes(inputs, outputs, test_inputs)

    @staticmethod
    def _surrounding_coordinates(i: int, j: int, i_max: int, j_max: int) -> set:
        directions = [
            (i - 1, j - 1),
            (i - 1, j),
            (i - 1, j + 1),
            (i, j - 1),
            (i, j + 1),
            (i + 1, j - 1),
            (i + 1, j),
            (i + 1, j + 1),
        ]

        return {
            (x, y)
            for x, y in directions
            if x >= 0 and y >= 0 and x < i_max and y < j_max
        }

    @staticmethod
    def _remove_zero_rows_and_cols(arr: np.ndarray) -> np.ndarray:
        # Remove rows with all zeros
        arr = arr[~np.all(arr == 0, axis=1)]
        # Remove columns with all zeros
        arr = arr[:, ~np.all(arr == 0, axis=0)]
        return arr

    @staticmethod
    def _find_smallest_indices_greater_than_q(arr: np.ndarray, q) -> Tuple[int, int]:
        rows_greater_than_q = np.any(arr > q, axis=1)
        cols_greater_than_q = np.any(arr > q, axis=0)

        min_i = np.where(rows_greater_than_q)[0]
        min_j = np.where(cols_greater_than_q)[0]

        if min_i.size > 0 and min_j.size > 0:
            return int(min_i[0]), int(min_j[0])
        else:
            raise ValueError(f"No row and column contain '{q}'")

    @staticmethod
    def _find_shape_by_search(
        starting_j: int,
        starting_k: int,
        grid: NDArray[np.int16],
        searched_space: set,
        shape_frontier: set,
        shape_type: ShapeType,
    ) -> Optional[Shape]:
        if grid.ndim != 2:
            raise ValueError("grid Array must be 2D")

        if not (
            shape_type == ShapeType.MIXED_COLOUR
            or shape_type == ShapeType.SINGLE_COLOUR
        ):
            raise ValueError(f"Shape type not supported: {shape_type}")

        current_colour = Colour(grid[starting_j, starting_k])
        colours: set[Colour] = {current_colour}
        # we make a mask the size of the
        # whole grid and then chop it down later
        mask = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.int16)
        mask[starting_j, starting_k] = grid[starting_j, starting_k]

        # loop over entire area surrounding the shape recursively
        # if the shape is not already in the searched_space
        shape_frontier = Interpreter._surrounding_coordinates(
            starting_j, starting_k, grid.shape[0], grid.shape[1]
        )
        shape_frontier = shape_frontier - searched_space
        searched_space.add((starting_j, starting_k))

        while shape_frontier:
            explore_j, explore_k = shape_frontier.pop()
            if grid[explore_j, explore_k] == 0:
                searched_space.add((explore_j, explore_k))
                continue

            if Colour(grid[explore_j, explore_k]) != current_colour:
                if shape_type == ShapeType.SINGLE_COLOUR:
                    continue
                elif shape_type == ShapeType.MIXED_COLOUR:
                    colours.add(Colour(grid[explore_j, explore_k]))
                else:
                    raise ValueError(f"Shape type not supported: {shape_type}")

            shape_frontier = shape_frontier.union(
                Interpreter._surrounding_coordinates(
                    explore_j,
                    explore_k,
                    grid.shape[0],
                    grid.shape[1],
                )
            )
            shape_frontier = shape_frontier - searched_space
            mask[explore_j, explore_k] = grid[explore_j, explore_k]
            searched_space.add((explore_j, explore_k))

        position = Interpreter._find_smallest_indices_greater_than_q(mask, 0)
        mask = Interpreter._remove_zero_rows_and_cols(mask)

        if shape_type == ShapeType.MIXED_COLOUR and len(colours) == 1:
            return None

        return Shape(position, mask, shape_type)

    @staticmethod
    def _interpret_individual_pixels(grid: NDArray[np.int16]) -> List[Shape]:
        if grid.ndim != 2:
            raise ValueError("grid Array must be 2D")

        pixels: List[Shape] = []
        for j in range(grid.shape[0]):
            for k in range(grid.shape[1]):
                # isn't blank
                if grid[j, k] != 0:
                    pixels.append(
                        Shape((j, k), np.array([[grid[j, k]]]), ShapeType.PIXEL)
                    )

        return pixels

    @staticmethod
    def _interpret_shapes_from_grid(
        grid: NDArray[np.int16], shape_type: ShapeType
    ) -> List[Shape]:
        searched_space: set = set()
        shape_frontier: set = set()
        shapes: List[Shape] = []
        for j in range(grid.shape[0]):
            for k in range(grid.shape[1]):
                if (j, k) in searched_space:
                    continue

                # isn't blank
                if grid[j, k] == 0:
                    searched_space.add((j, k))
                    continue

                discovered_shape = Interpreter._find_shape_by_search(
                    j, k, grid, searched_space, shape_frontier, shape_type
                )
                if discovered_shape is not None:
                    shapes.append(discovered_shape)
        return shapes

    @staticmethod
    def _interpret_shapes_from_grids(
        grids: List[NDArray[np.int16]],
    ) -> List[List[Shape]]:
        for arr in grids:
            if arr.ndim != 2:
                raise ValueError("Arrays in gtids must be 2D")

        shapes: List[List[Shape]] = [[] for _ in grids]

        for i, grid in enumerate(grids):
            # first we get all the single square shapes
            shapes[i].extend(Interpreter._interpret_individual_pixels(grid))
            # then add all the blocks of the same colour
            # note might need to consider shapes that don't join diagonally
            # right now a diagonal touch is conidered joining
            shapes[i].extend(
                Interpreter._interpret_shapes_from_grid(grid, ShapeType.SINGLE_COLOUR)
            )
            # then add all the blocks of solid colour, not necessarily the same colour
            shapes[i].extend(
                Interpreter._interpret_shapes_from_grid(grid, ShapeType.MIXED_COLOUR)
            )

        return shapes
