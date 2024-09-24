from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.colour import Color
from arc_2024.representations.shape import Shape


class Interpreter:
    """
    This class is responsible for interpreting the input data and
    creating a list of representations in the form of Shape objects.
    """

    inputs: NDArray[np.np.int16]
    outputs: NDArray[np.np.int16]
    test_input: NDArray[np.np.int16]

    def __init__(
        self,
        inputs: NDArray[np.np.int16],
        outputs: NDArray[np.np.int16],
        test_inputs: NDArray[np.np.int16],
    ):
        if inputs.ndim != 3:
            raise ValueError("inputs Array must be 3D")

        if outputs.ndim != 3:
            raise ValueError("outputs Array must be 3D")

        if test_inputs.ndim != 3:
            raise ValueError("test_inputs Array must be 3D")

        self.inputs = inputs
        self.outputs = outputs
        self.test_inputs = test_inputs

    from typing import NamedTuple

    class InterpretedShapes(NamedTuple):
        inputs: List[Shape]
        outputs: List[Shape]
        test_inputs: List[Shape]

    def interpret_shapes(self) -> InterpretedShapes:
        """
        This function interprets the task data and creates
        a list of representations in the form of Shape objects.
        """

        def surrounding_coordinates(i: int, j: int, i_max: int, j_max: int) -> set:
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

        def remove_zero_rows_and_cols(arr: np.ndarray) -> np.ndarray:
            # Remove rows with all zeros
            arr = arr[~np.all(arr == 0, axis=1)]
            # Remove columns with all zeros
            arr = arr[:, ~np.all(arr == 0, axis=0)]
            return arr

        def find_smallest_indices_with_q(arr: np.ndarray, q) -> Tuple[int, int]:
            rows_with_q = np.any(arr == q, axis=1)
            cols_with_q = np.any(arr == q, axis=0)

            min_i = np.where(rows_with_q)[0]
            min_j = np.where(cols_with_q)[0]

            if min_i.size > 0 and min_j.size > 0:
                return min_i[0], min_j[0]
            else:
                raise ValueError(f"No row and column contain '{q}'")

        inputs: List[Shape] = []
        outputs: List[Shape] = []
        test_inputs: List[Shape] = []

        # first we get all the single square shapes
        for i in range(self.inputs.shape[0]):
            for j in range(self.inputs.shape[1]):
                for k in range(self.inputs.shape[2]):
                    # isn't blank
                    if self.inputs[i, j, k] != 0:
                        colour = Color(self.inputs[i, j, k])
                        inputs.append(Shape(colour, (j, k), np.array([[1]])))

        # note might need to consider shapes that do join diagnally and also don't

        # then add all the blocks of the same colour
        for i in range(self.inputs.shape[0]):
            searched_space: set = set()
            shape_frontier: set = set()
            for j in range(self.inputs.shape[1]):
                for k in range(self.inputs.shape[2]):
                    # isn't blank
                    if self.inputs[i, j, k] == 0:
                        searched_space.add((j, k))
                        continue

                    current_colour = Color(self.inputs[i, j, k])
                    # we make a mask the size of the
                    # whole grid and then chop it down later
                    mask = np.zeros(
                        (self.inputs.shape[1], self.inputs.shape[2]), dtype=np.int16
                    )
                    mask[j, k] = 1

                    # loop over entire area surrounding the shape recursively
                    # if the shape is not already in the searched_space
                    shape_frontier = surrounding_coordinates(
                        j, k, self.inputs.shape[1], self.inputs.shape[2]
                    )
                    shape_frontier = shape_frontier - searched_space
                    searched_space.add((j, k))

                    while shape_frontier:
                        explore_j, explore_k = shape_frontier.pop()
                        if self.inputs[i, explore_j, explore_k] == 0:
                            searched_space.add((explore_j, explore_k))
                            continue

                        if (
                            Color(self.inputs[i, explore_j, explore_k])
                            != current_colour
                        ):
                            # we don't add different colours to the seached space
                            # as they need to be evaluated in another loop
                            continue

                        shape_frontier = shape_frontier.union(
                            surrounding_coordinates(
                                explore_j,
                                explore_k,
                                self.inputs.shape[1],
                                self.inputs.shape[2],
                            )
                        )
                        shape_frontier = shape_frontier - searched_space
                        mask[explore_j, explore_k] = 1
                        searched_space.add((explore_j, explore_k))

                    position = find_smallest_indices_with_q(mask, 1)
                    mask = remove_zero_rows_and_cols(mask)
                    inputs.append(Shape(current_colour, position, mask))
                    searched_space.add((j, k))

        # then add all the blocks of solid colour, not necessarily the same colour

        # inputs = self._interpret_input_shapes()
        # outputs = self._interpret_output_shapes()
        # test_inputs = self._interpret_test_input_shapes()

        return self.InterpretedShapes(inputs, outputs, test_inputs)
