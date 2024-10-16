import itertools
from enum import Enum
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.colour import Colour
from arc_2024.representations.helper import (
    find_smallest_indices_greater_than_q,
    remove_rows_and_cols_with_value_x,
    surrounding_coordinates,
)
from arc_2024.representations.rotatable_mask_shape import RotatableMaskShape
from arc_2024.representations.shape import Shape
from arc_2024.representations.shape_type import ShapeType


class Interpreter:
    """
    This class is responsible for interpreting the input data and
    creating a list of representations in the form of Shape objects.
    """

    class InterpretType(Enum):
        LOCAL_SEARCH = 1
        SEPERATOR = 2

    _COLOUR_GROUP_COUNT_THRESHOLD = 6  # max number of colours in groups that we track

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
        interpret_type: "Interpreter.InterpretType"

    def interpret_shapes(self) -> list[InterpretedShapes]:
        """
        This function interprets the task data and creates
        a list of representations in the form of Shape objects.
        """
        interpretations: list[Interpreter.InterpretedShapes] = []

        local_search_inputs: List[List[Shape]] = self._interpret_shapes_from_grids(
            self.inputs
        )
        local_search_outputs: List[List[Shape]] = self._interpret_shapes_from_grids(
            self.outputs
        )
        local_search_test_inputs: List[List[Shape]] = self._interpret_shapes_from_grids(
            self.test_inputs
        )

        seperator_inputs: List[List[Shape]] = self._interpret_shapes_by_seperator(
            self.inputs
        )
        seperator_outputs: List[List[Shape]] = self._interpret_shapes_by_seperator(
            self.outputs
        )
        seperator_test_inputs: List[List[Shape]] = self._interpret_shapes_by_seperator(
            self.test_inputs
        )

        # if the two interpretations are different we return both
        seperator_local_search_different = (
            self._shape_interpretations_not_subset(
                local_search_inputs, seperator_inputs
            )
            # or self._shape_interpretations_different(local_search_outputs, seperator_outputs) # noqa: E501
            or self._shape_interpretations_not_subset(
                local_search_test_inputs, seperator_test_inputs
            )
        )

        self._interpret_and_enrich_with_shape_groups(
            local_search_inputs, local_search_test_inputs
        )
        interpretations.append(
            self.InterpretedShapes(
                local_search_inputs,
                local_search_outputs,
                local_search_test_inputs,
                Interpreter.InterpretType.LOCAL_SEARCH,
            )
        )

        if seperator_local_search_different:
            self._interpret_and_enrich_with_shape_groups(
                seperator_inputs, seperator_test_inputs
            )
            interpretations.append(
                self.InterpretedShapes(
                    seperator_inputs,
                    seperator_outputs,
                    seperator_test_inputs,
                    Interpreter.InterpretType.SEPERATOR,
                )
            )

        return interpretations

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
        shape_frontier = surrounding_coordinates(
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
                surrounding_coordinates(
                    explore_j,
                    explore_k,
                    grid.shape[0],
                    grid.shape[1],
                )
            )
            shape_frontier = shape_frontier - searched_space
            mask[explore_j, explore_k] = grid[explore_j, explore_k]
            searched_space.add((explore_j, explore_k))

        position = find_smallest_indices_greater_than_q(mask, 0)
        mask = remove_rows_and_cols_with_value_x(mask, 0)

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

    @staticmethod
    def _interpret_and_enrich_with_shape_groups(
        inputs_shapes: List[List[Shape]], test_inputs_shapes: List[List[Shape]]
    ):
        # Group Colour Counts
        (
            hit_threshold,
            shapes_grouped_colour_counts,
        ) = Interpreter._get_shapes_group_colour_counts(inputs_shapes)
        (
            _,
            test_shapes_grouped_colour_counts,
        ) = Interpreter._get_shapes_group_colour_counts(test_inputs_shapes)
        Interpreter._enrich_with_group_colour_counts(
            hit_threshold, inputs_shapes, shapes_grouped_colour_counts
        )
        Interpreter._enrich_with_group_colour_counts(
            hit_threshold, test_inputs_shapes, test_shapes_grouped_colour_counts
        )

        # TODO: size group of order e.g.
        # BIGGEST-0 for biggest and SMALLEST-0 for smallest etc

        # Rotatable Mask Shapes
        Interpreter._enrich_with_rotatable_mask_shapes(
            inputs_shapes, test_inputs_shapes
        )

        # Remove groups if not in test input as no point in tracking them
        test_input_groups: set[str] = set()
        for test_shapes in test_inputs_shapes:
            for test_shape in test_shapes:
                test_input_groups.update(test_shape.shape_groups)

        for shapes in inputs_shapes:
            for shape in shapes:
                if not shape.shape_groups.issubset(test_input_groups):
                    shape.shape_groups = shape.shape_groups.intersection(
                        test_input_groups
                    )

    @staticmethod
    def _enrich_with_rotatable_mask_shapes(
        inputs_shapes: List[List[Shape]], test_inputs_shapes: List[List[Shape]]
    ):
        masks: List[NDArray[np.int16]] = []
        test_masks: List[NDArray[np.int16]] = []
        rotation_of_masks: List[NDArray[np.int16]] = []
        rotation_of_masks_in_test: List[NDArray[np.int16]] = []

        # get all the masks from the inputs
        for input_shapes in inputs_shapes:
            for input_shape in input_shapes:
                if input_shape.shape_type != ShapeType.PIXEL:
                    masks.append(input_shape.mask)

        # get all masks we see more than once
        while masks:
            mask = masks.pop()
            if Shape.is_mask_rotationally_symmetric(mask):
                continue

            non_matching_masks = [
                m for m in masks if not Shape.is_mask_rotation_of(mask, m)
            ]
            if len(non_matching_masks) < len(masks):
                rotation_of_masks.append(mask)
                masks = non_matching_masks

        # get all the masks from the test inputs
        for test_input_shapes in test_inputs_shapes:
            for test_input_shape in test_input_shapes:
                if test_input_shape.shape_type != ShapeType.PIXEL:
                    test_masks.append(test_input_shape.mask)

        # get all masks we see in test inputs
        for rot_mask in rotation_of_masks:
            for test_mask in test_masks:
                if Shape.is_mask_rotation_of(rot_mask, test_mask):
                    rotation_of_masks_in_test.append(rot_mask)
                    break

        for rot_num, rot_mask in enumerate(rotation_of_masks_in_test):
            for i in range(len(inputs_shapes)):
                for j in range(len(inputs_shapes[i])):
                    if Shape.is_mask_rotation_of(rot_mask, inputs_shapes[i][j].mask):
                        # create new shape
                        rotatable_mask_shape = RotatableMaskShape(
                            inputs_shapes[i][j].position,
                            inputs_shapes[i][j].mask,
                            rot_mask.astype(bool),
                            inputs_shapes[i][j].shape_type,
                        )

                        rotatable_mask_shape.add_group(
                            f"ROTATIONALLY_SYMMETRIC-{rot_num}"
                        )

                        inputs_shapes[i].append(rotatable_mask_shape)

        for rot_num, rot_mask in enumerate(rotation_of_masks_in_test):
            for i in range(len(test_inputs_shapes)):
                for j in range(len(test_inputs_shapes[i])):
                    if Shape.is_mask_rotation_of(
                        rot_mask, test_inputs_shapes[i][j].mask
                    ):
                        # create new shape
                        rotatable_mask_shape = RotatableMaskShape(
                            test_inputs_shapes[i][j].position,
                            test_inputs_shapes[i][j].mask,
                            rot_mask.astype(bool),
                            test_inputs_shapes[i][j].shape_type,
                        )

                        rotatable_mask_shape.add_group(
                            f"ROTATIONALLY_SYMMETRIC-{rot_num}"
                        )

                        test_inputs_shapes[i].append(rotatable_mask_shape)

    @staticmethod
    def _get_shapes_group_colour_counts(
        list_shapes: List[List[Shape]],
    ) -> tuple[bool, list[dict[Colour, int]]]:
        hit_threshold: bool = False
        shapes_grouped_colour_counts: list[dict[Colour, int]] = []
        for shapes in list_shapes:
            single_colour_shape_colours: list[Colour] = [
                shape.colour
                for shape in shapes
                if shape.shape_type == ShapeType.SINGLE_COLOUR
                and shape.colour is not None
            ]
            # sort here prior to groupby
            single_colour_shape_colours.sort(key=lambda colour: colour.value)
            group_colour_counts = {
                colour: len(list(colours))
                for colour, colours in itertools.groupby(
                    single_colour_shape_colours, key=lambda colour: colour
                )
            }
            shapes_grouped_colour_counts.append(group_colour_counts)

            if len(group_colour_counts) > 0:
                desc_ordered_counts = sorted(group_colour_counts.values(), reverse=True)
                if desc_ordered_counts[0] > Interpreter._COLOUR_GROUP_COUNT_THRESHOLD:
                    hit_threshold = True
                    break

        return hit_threshold, shapes_grouped_colour_counts

    @staticmethod
    def _enrich_with_group_colour_counts(
        hit_threshold: bool,
        list_shapes: List[List[Shape]],
        shapes_grouped_colour_counts: list[dict[Colour, int]],
    ):
        if hit_threshold:
            return

        for shapes, group_colour_counts in zip(
            list_shapes, shapes_grouped_colour_counts
        ):
            single_colour_shapes = [
                shape for shape in shapes if shape.shape_type == ShapeType.SINGLE_COLOUR
            ]
            # Create an ordered list of counts (ascending)
            asc_ordered_counts = sorted(set(group_colour_counts.values()))
            # and descending
            desc_ordered_counts = sorted(
                set(group_colour_counts.values()), reverse=True
            )

            for shape in single_colour_shapes:
                if shape.colour is None:
                    continue  # should never happen for single colour shapes
                group_colour_count = group_colour_counts[shape.colour]
                asc_colour_count_index = asc_ordered_counts.index(group_colour_count)
                desc_colour_count_index = desc_ordered_counts.index(group_colour_count)
                shape.add_group(f"GROUP_COLOUR_COUNT-{group_colour_count}")
                shape.add_group(f"GROUP_COLOUR_COUNT_ASC-{asc_colour_count_index}")
                shape.add_group(f"GROUP_COLOUR_COUNT_DESC-{desc_colour_count_index}")

    @staticmethod
    def _interpret_shapes_by_seperator(
        grids: list[NDArray[np.int16]],
    ) -> list[list[Shape]]:
        list_shapes: List[List[Shape]] = [[] for _ in grids]
        for i, grid in enumerate(grids):
            list_shapes[i].extend(
                Interpreter._interpret_individual_pixels(grid)
            )  # noqa: E501
            # need to modify function to take any colour
            # as a separator and also return the separator
            # doing this for grids will allow things like
            # 'is_above' for crossing lines
            sub_arrays_with_indices = Interpreter._split_array_on_zeros_with_indices(
                grid
            )
            for index, sub_array in sub_arrays_with_indices:
                distinct_positives = np.unique(sub_array[sub_array > 0])
                has_more_than_one_distinct_positive = len(distinct_positives) > 1
                shape_type = (
                    ShapeType.MIXED_COLOUR
                    if has_more_than_one_distinct_positive
                    else ShapeType.SINGLE_COLOUR
                )
                list_shapes[i].append(Shape(index, sub_array, shape_type))

        return list_shapes

    @staticmethod
    def _split_array_on_zeros_with_indices(
        array: NDArray[np.int16],
    ) -> list[tuple[tuple[int, int], NDArray[np.int16]]]:
        def correct_splits_indices(splits_indices: NDArray[np.int16]) -> list:
            corrected_splits_indices = set()
            previous_index = -2
            for index in np.sort(splits_indices):
                corrected_splits_indices.add(index)
                corrected_splits_indices.add(index + 1)
                if previous_index == index - 1:
                    corrected_splits_indices.remove(index)
                previous_index = index

            return sorted(list(corrected_splits_indices))

        # Find rows and columns that are completely zero
        non_zero_rows = np.any(array != 0, axis=1)
        non_zero_cols = np.any(array != 0, axis=0)

        # Split on zero rows
        row_splits_indices = np.where(~non_zero_rows)[0]
        corrected_row_splits_indices = correct_splits_indices(row_splits_indices)
        row_splits = np.split(array, corrected_row_splits_indices)

        col_splits_indices = np.where(~non_zero_cols)[0]
        corrected_col_splits_indices = correct_splits_indices(col_splits_indices)

        sub_arrays_with_indices: list[tuple[tuple[int, int], NDArray[np.int16]]] = []

        start_row = 0
        for segment in row_splits:
            if segment.size > 0:  # Ignore empty segments
                col_splits = np.split(segment, corrected_col_splits_indices, axis=1)
                start_col = 0
                for sub_array in col_splits:
                    if sub_array.size > 0 and not np.all(sub_array == 0):
                        # Record sub-array with its top-left index
                        sub_arrays_with_indices.append(
                            ((start_row, start_col), sub_array)
                        )

                    start_col += sub_array.shape[1]

            start_row += segment.shape[0]

        return sub_arrays_with_indices

    @staticmethod
    def _shape_interpretations_not_subset(
        larger_list: List[List[Shape]], smaller_list: List[List[Shape]]
    ) -> bool:
        for superset, subset in zip(larger_list, smaller_list):
            if not set(subset).issubset(set(superset)):
                return True
        return False
