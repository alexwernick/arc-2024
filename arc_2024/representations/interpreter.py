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
    _WHOLE_BOARD_GROUP_NAME = "WHOLE_BOARD"

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

        inputs_have_seperators, seperator_colour = self._do_inputs_have_seperators()

        if inputs_have_seperators:
            seperator_inputs: List[List[Shape]] = self._interpret_shapes_by_seperator(
                self.inputs, seperator_colour
            )
            seperator_outputs: List[List[Shape]] = [[] for _ in self.outputs]
            seperator_test_inputs: List[
                List[Shape]
            ] = self._interpret_shapes_by_seperator(self.test_inputs, seperator_colour)

            # if the two interpretations are different we return both
            seperator_local_search_different = self._shape_interpretations_not_subset(
                local_search_inputs, seperator_inputs
            ) or self._shape_interpretations_not_subset(
                local_search_test_inputs, seperator_test_inputs
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

            # We add the whole board as a shape
            Interpreter._extend_shapes_with_board_shape(grid, shapes[i])

        return shapes

    @staticmethod
    def _extend_shapes_with_board_shape(
        grid: NDArray[np.int16], shapes: List[Shape]
    ) -> None:
        if not Interpreter._has_any_colours(grid):
            return

        shape_type = Interpreter._single_or_mixed_shape(grid)
        whole_board_shape = Shape((0, 0), grid, shape_type)
        whole_board_shape.add_group(Interpreter._WHOLE_BOARD_GROUP_NAME)
        if whole_board_shape not in shapes:
            shapes.append(whole_board_shape)
        else:
            shapes[shapes.index(whole_board_shape)].add_group(
                Interpreter._WHOLE_BOARD_GROUP_NAME
            )

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

        # Size order groups
        Interpreter._enrich_with_shapes_size_ordering(inputs_shapes, test_inputs_shapes)

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
    def _enrich_with_shapes_size_ordering(
        list_shapes: List[List[Shape]],
        list_test_shapes: List[List[Shape]],
    ):
        def filter_shapes(shapes: List[Shape]) -> List[Shape]:
            return [
                shape
                for shape in shapes
                if shape.shape_type != ShapeType.PIXEL
                and not isinstance(shape, RotatableMaskShape)
                and Interpreter._WHOLE_BOARD_GROUP_NAME not in shape.shape_groups
            ]

        def get_shape_count(shapes: List[Shape]) -> int:
            return len(filter_shapes(shapes))

        def are_counts_of_shapes_same() -> bool:
            counts = set()
            for shapes in list_shapes:
                counts.add(get_shape_count(shapes))

            for shapes in list_test_shapes:
                counts.add(get_shape_count(shapes))

            return len(counts) == 1

        def assign_ordered_groups(shapes: List[Shape], counts_the_same: bool) -> None:
            # ascending
            shapes_ordered_by_size = sorted(
                filter_shapes(shapes), key=lambda s: s.num_of_coloured_pixels
            )
            sizes_without_duplicates = sorted(
                list({shape.num_of_coloured_pixels for shape in shapes_ordered_by_size})
            )
            if len(sizes_without_duplicates) <= 1:
                return

            for shape in shapes_ordered_by_size:
                index = sizes_without_duplicates.index(shape.num_of_coloured_pixels)
                if counts_the_same:
                    shape.add_group(f"ORDERED-SIZE-{index}")
                if index == 0:
                    shape.add_group("SMALLEST")
                if index == len(sizes_without_duplicates) - 1:
                    shape.add_group("BIGGEST")

        counts_the_same = are_counts_of_shapes_same()
        for shapes in list_shapes:
            assign_ordered_groups(shapes, counts_the_same)
        for shapes in list_test_shapes:
            assign_ordered_groups(shapes, counts_the_same)

    @staticmethod
    def _interpret_shapes_by_seperator_old(
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

    def _do_inputs_have_seperators(self) -> tuple[bool, int]:
        def has_seperator(input_grids, colour_value: int) -> bool:
            for grid in input_grids:
                seperator_colums = self._find_seperator_columns(grid, colour_value)
                seperator_rows = self._find_seperator_rows(grid, colour_value)

                if not seperator_colums and not seperator_rows:
                    return False
            return True

        for color in Colour:
            if has_seperator(self.inputs, color.value) and has_seperator(
                self.test_inputs, color.value
            ):
                return True, color.value

        return False, -1

    @staticmethod
    def _interpret_shapes_by_seperator(
        input_grids: list[NDArray[np.int16]], seperator_colour: int
    ) -> list[list[Shape]]:
        def get_seperators() -> List[tuple[List[Shape], List[Shape]]]:
            seperators: List[tuple[List[Shape], List[Shape]]] = []

            for grid in input_grids:
                seperator_colums = Interpreter._find_seperator_columns(
                    grid, seperator_colour
                )
                seperator_rows = Interpreter._find_seperator_rows(
                    grid, seperator_colour
                )

                vertical_seperators: List[Shape] = []
                horizontal_seperators: List[Shape] = []

                for col in seperator_colums:
                    mask = grid[:, col : col + 1].astype(np.int16)
                    vertical_seperators.append(
                        Shape((0, col), mask, Interpreter._single_or_mixed_shape(mask))
                    )

                for row in seperator_rows:
                    mask = grid[row : row + 1, :].astype(np.int16)
                    horizontal_seperators.append(
                        Shape((row, 0), mask, Interpreter._single_or_mixed_shape(mask))
                    )
                seperators.append((vertical_seperators, horizontal_seperators))

            return seperators

        seperators = get_seperators()

        seperator_inputs: List[List[Shape]] = [[] for _ in input_grids]

        for index, (vertical_seperators, horizontal_seperators) in enumerate(
            seperators
        ):
            vertical_indices: list[int] = []
            horizontal_indices: list[int] = []
            for vertical_seperator in vertical_seperators:
                vertical_indices.append(vertical_seperator.position[1])
                seperator_inputs[index].append(vertical_seperator)
            for horizontal_seperator in horizontal_seperators:
                horizontal_indices.append(horizontal_seperator.position[0])
                seperator_inputs[index].append(horizontal_seperator)

            # they should already be in order but belt and braces
            vertical_indices = sorted(vertical_indices)
            horizontal_indices = sorted(horizontal_indices)

            for vertical_indices_index in range(len(vertical_indices) + 1):
                if len(vertical_indices) == 0:
                    previous_vertical_index = -1
                    current_vertical_index = input_grids[index].shape[1]
                elif vertical_indices_index == len(vertical_indices):
                    previous_vertical_index = vertical_indices[
                        vertical_indices_index - 1
                    ]
                    current_vertical_index = input_grids[index].shape[1]
                    if previous_vertical_index == current_vertical_index - 1:
                        continue  # when the line is on the border
                elif vertical_indices[vertical_indices_index] == 0:
                    continue  # when the line is on the border
                else:
                    previous_vertical_index = (
                        vertical_indices[vertical_indices_index - 1]
                        if vertical_indices_index > 0
                        else -1
                    )
                    current_vertical_index = vertical_indices[vertical_indices_index]

                for horizontal_indices_index in range(len(horizontal_indices) + 1):
                    if len(horizontal_indices) == 0:
                        previous_horizontal_index = -1
                        current_horizontal_index = input_grids[index].shape[0]
                    elif horizontal_indices_index == len(horizontal_indices):
                        previous_horizontal_index = horizontal_indices[
                            horizontal_indices_index - 1
                        ]
                        current_horizontal_index = input_grids[index].shape[0]
                        if previous_horizontal_index == current_horizontal_index - 1:
                            continue  # when the line is on the border

                    elif horizontal_indices[horizontal_indices_index] == 0:
                        continue  # when the line is on the border
                    else:
                        previous_horizontal_index = (
                            horizontal_indices[horizontal_indices_index - 1]
                            if horizontal_indices_index > 0
                            else -1
                        )
                        current_horizontal_index = horizontal_indices[
                            horizontal_indices_index
                        ]

                    mask = input_grids[index][
                        previous_horizontal_index + 1 : current_horizontal_index,
                        previous_vertical_index + 1 : current_vertical_index,
                    ].astype(np.int16)

                    if not Interpreter._has_any_colours(mask):
                        continue

                    seperator_inputs[index].append(
                        Shape(
                            (
                                previous_horizontal_index + 1,
                                previous_vertical_index + 1,
                            ),
                            mask,
                            Interpreter._single_or_mixed_shape(mask),
                        )
                    )
        return seperator_inputs

    @staticmethod
    def _single_or_mixed_shape(mask: NDArray[np.int16]) -> ShapeType:
        distinct_positives = np.unique(mask[mask > 0])
        has_more_than_one_distinct_positive = len(distinct_positives) > 1
        return (
            ShapeType.MIXED_COLOUR
            if has_more_than_one_distinct_positive
            else ShapeType.SINGLE_COLOUR
        )

    @staticmethod
    def _has_any_colours(mask: NDArray[np.int16]) -> bool:
        distinct_positives = np.unique(mask[mask > 0])
        return len(distinct_positives) > 0

    @staticmethod
    def _find_seperator_columns(arr: NDArray[np.int16], colour_value: int) -> List[int]:
        uniform_columns = []
        for col in range(arr.shape[1]):
            if np.all(arr[:, col] == colour_value):
                uniform_columns.append(col)
        return Interpreter._uniform_to_seperators(uniform_columns)

    @staticmethod
    def _find_seperator_rows(arr: NDArray[np.int16], colour_value: int) -> List[int]:
        uniform_rows = []
        for row in range(arr.shape[0]):
            if np.all(arr[row, :] == colour_value):
                uniform_rows.append(row)

        return Interpreter._uniform_to_seperators(uniform_rows)

    @staticmethod
    def _uniform_to_seperators(uniform: List[int]) -> List[int]:
        seperators = []
        if len(uniform) <= 1:
            return uniform

        for index, row in enumerate(uniform):
            is_next_to_previous = False
            is_next_to_next = False
            if index == 0:
                is_next_to_next = uniform[index + 1] - row == 1
            elif index == len(uniform) - 1:
                is_next_to_previous = row - uniform[index - 1] == 1
            else:
                is_next_to_previous = row - uniform[index - 1] == 1
                is_next_to_next = uniform[index + 1] - row == 1

            if not is_next_to_previous and not is_next_to_next:
                seperators.append(row)

        return seperators
