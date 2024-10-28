from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, DefaultDict, List, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from arc_2024.inductive_logic_programming.first_order_logic import (
    ArgType,
    Literal,
    Predicate,
    RuleBasedPredicate,
    Variable,
)
from arc_2024.inductive_logic_programming.FOIL import FOIL
from arc_2024.representations.colour import Colour
from arc_2024.representations.rotatable_mask_shape import RotatableMaskShape
from arc_2024.representations.shape import Shape
from arc_2024.representations.shape_type import ShapeType


class Solver:
    inputs: list[NDArray[np.int16]]
    outputs: list[NDArray[np.int16]]
    test_inputs: list[NDArray[np.int16]]
    empty_test_outputs: list[NDArray[np.int16]]
    inputs_shapes: list[list[Shape]]
    outputs_shapes: list[list[Shape]]
    test_inputs_shapes: list[list[Shape]]

    _SHAPE_COUNT_MAX_FOR_SHAPE_SHAPE_PREDS = 20

    class ArgTypes(NamedTuple):
        colour_type_arg: ArgType
        example_number_arg: ArgType
        i_arg: ArgType
        j_arg: ArgType
        shape_arg: ArgType
        number_value_arg: ArgType

    class Variables(NamedTuple):
        v1: Variable
        v2: Variable
        v3: Variable
        v4: Variable

    class Predicates(NamedTuple):
        # input_pred: Predicate
        empty_pred: Predicate

        # positional predicates with shape
        above_pred: Predicate
        below_pred: Predicate
        left_of_pred: Predicate
        right_of_pred: Predicate

        inline_diagonally_right_left_pred: Predicate
        inline_diagonally_above_right_pred: Predicate
        inline_diagonally_below_left_pred: Predicate

        inline_diagonally_left_right_pred: Predicate
        inline_diagonally_above_left_pred: Predicate
        inline_diagonally_below_right_pred: Predicate

        inline_vertically_pred: Predicate
        inline_above_vertically_pred: Predicate
        inline_below_vertically_pred: Predicate

        inline_horizontally_pred: Predicate
        inline_left_horizontally_pred: Predicate
        inline_right_horizontally_pred: Predicate

        # mask related predicates
        inside_mask_not_overlapping_pred: Predicate
        inside_blank_space_pred: Predicate
        mask_overlapping_and_colour_pred: Predicate
        mask_overlapping_pred: Predicate
        mask_overlapping_and_colour_expanded_to_grid_pred: Predicate
        mask_overlapping_expanded_to_grid_pred: Predicate
        mask_overlapping_and_colour_repeated_grid_pred: Predicate
        mask_overlapping_repeated_grid_pred: Predicate
        inside_mask_not_overlapping_moved_to_grid_pred: Predicate  # noqa: E501
        mask_overlapping_moved_to_grid_pred: Predicate

        # distance predicates
        vertical_center_distance_more_than_pred: Predicate
        vertical_center_distance_less_than_pred: Predicate
        horizontal_center_distance_more_than_pred: Predicate
        horizontal_center_distance_less_than_pred: Predicate
        vertical_edge_distance_more_than_pred: Predicate
        vertical_edge_distance_less_than_pred: Predicate
        horizontal_edge_distance_more_than_pred: Predicate
        horizontal_edge_distance_less_than_pred: Predicate
        vertical_grid_top_distance_more_than_pred: Predicate
        vertical_grid_bottom_distance_more_than_pred: Predicate
        vertical_grid_top_distance_less_than_pred: Predicate
        vertical_grid_bottom_distance_less_than_pred: Predicate
        horizontal_grid_left_distance_more_than_pred: Predicate
        horizontal_grid_right_distance_more_than_pred: Predicate
        horizontal_grid_left_distance_less_than_pred: Predicate
        horizontal_grid_right_distance_less_than_pred: Predicate
        number_value_predicates: list[RuleBasedPredicate]

        # positional predicates with 2 shapes
        mask_overlapping_top_inline_top_pred: Predicate
        mask_overlapping_bot_inline_bot_pred: Predicate
        mask_overlapping_left_inline_left_pred: Predicate
        mask_overlapping_right_inline_right_pred: Predicate
        mask_overlapping_bot_top_touching_pred: Predicate
        mask_overlapping_top_bot_touching_pred: Predicate
        mask_overlapping_right_left_touching_pred: Predicate
        mask_overlapping_left_right_touching_pred: Predicate
        mask_overlapping_gravity_to_shape_pred: Predicate

        # position in relation to grid
        top_left_bottom_right_diag_pred: Predicate
        bottom_left_top_right_diag_pred: Predicate
        touching_grid_edge_pred: Predicate

        # shape predicates
        shape_colour_predicates: list[Predicate]
        colour_predicates: list[RuleBasedPredicate]
        shape_size_predicates: list[Predicate]
        shape_colour_count_predicates: list[Predicate]
        shape_group_predicates: list[Predicate]
        shape_colour_pred: Predicate
        shape_spanning_line_pred: Predicate
        shape_vertical_line_pred: Predicate
        shape_horizontal_line_pred: Predicate

        # grid predicates
        grid_colour_count_predicates: list[Predicate]

        def to_list(self) -> list[Predicate]:
            flattened: list[Predicate] = []
            for value in self:
                if isinstance(value, Predicate):
                    flattened.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Predicate):
                            flattened.append(item)
            return flattened

    BackgroundKnowledgeType = DefaultDict[str, set[tuple]]

    # We offset test numbers by 100 to avoid conflicts with the examples
    _TEST_EX_NUMBER_OFFSET = 100

    def __init__(
        self,
        inputs: list[NDArray[np.int16]],
        outputs: list[NDArray[np.int16]],
        test_inputs: list[NDArray[np.int16]],
        empty_test_outputs: list[NDArray[np.int16]],
        inputs_shapes: list[list[Shape]],
        outputs_shapes: list[list[Shape]],
        test_inputs_shapes: list[list[Shape]],
    ):
        for arr in inputs + outputs + test_inputs:
            if arr.ndim != 2:
                raise ValueError(
                    "Arrays in lists inputs, outputs & test_inputs must be 2D"
                )

        self.inputs = inputs
        self.outputs = outputs
        self.test_inputs = test_inputs
        self.empty_test_outputs = empty_test_outputs
        self.inputs_shapes = inputs_shapes
        self.outputs_shapes = outputs_shapes
        self.test_inputs_shapes = test_inputs_shapes

    def solve(
        self, beam_width: int = 1, max_clause_length: int = 4, timeout_seconds: int = 60
    ) -> list[NDArray[np.int16]]:
        """
        This function solves the task.
        """
        # We prepare data for the FOIL algorithm
        possible_colours: list[Colour] = self._extract_all_possible_colours(
            self.inputs, self.outputs, self.test_inputs
        )  # noqa: E501

        arg_types = self._create_args_types(possible_colours)
        variables = self._create_variables(arg_types)
        target_literal = self._create_target_literal(arg_types, variables)
        predicates = self._create_predicates(arg_types, possible_colours)
        self._add_predicate_relations(predicates)
        predicate_list = predicates.to_list()
        if self._has_duplicate_names(predicate_list):
            raise ValueError("Duplicate predicate names")

        examples = self._create_examples(possible_colours, variables)

        # Background facts for predicates
        background_knowledge = self._create_background_knowledge(
            predicates, possible_colours
        )

        # Filter out predicates that are not used
        predicate_list = self._filter_predicates(predicate_list, background_knowledge)

        # Clear caches to free up memory
        self._distance_until_overlap_horizontally.cache_clear()
        self._distance_until_overlap_vertically.cache_clear()

        # Run FOIL
        foil = FOIL(
            target_literal,
            predicate_list,
            background_knowledge,
            beam_width=beam_width,
            max_clause_length=max_clause_length,
            timeout_seconds=timeout_seconds,
            type_extension_limit={
                arg_types.example_number_arg: 1,
                arg_types.i_arg: 1,
                arg_types.j_arg: 1,
                arg_types.number_value_arg: 2,
                arg_types.shape_arg: 3,
                arg_types.colour_type_arg: 2,
            },
        )
        foil.fit(examples)

        # Calculate results
        return self._calculate_results(foil, possible_colours, variables)

    def _extract_all_possible_colour_counts_for_grids(self) -> List[int]:
        possible_counts: set[int] = set()
        for input in self.inputs:
            count = self._get_grid_colour_count(input)
            possible_counts.add(count)

        return list(possible_counts)

    def _get_possible_i_values_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, output in enumerate(self.outputs):
            possible_values[example_number] = list(range(output.shape[0]))

        for example_number, output in enumerate(self.empty_test_outputs):
            possible_values[example_number + self._TEST_EX_NUMBER_OFFSET] = list(
                range(output.shape[0])
            )

        def get_possible_i_values(example: dict[str, Any]) -> list:
            # V1 is set as example number
            example_number = example["V1"]
            return possible_values[example_number]

        return get_possible_i_values

    def _get_possible_j_values_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, output in enumerate(self.outputs):
            possible_values[example_number] = list(range(output.shape[1]))

        for example_number, output in enumerate(self.empty_test_outputs):
            possible_values[example_number + self._TEST_EX_NUMBER_OFFSET] = list(
                range(output.shape[1])
            )

        def get_possible_j_values(example: dict[str, Any]) -> list:
            # V1 is set as example number
            example_number = example["V1"]
            return possible_values[example_number]

        return get_possible_j_values

    def _get_possible_number_values_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, output in enumerate(self.outputs):
            # we just take the max of the x and y cords
            # this will create unnecessary values for non square grids
            max_value = max(output.shape[0], output.shape[1])
            possible_values[example_number] = list(range(max_value))

        for example_number, output in enumerate(self.empty_test_outputs):
            # we just take the max of the x and y cords
            # this will create unnecessary values for non square grids
            max_value = max(output.shape[0], output.shape[1])
            possible_values[example_number + self._TEST_EX_NUMBER_OFFSET] = list(
                range(max_value)
            )

        def get_possible_mumber_values(example: dict[str, Any]) -> list:
            # V1 is set as example number
            example_number = example["V1"]
            return possible_values[example_number]

        return get_possible_mumber_values

    def _get_possible_shapes_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, _ in enumerate(self.inputs):
            input_shapes: list[str] = []
            for i in range(len(self.inputs_shapes[example_number])):
                if self.inputs_shapes[example_number][i].shape_type == ShapeType.PIXEL:
                    continue
                input_shapes.append(self._generate_shape_name(example_number, True, i))

            output_shapes: list[str] = []
            # output_shapes = [
            #     self._generate_shape_name(example_number, False, i)
            #     for i in range(len(outputs_shapes[example_number]))
            # ]
            possible_values[example_number] = input_shapes + output_shapes

        for example_number, _ in enumerate(self.test_inputs):
            input_shapes = []
            for i in range(len(self.test_inputs_shapes[example_number])):
                if (
                    self.test_inputs_shapes[example_number][i].shape_type
                    == ShapeType.PIXEL
                ):
                    continue
                input_shapes.append(
                    self._generate_shape_name(
                        example_number + self._TEST_EX_NUMBER_OFFSET, True, i
                    )
                )

            output_shapes = []
            # output_shapes = [
            #     self._generate_shape_name(example_number + self._TEST_EX_NUMBER_OFFSET, False, i) # noqa: E501
            #     for i in range(len(outputs_shapes[example_number]))
            # ]
            possible_values[example_number + self._TEST_EX_NUMBER_OFFSET] = (
                input_shapes + output_shapes
            )

        def get_possible_shapes(example: dict[str, Any]) -> list:
            # V1 is set as example number
            example_number = example["V1"]
            return possible_values[example_number]

        return get_possible_shapes

    def _get_top_left_bottom_right_diag_eval_func(self) -> Callable[..., bool]:
        is_square: dict[int, bool] = {}
        for ex, output_grid in enumerate(self.outputs):
            if output_grid.shape[0] == output_grid.shape[1]:
                is_square[ex] = True
            else:
                is_square[ex] = False

        for ex, output_grid in enumerate(self.empty_test_outputs):
            if output_grid.shape[0] == output_grid.shape[1]:
                is_square[ex + self._TEST_EX_NUMBER_OFFSET] = True
            else:
                is_square[ex + self._TEST_EX_NUMBER_OFFSET] = False

        return lambda ex_number, i, j: is_square[ex_number] and i == j

    def _get_bottom_left_top_right_diag_eval_func(self) -> Callable[..., bool]:
        is_square: dict[int, bool] = {}
        heights: dict[int, int] = {}
        for ex, output_grid in enumerate(self.outputs):
            heights[ex] = output_grid.shape[0]
            if output_grid.shape[0] == output_grid.shape[1]:
                is_square[ex] = True
            else:
                is_square[ex] = False

        for ex, output_grid in enumerate(self.empty_test_outputs):
            heights[ex + self._TEST_EX_NUMBER_OFFSET] = output_grid.shape[0]
            if output_grid.shape[0] == output_grid.shape[1]:
                is_square[ex + self._TEST_EX_NUMBER_OFFSET] = True
            else:
                is_square[ex + self._TEST_EX_NUMBER_OFFSET] = False

        return (
            lambda ex_number, i, j: is_square[ex_number]
            and i + j == heights[ex_number] - 1
        )

    def _calculate_results(
        self,
        foil: FOIL,
        possible_colours: list[Colour],
        variables: Variables,
    ) -> List[NDArray[np.int16]]:
        # we iteratively populate the test outputs
        test_outputs: List[NDArray[np.int16]] = []
        for test_number, output_grid in enumerate(self.empty_test_outputs):
            offset_test_number = test_number + self._TEST_EX_NUMBER_OFFSET
            test_output = np.zeros_like(output_grid)
            for i in range(output_grid.shape[0]):
                for j in range(output_grid.shape[1]):
                    for possible_colour in possible_colours:
                        if foil.predict(
                            {
                                variables.v1.name: offset_test_number,
                                variables.v2.name: possible_colour,
                                variables.v3.name: i,
                                variables.v4.name: j,
                            }
                        ):
                            if test_output[i, j] != 0:
                                raise Exception("Multiple predictions for same cell")
                            test_output[i, j] = possible_colour.value
            test_outputs.append(test_output)

        return test_outputs

    def _append_background_knowledge_for_grids(
        self,
        inputs,
        grid_colour_count_predicates: list[Predicate],
        background_knowledge: dict[str, set[tuple]],
        ex_number_offset=0,
    ):
        for ex, input_grid in enumerate(inputs):
            unique_colour_count = self._get_grid_colour_count(input_grid)
            for grid_colour_count_pred in grid_colour_count_predicates:
                if (
                    self._generate_grid_colour_count_pred_name(unique_colour_count)
                    == grid_colour_count_pred.name
                ):
                    background_knowledge[grid_colour_count_pred.name].add(
                        (ex + ex_number_offset,)
                    )

    def _append_background_knowledge_with_raw_input(
        self,
        background_knowledge: BackgroundKnowledgeType,
        inputs: list[NDArray[np.int16]],
        predicates: Predicates,
        ex_number_offset=0,
    ) -> None:
        for ex_number, example_grid in enumerate(inputs):
            offset_ex_number = ex_number + ex_number_offset
            for i in range(example_grid.shape[0]):
                for j in range(example_grid.shape[1]):
                    value = example_grid[i, j]
                    if value == 0:
                        background_knowledge[predicates.empty_pred.name].add(
                            (offset_ex_number, i, j)
                        )
                    # else:
                    #     background_knowledge[predicates.input_pred.name].add(
                    #         (offset_ex_number, Colour(value), i, j)
                    #     )

    def _append_background_knowledge_for_shapes(
        self,
        background_knowledge: BackgroundKnowledgeType,
        outputs: list[NDArray[np.int16]],
        inputs: list[NDArray[np.int16]],
        inputs_shapes: list[list[Shape]],
        predicates: Predicates,
        possible_colours: list[Colour],
        ex_number_offset: int = 0,
    ) -> None:
        for ex_number, (input_shapes, output_grid, input_grid) in enumerate(
            zip(inputs_shapes, outputs, inputs)
        ):
            ex_test_number = ex_number + ex_number_offset

            for input_shape_index, input_shape in enumerate(input_shapes):
                # for now lets ignore pixels
                if input_shape.shape_type == ShapeType.PIXEL:
                    continue

                input_shape_name = self._generate_shape_name(
                    ex_test_number, True, input_shape_index
                )

                if input_shape.colour is not None:
                    background_knowledge[predicates.shape_colour_pred.name].add(
                        (ex_test_number, input_shape_name, input_shape.colour)
                    )

                is_vertical_line: bool = input_shape.is_vertical_line(input_grid)
                is_horizontal_line: bool = input_shape.is_horizontal_line(input_grid)

                if is_vertical_line or is_horizontal_line:
                    background_knowledge[predicates.shape_spanning_line_pred.name].add(
                        (ex_test_number, input_shape_name)
                    )

                if is_vertical_line:
                    background_knowledge[predicates.shape_vertical_line_pred.name].add(
                        (ex_test_number, input_shape_name)
                    )

                if is_horizontal_line:
                    background_knowledge[
                        predicates.shape_horizontal_line_pred.name
                    ].add((ex_test_number, input_shape_name))

                if input_shape.colour is not None:
                    for colour_pred in predicates.shape_colour_predicates:
                        if (
                            self._generate_shape_colour_pred_name(input_shape.colour)
                            == colour_pred.name
                        ):
                            background_knowledge[colour_pred.name].add(
                                (ex_test_number, input_shape_name)
                            )

                for shape_colour_count_pred in predicates.shape_colour_count_predicates:
                    if (
                        self._generate_shape_colour_count_pred_name(
                            input_shape.colour_count
                        )
                        == shape_colour_count_pred.name
                    ):
                        background_knowledge[shape_colour_count_pred.name].add(
                            (ex_test_number, input_shape_name)
                        )

                for shape_size_pred in predicates.shape_size_predicates:
                    if (
                        self._generate_shape_size_pred_name(
                            input_shape.num_of_coloured_pixels
                        )
                        == shape_size_pred.name
                    ):
                        background_knowledge[shape_size_pred.name].add(
                            (ex_test_number, input_shape_name)
                        )

                for shape_group_pred in predicates.shape_group_predicates:
                    for shape_group in input_shape.shape_groups:
                        if (
                            self._generate_shape_group_pred_name(shape_group)
                            == shape_group_pred.name
                        ):
                            background_knowledge[shape_group_pred.name].add(
                                (ex_test_number, input_shape_name)
                            )

                for i in range(output_grid.shape[0]):
                    for j in range(output_grid.shape[1]):
                        self._append_background_knowledge_for_shape(
                            background_knowledge,
                            i,
                            j,
                            input_shape,
                            output_grid,
                            input_grid,
                            ex_test_number,
                            input_shape_name,
                            predicates,
                            possible_colours,
                        )

            for input_shape_index_1, input_shape_1 in enumerate(input_shapes):
                # for now lets ignore pixels
                if input_shape_1.shape_type == ShapeType.PIXEL:
                    continue
                input_shape_name_1 = self._generate_shape_name(
                    ex_test_number, True, input_shape_index_1
                )
                for input_shape_index_2, input_shape_2 in enumerate(input_shapes):
                    # for now lets ignore pixels
                    if input_shape_2.shape_type == ShapeType.PIXEL:
                        continue
                    if input_shape_index_1 == input_shape_index_2:
                        continue

                    input_shape_name_2 = self._generate_shape_name(
                        ex_test_number, True, input_shape_index_2
                    )

                    for i in range(output_grid.shape[0]):
                        for j in range(output_grid.shape[1]):
                            self._append_background_knowledge_between_shapes(
                                background_knowledge,
                                i,
                                j,
                                input_shape_1,
                                input_shape_name_1,
                                input_shape_2,
                                input_shape_name_2,
                                ex_test_number,
                                predicates,
                            )

    def _append_background_knowledge_for_shape(
        self,
        background_knowledge: BackgroundKnowledgeType,
        output_i: int,
        output_j: int,
        input_shape: Shape,
        output_grid: NDArray[np.int16],
        input_grid: NDArray[np.int16],
        ex_number: int,
        input_shape_name: str,
        predicates: Predicates,
        possible_colours: list[Colour],
    ) -> None:
        if input_shape.is_above_ij(output_i, output_j):
            background_knowledge[predicates.above_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_below_ij(output_i, output_j):
            background_knowledge[predicates.below_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_left_of_ij(output_i, output_j):
            background_knowledge[predicates.left_of_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_right_of_ij(output_i, output_j):
            background_knowledge[predicates.right_of_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        is_inline_diagonally_above_right = (
            input_shape.is_inline_diagonally_above_right_ij(output_i, output_j)
        )
        is_inline_diagonally_above_left = (
            input_shape.is_inline_diagonally_above_left_ij(output_i, output_j)
        )
        is_inline_diagonally_below_right = (
            input_shape.is_inline_diagonally_below_right_ij(output_i, output_j)
        )
        is_inline_diagonally_below_left = (
            input_shape.is_inline_diagonally_below_left_ij(output_i, output_j)
        )

        if is_inline_diagonally_above_right:
            background_knowledge[
                predicates.inline_diagonally_above_right_pred.name
            ].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_diagonally_above_left:
            background_knowledge[predicates.inline_diagonally_above_left_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_diagonally_below_right:
            background_knowledge[
                predicates.inline_diagonally_below_right_pred.name
            ].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_diagonally_below_left:
            background_knowledge[predicates.inline_diagonally_below_left_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_diagonally_above_right or is_inline_diagonally_below_left:
            background_knowledge[predicates.inline_diagonally_right_left_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_diagonally_above_left or is_inline_diagonally_below_right:
            background_knowledge[predicates.inline_diagonally_left_right_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        is_inline_above_vertically = input_shape.is_inline_above_vertically_ij(
            output_i, output_j
        )

        is_inline_below_vertically = input_shape.is_inline_below_vertically_ij(
            output_i, output_j
        )

        if is_inline_above_vertically:
            background_knowledge[predicates.inline_above_vertically_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_below_vertically:
            background_knowledge[predicates.inline_below_vertically_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_above_vertically or is_inline_below_vertically:
            background_knowledge[predicates.inline_vertically_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        is_inline_left_horizontally = input_shape.is_inline_left_horizontally_ij(
            output_i, output_j
        )
        is_inline_right_horizontally = input_shape.is_inline_right_horizontally_ij(
            output_i, output_j
        )

        if is_inline_left_horizontally:
            background_knowledge[predicates.inline_left_horizontally_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_right_horizontally:
            background_knowledge[predicates.inline_right_horizontally_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if is_inline_left_horizontally or is_inline_right_horizontally:
            background_knowledge[predicates.inline_horizontally_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        # if input_shape.is_ij_inside_mask(output_i, output_j):
        #     background_knowledge[predicates.inside_mask_pred.name].add(
        #         (
        #             ex_number,
        #             output_i,
        #             output_j,
        #             input_shape_name,
        #         )
        #     )

        if input_shape.is_ij_inside_blank_space(output_i, output_j):
            background_knowledge[predicates.inside_blank_space_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_mask_overlapping_ij(output_i, output_j):
            background_knowledge[predicates.mask_overlapping_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        self._append_background_knowledge_for_expandable_shapes(
            background_knowledge,
            output_i,
            output_j,
            input_shape,
            output_grid,
            ex_number,
            input_shape_name,
            predicates.mask_overlapping_expanded_to_grid_pred,
            predicates.mask_overlapping_and_colour_expanded_to_grid_pred,
            possible_colours,
        )

        self._append_background_knowledge_for_repeatable_shapes(
            background_knowledge,
            output_i,
            output_j,
            input_shape,
            output_grid,
            ex_number,
            input_shape_name,
            predicates.mask_overlapping_repeated_grid_pred,
            predicates.mask_overlapping_and_colour_repeated_grid_pred,
            possible_colours,
        )

        for colour in possible_colours:
            if input_shape.is_mask_overlapping_and_colour_ij(
                output_i, output_j, colour
            ):
                background_knowledge[
                    predicates.mask_overlapping_and_colour_pred.name
                ].add(
                    (
                        ex_number,
                        colour,
                        output_i,
                        output_j,
                        input_shape_name,
                    )
                )

        # for now we just look at cases where the whole shape
        # mask is the size of the output grid
        # we ignore when input grid and output grid the same size
        if (
            output_grid.shape == input_shape.mask.shape
            and input_grid.shape != output_grid.shape
        ):
            if input_shape.is_mask_overlapping_ij(
                output_i + input_shape.position[0], output_j + input_shape.position[1]
            ):
                background_knowledge[
                    predicates.mask_overlapping_moved_to_grid_pred.name
                ].add(
                    (
                        ex_number,
                        output_i,
                        output_j,
                        input_shape_name,
                    )
                )

            if input_shape.is_ij_inside_mask_not_overlapping(
                output_i + input_shape.position[0], output_j + input_shape.position[1]
            ):
                background_knowledge[
                    predicates.inside_mask_not_overlapping_moved_to_grid_pred.name
                ].add(
                    (
                        ex_number,
                        output_i,
                        output_j,
                        input_shape_name,
                    )
                )

        if input_shape.is_ij_inside_mask_not_overlapping(output_i, output_j):
            background_knowledge[predicates.inside_mask_not_overlapping_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        vertical_center_distance = input_shape.vertical_distance_from_center_ij(
            output_i,
            output_j,
        )
        horizontal_center_distance = input_shape.horizontal_distance_from_center_ij(
            output_i,
            output_j,
        )
        vertical_edge_distance = input_shape.vertical_distance_from_edge_ij(
            output_i,
            output_j,
        )
        horizontal_edge_distance = input_shape.horizontal_distance_from_edge_ij(
            output_i,
            output_j,
        )

        for number in range(self._get_max_num_arg_value()):
            if vertical_center_distance > number:
                background_knowledge[
                    predicates.vertical_center_distance_more_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

            if vertical_center_distance < number:
                background_knowledge[
                    predicates.vertical_center_distance_less_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

            if horizontal_center_distance > number:
                background_knowledge[
                    predicates.horizontal_center_distance_more_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

            if horizontal_center_distance < number:
                background_knowledge[
                    predicates.horizontal_center_distance_less_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

            if vertical_edge_distance > number:
                background_knowledge[
                    predicates.vertical_edge_distance_more_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

            if vertical_edge_distance < number:
                background_knowledge[
                    predicates.vertical_edge_distance_less_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

            if horizontal_edge_distance > number:
                background_knowledge[
                    predicates.horizontal_edge_distance_more_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

            if horizontal_edge_distance < number:
                background_knowledge[
                    predicates.horizontal_edge_distance_less_than_pred.name
                ].add((ex_number, output_i, output_j, input_shape_name, number))

        # if input_shape.shape_type == ShapeType.MIXED_COLOUR:
        #     background_knowledge[self._MIXED_COLOUR_SHAPE_PRED_NAME].add(
        #         (input_shape_name,)
        #     )

        # if input_shape.shape_type == ShapeType.SINGLE_COLOUR:
        #     background_knowledge[self._SINGLE_COLOUR_SHAPE_PRED_NAME].add(
        #         (input_shape_name,)
        #     )

    def _append_background_knowledge_between_shapes(
        self,
        background_knowledge: BackgroundKnowledgeType,
        output_i: int,
        output_j: int,
        input_shape_1: Shape,
        input_shape_name_1: str,
        input_shape_2: Shape,
        input_shape_name_2: str,
        ex_number: int,
        predicates: Predicates,
    ) -> None:
        if isinstance(input_shape_1, RotatableMaskShape) or isinstance(
            input_shape_2, RotatableMaskShape
        ):
            return  # for now we don't have time to consider these complications

        self._append_background_knowledge_for_mask_inline_with_shape(
            background_knowledge,
            output_i,
            output_j,
            input_shape_1,
            input_shape_name_1,
            input_shape_2,
            input_shape_name_2,
            ex_number,
            predicates.mask_overlapping_top_inline_top_pred,
            predicates.mask_overlapping_bot_inline_bot_pred,
            predicates.mask_overlapping_left_inline_left_pred,
            predicates.mask_overlapping_right_inline_right_pred,
            predicates.mask_overlapping_bot_top_touching_pred,
            predicates.mask_overlapping_top_bot_touching_pred,
            predicates.mask_overlapping_right_left_touching_pred,
            predicates.mask_overlapping_left_right_touching_pred,
        )

        self._append_background_knowledge_for_mask_gravity_to_shape(
            background_knowledge,
            output_i,
            output_j,
            input_shape_1,
            input_shape_name_1,
            input_shape_2,
            input_shape_name_2,
            ex_number,
            predicates.mask_overlapping_gravity_to_shape_pred,
        )

    @staticmethod
    def _append_background_knowledge_for_mask_inline_with_shape(
        background_knowledge: BackgroundKnowledgeType,
        output_i: int,
        output_j: int,
        input_shape_1: Shape,
        input_shape_name_1: str,
        input_shape_2: Shape,
        input_shape_name_2: str,
        ex_number: int,
        mask_overlapping_top_inline_top_pred: Predicate,
        mask_overlapping_bot_inline_bot_pred: Predicate,
        mask_overlapping_left_inline_left_pred: Predicate,
        mask_overlapping_right_inline_right_pred: Predicate,
        mask_overlapping_bot_top_touching_pred: Predicate,
        mask_overlapping_top_bot_touching_pred: Predicate,
        mask_overlapping_right_left_touching_pred: Predicate,
        mask_overlapping_left_right_touching_pred: Predicate,
    ):
        distance_between_tops = input_shape_2.top_most - input_shape_1.top_most
        distance_between_bottoms = input_shape_2.bottom_most - input_shape_1.bottom_most
        distance_between_lefts = input_shape_2.left_most - input_shape_1.left_most
        distance_between_rights = input_shape_2.right_most - input_shape_1.right_most

        distance_between_bot_top_touching = (
            input_shape_2.top_most - input_shape_1.bottom_most - 1
        )
        distance_between_top_bot_touching = (
            input_shape_2.bottom_most - input_shape_1.top_most + 1
        )
        distance_between_right_left_touching = (
            input_shape_2.left_most - input_shape_1.right_most - 1
        )
        distance_between_left_right_touching = (
            input_shape_2.right_most - input_shape_1.left_most + 1
        )

        if input_shape_1.is_mask_overlapping_ij(
            output_i - distance_between_tops, output_j
        ):
            # in shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_top_inline_top_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

        if input_shape_1.is_mask_overlapping_ij(
            output_i - distance_between_bottoms, output_j
        ):
            # is shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_bot_inline_bot_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

        if input_shape_1.is_mask_overlapping_ij(
            output_i, output_j - distance_between_lefts
        ):
            # in shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_left_inline_left_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

        if input_shape_1.is_mask_overlapping_ij(
            output_i, output_j - distance_between_rights
        ):
            # in shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_right_inline_right_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

        if input_shape_1.is_mask_overlapping_ij(
            output_i - distance_between_bot_top_touching, output_j
        ):
            # in shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_bot_top_touching_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

        if input_shape_1.is_mask_overlapping_ij(
            output_i - distance_between_top_bot_touching, output_j
        ):
            # in shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_top_bot_touching_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

        if input_shape_1.is_mask_overlapping_ij(
            output_i, output_j - distance_between_right_left_touching
        ):
            # in shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_right_left_touching_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

        if input_shape_1.is_mask_overlapping_ij(
            output_i, output_j - distance_between_left_right_touching
        ):
            # in shape 1 and inline with shape 2
            background_knowledge[mask_overlapping_left_right_touching_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name_1,
                    input_shape_name_2,
                )
            )

    def _append_background_knowledge_for_mask_gravity_to_shape(
        self,
        background_knowledge: BackgroundKnowledgeType,
        output_i: int,
        output_j: int,
        input_shape_1: Shape,
        input_shape_name_1: str,
        input_shape_2: Shape,
        input_shape_name_2: str,
        ex_number: int,
        mask_overlapping_gravity_to_shape_pred: Predicate,
    ):
        vertical_distance_to_touch = self._distance_until_overlap_vertically(
            input_shape_1, input_shape_2
        )
        horizontal_distance_to_touch = self._distance_until_overlap_horizontally(
            input_shape_1, input_shape_2
        )

        if vertical_distance_to_touch is None and horizontal_distance_to_touch is None:
            return

        vertical_distance_closest: bool = False

        if (
            vertical_distance_to_touch is not None
            and horizontal_distance_to_touch is not None
        ):
            vertical_distance_closest = abs(vertical_distance_to_touch) < abs(
                horizontal_distance_to_touch
            )
        else:
            vertical_distance_closest = vertical_distance_to_touch is not None

        if vertical_distance_closest:
            if input_shape_1.is_mask_overlapping_ij(
                output_i - vertical_distance_to_touch, output_j  # type: ignore
            ):
                # overlapping shape 1 and touching closest on shape 2
                background_knowledge[mask_overlapping_gravity_to_shape_pred.name].add(
                    (
                        ex_number,
                        output_i,
                        output_j,
                        input_shape_name_1,
                        input_shape_name_2,
                    )
                )

        else:
            if input_shape_1.is_mask_overlapping_ij(
                output_i, output_j - horizontal_distance_to_touch  # type: ignore
            ):
                # overlapping shape 1 and touching closest on shape 2
                background_knowledge[mask_overlapping_gravity_to_shape_pred.name].add(
                    (
                        ex_number,
                        output_i,
                        output_j,
                        input_shape_name_1,
                        input_shape_name_2,
                    )
                )

    def _create_args_types(
        self,
        possible_colours: list[Colour],
    ) -> ArgTypes:
        # args in target predicate & body predicates
        colour_type_arg = ArgType("colour", possible_colours)
        example_number_arg = ArgType(
            "example_number",
            list(range(len(self.inputs_shapes))),  # never actually gets extended
        )
        i_arg = ArgType("i", possible_values_fn=self._get_possible_i_values_func())
        j_arg = ArgType("j", possible_values_fn=self._get_possible_j_values_func())

        # args not in target but in possible body predicates
        shape_arg = ArgType(
            "shape",
            possible_values_fn=self._get_possible_shapes_func(),
        )

        number_value_arg = ArgType(
            "number_value", possible_values_fn=self._get_possible_number_values_func()
        )

        return self.ArgTypes(
            colour_type_arg=colour_type_arg,
            example_number_arg=example_number_arg,
            i_arg=i_arg,
            j_arg=j_arg,
            shape_arg=shape_arg,
            number_value_arg=number_value_arg,
        )

    def _create_variables(self, arg_types: ArgTypes) -> Variables:
        return self.Variables(
            v1=Variable("V1", arg_types.example_number_arg),
            v2=Variable("V2", arg_types.colour_type_arg),
            v3=Variable("V3", arg_types.i_arg),
            v4=Variable("V4", arg_types.j_arg),
        )

    def _create_target_literal(
        self, arg_types: ArgTypes, variables: "Solver.Variables"
    ) -> Literal:
        target_predicate = Predicate(
            "output",
            4,
            [
                arg_types.example_number_arg,
                arg_types.colour_type_arg,
                arg_types.i_arg,
                arg_types.j_arg,
            ],
        )  # noqa: E501
        return Literal(
            predicate=target_predicate,
            args=[variables.v1, variables.v2, variables.v3, variables.v4],
        )

    def _create_predicates(
        self,
        arg_types: ArgTypes,
        possible_colours: list[Colour],
    ) -> Predicates:
        ex_num_arg = arg_types.example_number_arg
        colour_type_arg = arg_types.colour_type_arg
        i_arg = arg_types.i_arg
        j_arg = arg_types.j_arg
        shape_arg = arg_types.shape_arg
        number_value_arg = arg_types.number_value_arg

        # input_pred = Predicate(
        #     "input", 4, [ex_num_arg, colour_type_arg, i_arg, j_arg]
        # )  # noqa: E501
        empty_pred = Predicate("empty", 3, [ex_num_arg, i_arg, j_arg])
        above_pred = Predicate("above", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        below_pred = Predicate("below", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        left_of_pred = Predicate("left-of", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        right_of_pred = Predicate("right-of", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        inline_diagonally_above_right_pred = Predicate(
            "inline-diagonally-above-right",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_diagonally_above_left_pred = Predicate(
            "inline-diagonally-above-left",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_diagonally_below_right_pred = Predicate(
            "inline-diagonally-below-right",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_diagonally_below_left_pred = Predicate(
            "inline-diagonally-below-left",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_diagonally_right_left_pred = Predicate(
            "inline-diagonally-right-left",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_diagonally_left_right_pred = Predicate(
            "inline-diagonally-left-right",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_vertically_pred = Predicate(
            "inline-vertically",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )
        inline_above_vertically_pred = Predicate(
            "inline-above-vertically",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_below_vertically_pred = Predicate(
            "inline-below-vertically",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_horizontally_pred = Predicate(
            "inline-horizontally",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
            allow_negation=False,
        )
        inline_left_horizontally_pred = Predicate(
            "inline-left-horizontally",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_right_horizontally_pred = Predicate(
            "inline-right-horizontally",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        mask_overlapping_pred = Predicate(
            "mask-overlapping",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )
        mask_overlapping_and_colour_pred = Predicate(
            "mask-overlapping-and-colour",
            5,
            [ex_num_arg, colour_type_arg, i_arg, j_arg, shape_arg],
        )
        mask_overlapping_expanded_to_grid_pred = Predicate(
            "mask-overlapping-expanded-to-grid",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )
        mask_overlapping_and_colour_expanded_to_grid_pred = Predicate(
            "mask-overlapping-and-colour-expanded-to-grid",
            5,
            [ex_num_arg, colour_type_arg, i_arg, j_arg, shape_arg],
        )
        mask_overlapping_repeated_grid_pred = Predicate(
            "mask-overlapping-repeated-grid",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )
        mask_overlapping_and_colour_repeated_grid_pred = Predicate(
            "mask-overlapping-and-colour-repeated-grid",
            5,
            [ex_num_arg, colour_type_arg, i_arg, j_arg, shape_arg],
        )
        mask_overlapping_moved_to_grid_pred = Predicate(
            "mask-overlapping-moved-to-grid",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )
        inside_mask_not_overlapping_moved_to_grid_pred = Predicate(
            "inside-mask-not-overlapping-moved-to-grid",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )
        mask_overlapping_top_inline_top_pred = Predicate(
            "mask-overlapping-top-inline-top-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_bot_inline_bot_pred = Predicate(
            "mask-overlapping-bot-inline-bot-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_left_inline_left_pred = Predicate(
            "mask-overlapping-left-inline-left-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_right_inline_right_pred = Predicate(
            "mask-overlapping-right-inline-right-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_bot_top_touching_pred = Predicate(
            "mask-overlapping-bot-top-touching-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_top_bot_touching_pred = Predicate(
            "mask-overlapping-top-bot-touching-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_right_left_touching_pred = Predicate(
            "mask-overlapping-right-left-touching-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_left_right_touching_pred = Predicate(
            "mask-overlapping-left-right-touching-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        mask_overlapping_gravity_to_shape_pred = Predicate(
            "mask-overlapping-gravity-to-shape",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, shape_arg],
        )
        # inside_mask_pred = Predicate(
        #     "inside-mask", 4, [ex_num_arg, i_arg, j_arg, shape_arg]
        # )
        inside_blank_space_pred = Predicate(
            "inside-blank-space", 4, [ex_num_arg, i_arg, j_arg, shape_arg]
        )
        inside_mask_not_overlapping_pred = Predicate(
            "inside-mask-not-overlapping",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )
        top_left_bottom_right_diag_pred = RuleBasedPredicate(
            "top-left-bottom-right-diag",
            3,
            [ex_num_arg, i_arg, j_arg],
            self._get_top_left_bottom_right_diag_eval_func(),
        )
        bottom_left_top_right_diag_pred = RuleBasedPredicate(
            "bottom-left-top-right-diag",
            3,
            [ex_num_arg, i_arg, j_arg],
            self._get_bottom_left_top_right_diag_eval_func(),
        )
        shape_colour_pred = Predicate(
            "shape-colour",
            3,
            [ex_num_arg, shape_arg, colour_type_arg],
        )
        shape_spanning_line_pred = Predicate(
            "shape-spanning-line",
            2,
            [ex_num_arg, shape_arg],
        )
        shape_vertical_line_pred = Predicate(
            "shape-vertical-line",
            2,
            [ex_num_arg, shape_arg],
        )
        shape_horizontal_line_pred = Predicate(
            "shape-horizontal-line",
            2,
            [ex_num_arg, shape_arg],
        )
        # mixed_colour_shape_pred = Predicate(
        #     "mixed-colour-shape, 1, [shape_arg]
        # )
        # single_colour_shape_pred = Predicate(
        #     "single-colour-shape", 1, [shape_arg]
        # )

        shape_colour_predicates: list[Predicate] = []
        colour_predicates: list[RuleBasedPredicate] = []

        for colour in possible_colours:
            shape_colour_predicates.append(
                Predicate(
                    self._generate_shape_colour_pred_name(colour),
                    2,
                    [ex_num_arg, shape_arg],
                    allow_negation=False,
                )
            )

            colour_predicates.append(
                RuleBasedPredicate(
                    f"colour-{colour.name}",
                    1,
                    [colour_type_arg],
                    self._get_colour_eval_func(colour),
                )
            )

        shape_size_predicates: list[Predicate] = []
        for size in self._extract_all_possible_sizes_for_shapes(self.inputs_shapes):
            shape_size_predicates.append(
                Predicate(
                    self._generate_shape_size_pred_name(size),
                    2,
                    [ex_num_arg, shape_arg],
                )
            )

        # max_is: set[int] = set()
        # max_js: set[int] = set()
        # for output in self.outputs:
        #     max_is.add(output.shape[0])
        #     max_js.add(output.shape[1])

        # max_i = max(max_is)
        # max_j = max(max_js)

        # inequality_predicates: list[RuleBasedPredicate] = []
        # for i in range(max_i):
        #     if i != max_i:
        #         inequality_predicates.append(
        #             RuleBasedPredicate(
        #                 f"i-greater-than-{i}",
        #                 1,
        #                 [i_arg],
        #                 self._get_more_than_eval_func(i),
        #             )
        #         )

        #     if i != 0:
        #         inequality_predicates.append(
        #             RuleBasedPredicate(
        #                 f"i-less-than-{i}",
        #                 1,
        #                 [i_arg],
        #                 self._get_less_than_eval_func(i),
        #             )
        #         )

        # for j in range(max_j):
        #     if i != max_j:
        #         inequality_predicates.append(
        #             RuleBasedPredicate(
        #                 f"j-greater-than-{j}",
        #                 1,
        #                 [j_arg],
        #                 self._get_more_than_eval_func(j),
        #             )
        #         )

        #     if j != 0:
        #         inequality_predicates.append(
        #             RuleBasedPredicate(
        #                 f"j-less-than-{j}",
        #                 1,
        #                 [j_arg],
        #                 self._get_less_than_eval_func(j),
        #             )
        #         )

        shape_colour_count_predicates: list[Predicate] = []
        for possible_count in self._extract_all_possible_colour_counts_for_shapes(
            self.inputs_shapes
        ):
            shape_colour_count_predicates.append(
                Predicate(
                    self._generate_shape_colour_count_pred_name(possible_count),
                    2,
                    [ex_num_arg, shape_arg],
                )
            )

        grid_colour_count_predicates: list[Predicate] = []
        for possible_count in self._extract_all_possible_colour_counts_for_grids():
            grid_colour_count_predicates.append(
                Predicate(
                    self._generate_grid_colour_count_pred_name(possible_count),
                    1,
                    [ex_num_arg],
                )
            )

        shape_group_predicates: list[Predicate] = []
        for shape_group in self._extract_all_possible_shape_groups(self.inputs_shapes):
            shape_group_pred = Predicate(
                self._generate_shape_group_pred_name(shape_group),
                2,
                [ex_num_arg, shape_arg],
            )
            shape_group_predicates.append(shape_group_pred)

        vertical_center_distance_more_than_pred = Predicate(
            "vertical-center-distance-more-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        vertical_center_distance_less_than_pred = Predicate(
            "vertical-center-distance-less-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        horizontal_center_distance_more_than_pred = Predicate(
            "horizontal-center-distance-more-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        horizontal_center_distance_less_than_pred = Predicate(
            "horizontal-center-distance-less-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        vertical_edge_distance_more_than_pred = Predicate(
            "vertical-edge-distance-more-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        vertical_edge_distance_less_than_pred = Predicate(
            "vertical-edge-distance-less-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        horizontal_edge_distance_more_than_pred = Predicate(
            "horizontal-edge-distance-more-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        horizontal_edge_distance_less_than_pred = Predicate(
            "horizontal-edge-distance-less-than",
            5,
            [ex_num_arg, i_arg, j_arg, shape_arg, number_value_arg],
        )

        vertical_grid_top_distance_more_than_pred = RuleBasedPredicate(
            "vertical-grid-top-distance-more-than",
            3,
            [ex_num_arg, i_arg, number_value_arg],
            self._get_grid_top_more_than_eval_func(),
        )

        vertical_grid_bottom_distance_more_than_pred = RuleBasedPredicate(
            "vertical-grid-bottom-distance-more-than",
            3,
            [ex_num_arg, i_arg, number_value_arg],
            self._get_grid_bottom_more_than_eval_func(),
        )

        vertical_grid_top_distance_less_than_pred = RuleBasedPredicate(
            "vertical-grid-top-distance-less-than",
            3,
            [ex_num_arg, i_arg, number_value_arg],
            self._get_grid_top_less_than_eval_func(),
        )

        vertical_grid_bottom_distance_less_than_pred = RuleBasedPredicate(
            "vertical-grid-bottom-distance-less-than",
            3,
            [ex_num_arg, i_arg, number_value_arg],
            self._get_grid_bottom_less_than_eval_func(),
        )

        horizontal_grid_left_distance_more_than_pred = RuleBasedPredicate(
            "horizontal-grid-left-distance-more-than",
            3,
            [ex_num_arg, j_arg, number_value_arg],
            self._get_grid_left_more_than_eval_func(),
        )

        horizontal_grid_right_distance_more_than_pred = RuleBasedPredicate(
            "horizontal-grid-right-distance-more-than",
            3,
            [ex_num_arg, j_arg, number_value_arg],
            self._get_grid_right_more_than_eval_func(),
        )

        horizontal_grid_left_distance_less_than_pred = RuleBasedPredicate(
            "horizontal-grid-left-distance-less-than",
            3,
            [ex_num_arg, j_arg, number_value_arg],
            self._get_grid_left_less_than_eval_func(),
        )

        horizontal_grid_right_distance_less_than_pred = RuleBasedPredicate(
            "horizontal-grid-right-distance-less-than",
            3,
            [ex_num_arg, j_arg, number_value_arg],
            self._get_grid_right_less_than_eval_func(),
        )

        number_value_predicates: list[RuleBasedPredicate] = []
        for number in range(self._get_max_num_arg_value()):
            number_value_predicates.append(
                RuleBasedPredicate(
                    f"number-{number}",
                    1,
                    [number_value_arg],
                    self._get_number_eval_func(number),
                )
            )

        touching_grid_edge_pred = RuleBasedPredicate(
            "touching-grid-egde",
            3,
            [ex_num_arg, i_arg, j_arg],
            self._get_touching_grid_edge_eval_func(),
        )

        return self.Predicates(
            # input_pred=input_pred,
            empty_pred=empty_pred,
            above_pred=above_pred,
            below_pred=below_pred,
            left_of_pred=left_of_pred,
            right_of_pred=right_of_pred,
            inline_diagonally_above_right_pred=inline_diagonally_above_right_pred,
            inline_diagonally_above_left_pred=inline_diagonally_above_left_pred,
            inline_diagonally_below_right_pred=inline_diagonally_below_right_pred,
            inline_diagonally_below_left_pred=inline_diagonally_below_left_pred,
            inline_diagonally_right_left_pred=inline_diagonally_right_left_pred,
            inline_diagonally_left_right_pred=inline_diagonally_left_right_pred,
            inline_vertically_pred=inline_vertically_pred,
            inline_above_vertically_pred=inline_above_vertically_pred,
            inline_below_vertically_pred=inline_below_vertically_pred,
            inline_horizontally_pred=inline_horizontally_pred,
            inline_left_horizontally_pred=inline_left_horizontally_pred,
            inline_right_horizontally_pred=inline_right_horizontally_pred,
            mask_overlapping_pred=mask_overlapping_pred,
            mask_overlapping_and_colour_pred=mask_overlapping_and_colour_pred,
            mask_overlapping_expanded_to_grid_pred=mask_overlapping_expanded_to_grid_pred,  # noqa: E501
            mask_overlapping_and_colour_expanded_to_grid_pred=mask_overlapping_and_colour_expanded_to_grid_pred,  # noqa: E501
            mask_overlapping_repeated_grid_pred=mask_overlapping_repeated_grid_pred,
            mask_overlapping_and_colour_repeated_grid_pred=mask_overlapping_and_colour_repeated_grid_pred,  # noqa: E501
            mask_overlapping_moved_to_grid_pred=mask_overlapping_moved_to_grid_pred,
            inside_mask_not_overlapping_moved_to_grid_pred=inside_mask_not_overlapping_moved_to_grid_pred,  # noqa: E501
            mask_overlapping_top_inline_top_pred=mask_overlapping_top_inline_top_pred,
            mask_overlapping_bot_inline_bot_pred=mask_overlapping_bot_inline_bot_pred,
            mask_overlapping_left_inline_left_pred=mask_overlapping_left_inline_left_pred,  # noqa: E501
            mask_overlapping_right_inline_right_pred=mask_overlapping_right_inline_right_pred,  # noqa: E501
            mask_overlapping_bot_top_touching_pred=mask_overlapping_bot_top_touching_pred,  # noqa: E501
            mask_overlapping_top_bot_touching_pred=mask_overlapping_top_bot_touching_pred,  # noqa: E501
            mask_overlapping_right_left_touching_pred=mask_overlapping_right_left_touching_pred,  # noqa: E501
            mask_overlapping_left_right_touching_pred=mask_overlapping_left_right_touching_pred,  # noqa: E501
            mask_overlapping_gravity_to_shape_pred=mask_overlapping_gravity_to_shape_pred,  # noqa: E501
            # inside_mask_pred=inside_mask_pred,
            inside_blank_space_pred=inside_blank_space_pred,
            inside_mask_not_overlapping_pred=inside_mask_not_overlapping_pred,
            top_left_bottom_right_diag_pred=top_left_bottom_right_diag_pred,
            bottom_left_top_right_diag_pred=bottom_left_top_right_diag_pred,
            shape_colour_predicates=shape_colour_predicates,
            colour_predicates=colour_predicates,
            shape_size_predicates=shape_size_predicates,
            # inequality_predicates=inequality_predicates,
            shape_colour_count_predicates=shape_colour_count_predicates,
            grid_colour_count_predicates=grid_colour_count_predicates,
            shape_group_predicates=shape_group_predicates,
            shape_colour_pred=shape_colour_pred,
            shape_spanning_line_pred=shape_spanning_line_pred,
            shape_vertical_line_pred=shape_vertical_line_pred,
            shape_horizontal_line_pred=shape_horizontal_line_pred,
            vertical_center_distance_more_than_pred=vertical_center_distance_more_than_pred,  # noqa: E501
            vertical_center_distance_less_than_pred=vertical_center_distance_less_than_pred,  # noqa: E501
            horizontal_center_distance_more_than_pred=horizontal_center_distance_more_than_pred,  # noqa: E501
            horizontal_center_distance_less_than_pred=horizontal_center_distance_less_than_pred,  # noqa: E501
            vertical_edge_distance_more_than_pred=vertical_edge_distance_more_than_pred,  # noqa: E501
            vertical_edge_distance_less_than_pred=vertical_edge_distance_less_than_pred,  # noqa: E501
            horizontal_edge_distance_more_than_pred=horizontal_edge_distance_more_than_pred,  # noqa: E501
            horizontal_edge_distance_less_than_pred=horizontal_edge_distance_less_than_pred,  # noqa: E501
            vertical_grid_top_distance_more_than_pred=vertical_grid_top_distance_more_than_pred,  # noqa: E501
            vertical_grid_bottom_distance_more_than_pred=vertical_grid_bottom_distance_more_than_pred,  # noqa: E501
            vertical_grid_top_distance_less_than_pred=vertical_grid_top_distance_less_than_pred,  # noqa: E501
            vertical_grid_bottom_distance_less_than_pred=vertical_grid_bottom_distance_less_than_pred,  # noqa: E501
            horizontal_grid_left_distance_more_than_pred=horizontal_grid_left_distance_more_than_pred,  # noqa: E501
            horizontal_grid_right_distance_more_than_pred=horizontal_grid_right_distance_more_than_pred,  # noqa: E501
            horizontal_grid_left_distance_less_than_pred=horizontal_grid_left_distance_less_than_pred,  # noqa: E501
            horizontal_grid_right_distance_less_than_pred=horizontal_grid_right_distance_less_than_pred,  # noqa: E501
            number_value_predicates=number_value_predicates,
            touching_grid_edge_pred=touching_grid_edge_pred,
        )

    def _create_examples(
        self, possible_colours: list[Colour], variables: Variables
    ) -> list[tuple[bool, dict[str, Any]]]:
        examples: list[tuple[bool, dict[str, Any]]] = []
        for ex_number, example_grid in enumerate(self.outputs):
            for i in range(example_grid.shape[0]):
                for j in range(example_grid.shape[1]):
                    value: int = int(example_grid[i, j])
                    if value != 0:
                        positive_example = (
                            True,
                            {
                                variables.v1.name: ex_number,
                                variables.v2.name: Colour(value),
                                variables.v3.name: i,
                                variables.v4.name: j,
                            },
                        )
                        examples.append(positive_example)

                    # negative examples
                    for possible_colour in possible_colours:
                        if possible_colour.value != value:
                            negative_example = (
                                False,
                                {
                                    variables.v1.name: ex_number,
                                    variables.v2.name: possible_colour,
                                    variables.v3.name: i,
                                    variables.v4.name: j,
                                },
                            )
                            examples.append(negative_example)
        return examples

    def _create_background_knowledge(
        self,
        predicates: Predicates,
        possible_colours: list[Colour],
    ) -> BackgroundKnowledgeType:
        background_knowledge: Solver.BackgroundKnowledgeType = defaultdict(set)

        # add grid bk
        self._append_background_knowledge_for_grids(
            self.inputs, predicates.grid_colour_count_predicates, background_knowledge
        )

        # bk relating input shapes and i,j outputs
        self._append_background_knowledge_for_shapes(
            background_knowledge,
            self.outputs,
            self.inputs,
            self.inputs_shapes,
            predicates,
            possible_colours,
        )

        # Raw input bk
        self._append_background_knowledge_with_raw_input(
            background_knowledge, self.inputs, predicates
        )

        # Now we add the bk for the test inputs

        # bk for grids
        self._append_background_knowledge_for_grids(
            self.test_inputs,
            predicates.grid_colour_count_predicates,
            background_knowledge,
            ex_number_offset=self._TEST_EX_NUMBER_OFFSET,
        )

        # bk relating shapes
        if (
            self._get_non_pixel_max_shape_counts()
            <= self._SHAPE_COUNT_MAX_FOR_SHAPE_SHAPE_PREDS
        ):
            self._append_background_knowledge_for_shapes(
                background_knowledge,
                self.empty_test_outputs,
                self.test_inputs,
                self.test_inputs_shapes,
                predicates,
                possible_colours,
                ex_number_offset=self._TEST_EX_NUMBER_OFFSET,
            )

        self._append_background_knowledge_with_raw_input(
            background_knowledge,
            self.test_inputs,
            predicates,
            ex_number_offset=self._TEST_EX_NUMBER_OFFSET,
        )

        return background_knowledge

    def _get_max_num_arg_value(self) -> int:
        max_num = 0
        for output in self.outputs:
            max_num = max(output.shape[0], output.shape[1], max_num)

        return max_num

    @staticmethod
    def _extract_all_possible_colours(
        *list_arrays: list[NDArray[np.int16]],
    ) -> List[Colour]:
        possible_colours: set[Colour] = set()

        for arrays in list_arrays:
            for array in arrays:
                distinct_positives = np.unique(array[array > 0]).tolist()
                for positive in distinct_positives:
                    possible_colours.add(Colour(positive))

        return list(possible_colours)

    @staticmethod
    def _extract_all_possible_colour_counts_for_shapes(
        inputs_shapes: List[List[Shape]],
    ) -> List[int]:
        possible_counts: set[int] = set()
        for input_shapes in inputs_shapes:
            for input_shape in input_shapes:
                possible_counts.add(input_shape.colour_count)

        return list(possible_counts)

    @staticmethod
    def _extract_all_possible_sizes_for_shapes(
        inputs_shapes: List[List[Shape]],
    ) -> List[int]:
        possible_sizes: set[int] = set()
        for input_shapes in inputs_shapes:
            for input_shape in input_shapes:
                possible_sizes.add(input_shape.num_of_coloured_pixels)

        return list(possible_sizes)

    @staticmethod
    def _extract_all_possible_shape_groups(
        inputs_shapes: List[List[Shape]],
    ) -> List[str]:
        possible_groups: set[str] = set()
        for input_shapes in inputs_shapes:
            for input_shape in input_shapes:
                possible_groups = possible_groups.union(input_shape.shape_groups)

        return list(possible_groups)

    @staticmethod
    def _get_grid_colour_count(grid: NDArray[np.int16]):
        unique_elements = np.unique(grid)
        # remove 0
        unique_elements = [x for x in unique_elements if x != 0]
        return len(unique_elements)

    @staticmethod
    def _get_colour_eval_func(colour: Colour) -> Callable[..., bool]:
        return lambda colour_to_check: colour_to_check == colour

    @staticmethod
    def _get_number_eval_func(number: int) -> Callable[..., bool]:
        return lambda number_to_check: number_to_check == number

    @staticmethod
    def _get_less_than_eval_func(value: int) -> Callable[..., bool]:
        return lambda x: x < value

    @staticmethod
    def _get_more_than_eval_func(value: int) -> Callable[..., bool]:
        return lambda x: x > value

    @staticmethod
    def _get_grid_top_more_than_eval_func() -> Callable[..., bool]:
        return lambda _, i, value: i > value

    def _get_grid_bottom_more_than_eval_func(self) -> Callable[..., bool]:
        output_grid_heights = {}
        for ex, output_grid in enumerate(self.outputs):
            output_grid_heights[ex] = output_grid.shape[0]

        for ex, test_output_grid in enumerate(self.empty_test_outputs):
            output_grid_heights[
                ex + self._TEST_EX_NUMBER_OFFSET
            ] = test_output_grid.shape[0]

        return (
            lambda ex_number, i, value: output_grid_heights[ex_number] - i - 1 > value
        )

    @staticmethod
    def _get_grid_top_less_than_eval_func() -> Callable[..., bool]:
        return lambda _, i, value: i < value

    def _get_grid_bottom_less_than_eval_func(self) -> Callable[..., bool]:
        output_grid_heights = {}
        for ex, output_grid in enumerate(self.outputs):
            output_grid_heights[ex] = output_grid.shape[0]

        for ex, test_output_grid in enumerate(self.empty_test_outputs):
            output_grid_heights[
                ex + self._TEST_EX_NUMBER_OFFSET
            ] = test_output_grid.shape[0]

        return (
            lambda ex_number, i, value: output_grid_heights[ex_number] - i - 1 < value
        )

    @staticmethod
    def _get_grid_left_more_than_eval_func() -> Callable[..., bool]:
        return lambda _, j, value: j > value

    def _get_grid_right_more_than_eval_func(self) -> Callable[..., bool]:
        output_grid_widths = {}
        for ex, output_grid in enumerate(self.outputs):
            output_grid_widths[ex] = output_grid.shape[1]

        for ex, test_output_grid in enumerate(self.empty_test_outputs):
            output_grid_widths[
                ex + self._TEST_EX_NUMBER_OFFSET
            ] = test_output_grid.shape[1]

        return lambda ex_number, j, value: output_grid_widths[ex_number] - j - 1 > value

    @staticmethod
    def _get_grid_left_less_than_eval_func() -> Callable[..., bool]:
        return lambda _, j, value: j < value

    def _get_grid_right_less_than_eval_func(self) -> Callable[..., bool]:
        output_grid_widths = {}
        for ex, output_grid in enumerate(self.outputs):
            output_grid_widths[ex] = output_grid.shape[1]

        for ex, test_output_grid in enumerate(self.empty_test_outputs):
            output_grid_widths[
                ex + self._TEST_EX_NUMBER_OFFSET
            ] = test_output_grid.shape[1]

        return lambda ex_number, j, value: output_grid_widths[ex_number] - j - 1 < value

    def _get_touching_grid_edge_eval_func(self) -> Callable[..., bool]:
        output_grid_widths = {}
        output_grid_heights = {}
        for ex, output_grid in enumerate(self.outputs):
            output_grid_widths[ex] = output_grid.shape[1]
            output_grid_heights[ex] = output_grid.shape[0]

        for ex, test_output_grid in enumerate(self.empty_test_outputs):
            output_grid_widths[
                ex + self._TEST_EX_NUMBER_OFFSET
            ] = test_output_grid.shape[1]
            output_grid_heights[
                ex + self._TEST_EX_NUMBER_OFFSET
            ] = test_output_grid.shape[0]

        return (
            lambda ex_number, i, j: i == 0
            or j == 0
            or i == output_grid_heights[ex_number] - 1
            or j == output_grid_widths[ex_number] - 1
        )

    @staticmethod
    def _generate_shape_colour_pred_name(colour: Colour) -> str:
        return f"shape-colour-{colour.name}"

    @staticmethod
    def _generate_shape_colour_count_pred_name(colour_count: int) -> str:
        return f"shape-colour-count-{colour_count}"

    @staticmethod
    def _generate_shape_size_pred_name(size: int) -> str:
        return f"shape-size-{size}"

    @staticmethod
    def _generate_grid_colour_count_pred_name(colour_count: int) -> str:
        return f"grid-colour-count-{colour_count}"

    @staticmethod
    def _generate_shape_group_pred_name(group: str) -> str:
        return f"shape-group-{group}"

    @staticmethod
    def _generate_shape_name(example_number: int, is_input: bool, index: int) -> str:
        inp_str = "input" if is_input else "output"
        return f"{example_number}_{inp_str}_{index}"

    @staticmethod
    def _has_duplicate_names(predicate_list: list[Predicate]):
        names = set()
        for pred in predicate_list:
            name = pred.name
            if name in names:
                return True
            names.add(name)
        return False

    @staticmethod
    def _add_predicate_relations(predicates: Predicates):
        above_pred = predicates.above_pred
        below_pred = predicates.below_pred
        left_of_pred = predicates.left_of_pred
        right_of_pred = predicates.right_of_pred
        inline_diagonally_above_right_pred = (
            predicates.inline_diagonally_above_right_pred
        )
        inline_diagonally_above_left_pred = predicates.inline_diagonally_above_left_pred
        inline_diagonally_below_right_pred = (
            predicates.inline_diagonally_below_right_pred
        )
        inline_diagonally_below_left_pred = predicates.inline_diagonally_below_left_pred
        inline_diagonally_left_right_pred = predicates.inline_diagonally_left_right_pred
        inline_diagonally_right_left_pred = predicates.inline_diagonally_right_left_pred
        inline_vertically_pred = predicates.inline_vertically_pred
        inline_above_vertically_pred = predicates.inline_above_vertically_pred
        inline_below_vertically_pred = predicates.inline_below_vertically_pred
        inline_horizontally_pred = predicates.inline_horizontally_pred
        inline_left_horizontally_pred = predicates.inline_left_horizontally_pred
        inline_right_horizontally_pred = predicates.inline_right_horizontally_pred
        mask_overlapping_pred = predicates.mask_overlapping_pred
        mask_overlapping_bot_top_touching_pred = (
            predicates.mask_overlapping_bot_top_touching_pred
        )
        mask_overlapping_top_bot_touching_pred = (
            predicates.mask_overlapping_top_bot_touching_pred
        )
        mask_overlapping_right_left_touching_pred = (
            predicates.mask_overlapping_right_left_touching_pred
        )
        mask_overlapping_left_right_touching_pred = (
            predicates.mask_overlapping_left_right_touching_pred
        )
        mask_overlapping_gravity_to_shape_pred = (
            predicates.mask_overlapping_gravity_to_shape_pred
        )
        inside_blank_space_pred = predicates.inside_blank_space_pred
        inside_mask_not_overlapping_pred = predicates.inside_mask_not_overlapping_pred
        colour_predicates = predicates.colour_predicates
        shape_size_predicates = predicates.shape_size_predicates
        shape_colour_count_predicates = predicates.shape_colour_count_predicates
        grid_colour_count_predicates = predicates.grid_colour_count_predicates
        shape_vertical_line_pred = predicates.shape_vertical_line_pred
        shape_horizontal_line_pred = predicates.shape_horizontal_line_pred

        mask_overlapping_and_colour_expanded_to_grid_pred = (
            predicates.mask_overlapping_and_colour_expanded_to_grid_pred
        )  # noqa: E501
        mask_overlapping_expanded_to_grid_pred = (
            predicates.mask_overlapping_expanded_to_grid_pred
        )  # noqa: E501
        mask_overlapping_repeated_grid_pred = (
            predicates.mask_overlapping_repeated_grid_pred
        )  # noqa: E501
        mask_overlapping_and_colour_repeated_grid_pred = (
            predicates.mask_overlapping_and_colour_repeated_grid_pred
        )  # noqa: E501

        above_pred.add_incompatible_predicates(
            {
                below_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                inline_below_vertically_pred,
                inline_horizontally_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
            }
        )

        below_pred.add_incompatible_predicates(
            {
                above_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_above_vertically_pred,
                inline_horizontally_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
            }
        )

        left_of_pred.add_incompatible_predicates(
            {
                right_of_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_below_right_pred,
                inline_right_horizontally_pred,
                inline_vertically_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
            }
        )

        right_of_pred.add_incompatible_predicates(
            {
                left_of_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_left_pred,
                inline_left_horizontally_pred,
                inline_vertically_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
            }
        )

        inline_diagonally_above_right_pred.add_incompatible_predicates(
            {
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                below_pred,
                left_of_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                inline_vertically_pred,
                inline_horizontally_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_diagonally_above_left_pred.add_incompatible_predicates(
            {
                inline_diagonally_above_right_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                below_pred,
                right_of_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                inline_vertically_pred,
                inline_horizontally_pred,
                inline_diagonally_right_left_pred,
            }
        )

        inline_diagonally_below_right_pred.add_incompatible_predicates(
            {
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_left_pred,
                above_pred,
                left_of_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                inline_vertically_pred,
                inline_horizontally_pred,
                inline_diagonally_right_left_pred,
            }
        )

        inline_diagonally_below_left_pred.add_incompatible_predicates(
            {
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                above_pred,
                right_of_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                inline_vertically_pred,
                inline_horizontally_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_above_vertically_pred.add_incompatible_predicates(
            {
                inline_below_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                left_of_pred,
                right_of_pred,
                below_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                inline_horizontally_pred,
                inline_diagonally_right_left_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_below_vertically_pred.add_incompatible_predicates(
            {
                inline_above_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                left_of_pred,
                right_of_pred,
                above_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                inline_horizontally_pred,
                inline_diagonally_right_left_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_left_horizontally_pred.add_incompatible_predicates(
            {
                inline_right_horizontally_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                above_pred,
                below_pred,
                right_of_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                inline_vertically_pred,
                inline_diagonally_right_left_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_right_horizontally_pred.add_incompatible_predicates(
            {
                inline_left_horizontally_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                above_pred,
                below_pred,
                left_of_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                inline_vertically_pred,
                inline_diagonally_right_left_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_vertically_pred.add_incompatible_predicates(
            {
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                left_of_pred,
                right_of_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                inline_horizontally_pred,
                inline_diagonally_right_left_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_horizontally_pred.add_incompatible_predicates(
            {
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                above_pred,
                below_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
                inline_vertically_pred,
                inline_diagonally_right_left_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_diagonally_right_left_pred.add_incompatible_predicates(
            {
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                inline_vertically_pred,
                inline_horizontally_pred,
                inline_diagonally_left_right_pred,
            }
        )

        inline_diagonally_left_right_pred.add_incompatible_predicates(
            {
                inline_diagonally_above_right_pred,
                inline_diagonally_below_left_pred,
                inline_above_vertically_pred,
                inline_below_vertically_pred,
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
                inline_vertically_pred,
                inline_horizontally_pred,
                inline_diagonally_right_left_pred,
            }
        )

        mask_overlapping_bot_top_touching_pred.add_incompatible_predicates(
            {
                mask_overlapping_top_bot_touching_pred,
                mask_overlapping_right_left_touching_pred,
                mask_overlapping_left_right_touching_pred,
                mask_overlapping_gravity_to_shape_pred,
            }
        )

        mask_overlapping_top_bot_touching_pred.add_incompatible_predicates(
            {
                mask_overlapping_bot_top_touching_pred,
                mask_overlapping_right_left_touching_pred,
                mask_overlapping_left_right_touching_pred,
                mask_overlapping_gravity_to_shape_pred,
            }
        )

        mask_overlapping_right_left_touching_pred.add_incompatible_predicates(
            {
                mask_overlapping_bot_top_touching_pred,
                mask_overlapping_top_bot_touching_pred,
                mask_overlapping_left_right_touching_pred,
                mask_overlapping_gravity_to_shape_pred,
            }
        )

        mask_overlapping_left_right_touching_pred.add_incompatible_predicates(
            {
                mask_overlapping_bot_top_touching_pred,
                mask_overlapping_top_bot_touching_pred,
                mask_overlapping_right_left_touching_pred,
                mask_overlapping_gravity_to_shape_pred,
            }
        )

        mask_overlapping_gravity_to_shape_pred.add_incompatible_predicates(
            {
                mask_overlapping_bot_top_touching_pred,
                mask_overlapping_top_bot_touching_pred,
                mask_overlapping_right_left_touching_pred,
                mask_overlapping_left_right_touching_pred,
            }
        )

        mask_overlapping_pred.add_incompatible_predicates(
            {
                inside_blank_space_pred,
                mask_overlapping_repeated_grid_pred,
                mask_overlapping_expanded_to_grid_pred,
            }
        )

        mask_overlapping_and_colour_expanded_to_grid_pred.add_incompatible_predicate(
            mask_overlapping_and_colour_repeated_grid_pred
        )

        mask_overlapping_and_colour_repeated_grid_pred.add_incompatible_predicate(
            mask_overlapping_and_colour_expanded_to_grid_pred
        )

        mask_overlapping_expanded_to_grid_pred.add_incompatible_predicates(
            {mask_overlapping_pred, mask_overlapping_repeated_grid_pred}
        )

        mask_overlapping_repeated_grid_pred.add_incompatible_predicates(
            {mask_overlapping_pred, mask_overlapping_expanded_to_grid_pred}
        )

        inside_blank_space_pred.add_incompatible_predicate(mask_overlapping_pred)
        inside_mask_not_overlapping_pred.add_incompatible_predicate(
            mask_overlapping_pred
        )
        shape_vertical_line_pred.add_incompatible_predicate(shape_horizontal_line_pred)
        shape_horizontal_line_pred.add_incompatible_predicate(shape_vertical_line_pred)

        for col_pred_1 in colour_predicates:
            for col_pred_2 in colour_predicates:
                if col_pred_1 != col_pred_2:
                    col_pred_1.add_incompatible_predicate(col_pred_2)

        for shape_size_1 in shape_size_predicates:
            for shape_size_2 in shape_size_predicates:
                if shape_size_1 != shape_size_2:
                    shape_size_1.add_incompatible_predicate(shape_size_2)

        for shape_colour_count_1 in shape_colour_count_predicates:
            for shape_colour_count_2 in shape_colour_count_predicates:
                if shape_colour_count_1 != shape_colour_count_2:
                    shape_colour_count_1.add_incompatible_predicate(
                        shape_colour_count_2
                    )

        for grid_colour_count_1 in grid_colour_count_predicates:
            for grid_colour_count_2 in grid_colour_count_predicates:
                if grid_colour_count_1 != grid_colour_count_2:
                    grid_colour_count_1.add_incompatible_predicate(grid_colour_count_2)

        above_pred.add_more_specialised_predicates(
            {
                inline_above_vertically_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_above_left_pred,
            }
        )

        below_pred.add_more_specialised_predicates(
            {
                inline_below_vertically_pred,
                inline_diagonally_below_right_pred,
                inline_diagonally_below_left_pred,
            }
        )

        left_of_pred.add_more_specialised_predicates(
            {
                inline_left_horizontally_pred,
                inline_diagonally_above_left_pred,
                inline_diagonally_below_left_pred,
            }
        )

        right_of_pred.add_more_specialised_predicates(
            {
                inline_right_horizontally_pred,
                inline_diagonally_above_right_pred,
                inline_diagonally_below_right_pred,
            }
        )

        inside_mask_not_overlapping_pred.add_more_specialised_predicates(
            {inside_blank_space_pred}
        )

        inline_horizontally_pred.add_more_specialised_predicates(
            {
                inline_left_horizontally_pred,
                inline_right_horizontally_pred,
            }
        )

        inline_vertically_pred.add_more_specialised_predicates(
            {
                inline_above_vertically_pred,
                inline_below_vertically_pred,
            }
        )

        inline_diagonally_right_left_pred.add_more_specialised_predicates(
            {
                inline_diagonally_above_right_pred,
                inline_diagonally_below_left_pred,
            }
        )

        inline_diagonally_left_right_pred.add_more_specialised_predicates(
            {
                inline_diagonally_above_left_pred,
                inline_diagonally_below_right_pred,
            }
        )

    @staticmethod
    def ranges_overlap(start1, end1, start2, end2):
        return max(start1, start2) <= min(end1, end2)

    @lru_cache(maxsize=5000000)
    def _distance_until_overlap_vertically(
        self, input_shape_1: Shape, input_shape_2: Shape
    ) -> Optional[int]:
        if input_shape_1.is_mask_overlapping(input_shape_2):
            return None

        distance_up: Optional[int] = -1
        shifted_position = (input_shape_1.position[0] - 1, input_shape_1.position[1])
        shifted_shape = Shape(
            shifted_position, input_shape_1.mask, input_shape_1.shape_type
        )
        did_overlap = False
        while shifted_shape.position[0] >= 0:
            did_overlap = shifted_shape.is_mask_overlapping(input_shape_2)
            if did_overlap:
                distance_up += 1  # type: ignore
                break

            shifted_position = (
                shifted_shape.position[0] - 1,
                shifted_shape.position[1],
            )
            shifted_shape = Shape(
                shifted_position, input_shape_1.mask, input_shape_1.shape_type
            )
            distance_up -= 1  # type: ignore

        if not did_overlap:
            distance_up = None

        distance_down: Optional[int] = 1
        shifted_position = (input_shape_1.position[0] + 1, input_shape_1.position[1])
        shifted_shape = Shape(
            shifted_position, input_shape_1.mask, input_shape_1.shape_type
        )
        did_overlap = False
        while shifted_shape.position[0] <= input_shape_2.bottom_most:
            did_overlap = shifted_shape.is_mask_overlapping(input_shape_2)
            if did_overlap:
                distance_down -= 1  # type: ignore
                break

            shifted_position = (
                shifted_shape.position[0] + 1,
                shifted_shape.position[1],
            )
            shifted_shape = Shape(
                shifted_position, input_shape_1.mask, input_shape_1.shape_type
            )
            distance_down += 1  # type: ignore

        if not did_overlap:
            distance_down = None

        if distance_up is None and distance_down is None:
            return None

        if distance_up is None:
            return distance_down

        if distance_down is None:
            return distance_up

        if abs(distance_up) < distance_down:
            return distance_up

        return distance_down

    @lru_cache(maxsize=5000000)
    def _distance_until_overlap_horizontally(
        self, input_shape_1: Shape, input_shape_2: Shape
    ) -> Optional[int]:
        if input_shape_1.is_mask_overlapping(input_shape_2):
            return None

        distance_left: Optional[int] = -1
        shifted_position = (input_shape_1.position[0], input_shape_1.position[1] - 1)
        shifted_shape = Shape(
            shifted_position, input_shape_1.mask, input_shape_1.shape_type
        )
        did_overlap = False
        while shifted_shape.position[1] >= 0:
            did_overlap = shifted_shape.is_mask_overlapping(input_shape_2)
            if did_overlap:
                distance_left += 1  # type: ignore
                break
            shifted_position = (
                shifted_shape.position[0],
                shifted_shape.position[1] - 1,
            )
            shifted_shape = Shape(
                shifted_position, input_shape_1.mask, input_shape_1.shape_type
            )
            distance_left -= 1  # type: ignore

        if not did_overlap:
            distance_left = None

        distance_right: Optional[int] = 1
        shifted_position = (input_shape_1.position[0], input_shape_1.position[1] + 1)
        shifted_shape = Shape(
            shifted_position, input_shape_1.mask, input_shape_1.shape_type
        )
        did_overlap = False
        while shifted_shape.position[1] <= input_shape_2.right_most:
            did_overlap = shifted_shape.is_mask_overlapping(input_shape_2)
            if did_overlap:
                distance_right -= 1  # type: ignore
                break
            shifted_position = (
                shifted_shape.position[0],
                shifted_shape.position[1] + 1,
            )
            shifted_shape = Shape(
                shifted_position, input_shape_1.mask, input_shape_1.shape_type
            )
            distance_right += 1  # type: ignore

        if not did_overlap:
            distance_right = None

        if distance_left is None and distance_right is None:
            return None

        if distance_left is None:
            return distance_right

        if distance_right is None:
            return distance_left

        if abs(distance_left) < distance_right:
            return distance_left

        return distance_right

    @staticmethod
    def _filter_predicates(
        predicate_list: list[Predicate],
        background_knowledge: "Solver.BackgroundKnowledgeType",
    ):
        return [
            pred
            for pred in predicate_list
            if len(background_knowledge[pred.name]) > 0
            or isinstance(pred, RuleBasedPredicate)
        ]

    @staticmethod
    def _is_shape_expandable_to_grid(
        input_shape: Shape, output_grid: NDArray[np.int16]
    ) -> tuple[bool, int]:
        """
        returns if the shape is expandable to the grid and the expand multiplier
        """
        if input_shape.width == 1 and input_shape.height == 1:
            return False, 0

        grid_height = output_grid.shape[0]
        grid_width = output_grid.shape[1]
        for expand_multiplier in range(2, 16):
            if (
                input_shape.height * expand_multiplier == grid_height
                and input_shape.width * expand_multiplier == grid_width
            ):
                return True, expand_multiplier

            if (
                input_shape.height * expand_multiplier > grid_height
                or input_shape.width * expand_multiplier > grid_width
            ):
                return False, 0

        return False, 0

    @staticmethod
    def _append_background_knowledge_for_expandable_shapes(
        background_knowledge: BackgroundKnowledgeType,
        output_i: int,
        output_j: int,
        input_shape: Shape,
        output_grid: NDArray[np.int16],
        ex_number: int,
        input_shape_name: str,
        mask_overlapping_expanded_to_grid_pred: Predicate,
        mask_overlapping_and_colour_expanded_to_grid_pred: Predicate,
        possible_colours: list[Colour],
    ):
        is_expandable, expand_factor = Solver._is_shape_expandable_to_grid(
            input_shape, output_grid
        )  # noqa: E501

        if not is_expandable:
            return

        shrunk_i = output_i // expand_factor
        shrunk_j = output_j // expand_factor

        if input_shape.is_mask_overlapping_ij(
            shrunk_i + input_shape.position[0], shrunk_j + input_shape.position[1]
        ):
            background_knowledge[mask_overlapping_expanded_to_grid_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        for colour in possible_colours:
            if input_shape.is_mask_overlapping_and_colour_ij(
                shrunk_i + input_shape.position[0],
                shrunk_j + input_shape.position[1],
                colour,
            ):
                background_knowledge[
                    mask_overlapping_and_colour_expanded_to_grid_pred.name
                ].add(
                    (
                        ex_number,
                        colour,
                        output_i,
                        output_j,
                        input_shape_name,
                    )
                )

    @staticmethod
    def _is_shape_repeatable_in_grid(
        input_shape: Shape, output_grid: NDArray[np.int16]
    ) -> bool:
        """
        returns if the shape is repeatable to the grid
        """
        if input_shape.width == 1 and input_shape.height == 1:
            return False

        grid_height = output_grid.shape[0]
        grid_width = output_grid.shape[1]
        for expand_multiplier in range(2, 16):
            if (
                input_shape.height * expand_multiplier == grid_height
                and input_shape.width * expand_multiplier == grid_width
            ):
                return True

            if (
                input_shape.height * expand_multiplier > grid_height
                or input_shape.width * expand_multiplier > grid_width
            ):
                return False

        return False

    @staticmethod
    def _append_background_knowledge_for_repeatable_shapes(
        background_knowledge: BackgroundKnowledgeType,
        output_i: int,
        output_j: int,
        input_shape: Shape,
        output_grid: NDArray[np.int16],
        ex_number: int,
        input_shape_name: str,
        mask_overlapping_repeated_grid_pred: Predicate,
        mask_overlapping_and_colour_repeated_grid_pred: Predicate,
        possible_colours: list[Colour],
    ):
        is_repeatable = Solver._is_shape_repeatable_in_grid(
            input_shape, output_grid
        )  # noqa: E501

        if not is_repeatable:
            return

        shrunk_i = output_i % input_shape.width
        shrunk_j = output_j % input_shape.height

        if input_shape.is_mask_overlapping_ij(
            shrunk_i + input_shape.position[0], shrunk_j + input_shape.position[1]
        ):
            background_knowledge[mask_overlapping_repeated_grid_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        for colour in possible_colours:
            if input_shape.is_mask_overlapping_and_colour_ij(
                shrunk_i + input_shape.position[0],
                shrunk_j + input_shape.position[1],
                colour,
            ):
                background_knowledge[
                    mask_overlapping_and_colour_repeated_grid_pred.name
                ].add(
                    (
                        ex_number,
                        colour,
                        output_i,
                        output_j,
                        input_shape_name,
                    )
                )

    def _get_non_pixel_max_shape_counts(self) -> int:
        max_shape_count = 0
        for shapes in self.inputs_shapes:
            non_pixel_shapes = [
                shape for shape in shapes if not shape.shape_type == ShapeType.PIXEL
            ]
            max_shape_count = max(max_shape_count, len(non_pixel_shapes))

        for shapes in self.test_inputs_shapes:
            non_pixel_shapes = [
                shape for shape in shapes if not shape.shape_type == ShapeType.PIXEL
            ]
            max_shape_count = max(max_shape_count, len(non_pixel_shapes))

        return max_shape_count
