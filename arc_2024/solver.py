from collections import defaultdict
from typing import Any, Callable, DefaultDict, List, NamedTuple

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
from arc_2024.representations.interpreter import Interpreter
from arc_2024.representations.shape import Shape, ShapeType


class Solver:
    class ArgTypes(NamedTuple):
        colour_type_arg: ArgType
        example_number_arg: ArgType
        i_arg: ArgType
        j_arg: ArgType
        shape_arg: ArgType

    class Variables(NamedTuple):
        v1: Variable
        v2: Variable
        v3: Variable
        v4: Variable

    class Predicates(NamedTuple):
        input_pred: Predicate
        empty_pred: Predicate
        above_pred: Predicate
        below_pred: Predicate
        left_of_pred: Predicate
        right_of_pred: Predicate
        inline_horizontally_above_right_pred: Predicate
        inline_horizontally_above_left_pred: Predicate
        inline_horizontally_below_right_pred: Predicate
        inline_horizontally_below_left_pred: Predicate
        inline_above_vertically_pred: Predicate
        inline_below_vertically_pred: Predicate
        inline_left_horizontally_pred: Predicate
        inline_right_horizontally_pred: Predicate
        mask_overlapping_pred: Predicate
        inside_pred: Predicate
        inside_not_overlapping_pred: Predicate
        top_left_bottom_right_diag_pred: Predicate
        bottom_left_top_right_diag_pred: Predicate
        shape_colour_predicates: list[Predicate]
        colour_predicates: list[RuleBasedPredicate]
        shape_size_predicates: list[Predicate]
        inequality_predicates: list[RuleBasedPredicate]
        shape_colour_count_predicates: list[Predicate]
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

    inputs: list[NDArray[np.int16]]
    outputs: list[NDArray[np.int16]]
    test_inputs: list[NDArray[np.int16]]

    # We offset test numbers by 100 to avoid conflicts with the examples
    _TEST_EX_NUMBER_OFFSET = 100

    def __init__(
        self,
        inputs: list[NDArray[np.int16]],
        outputs: list[NDArray[np.int16]],
        test_inputs: list[NDArray[np.int16]],
    ):
        for arr in inputs + outputs + test_inputs:
            if arr.ndim != 2:
                raise ValueError(
                    "Arrays in lists inputs, outputs & test_inputs must be 2D"
                )

        self.inputs = inputs
        self.outputs = outputs
        self.test_inputs = test_inputs

    def solve(self) -> List[NDArray[np.int16]]:
        """
        This function solves the task.
        """
        interpreter = Interpreter(self.inputs, self.outputs, self.test_inputs)

        (
            inputs_shapes,
            outputs_shapes,
            test_inputs_shapes,
        ) = interpreter.interpret_shapes()

        # We prepare data for the FOIL algorithm
        possible_colours: list[Colour] = self._extract_all_possible_colours(
            inputs_shapes, outputs_shapes
        )  # noqa: E501

        arg_types = self._create_args_types(
            possible_colours, inputs_shapes, outputs_shapes
        )
        variables = self._create_variables(arg_types)
        target_literal = self._create_target_literal(arg_types, variables)
        predicates = self._create_predicates(arg_types, possible_colours, inputs_shapes)
        predicate_list = predicates.to_list()
        examples = self._create_examples(possible_colours, variables)

        # Background facts for predicates
        background_knowledge = self._create_background_knowledge(
            predicates, inputs_shapes, test_inputs_shapes
        )

        foil = FOIL(target_literal, predicate_list, background_knowledge)
        foil.fit(examples)

        return self._calculate_results(
            foil,
            possible_colours,
            variables,
        )

    def _extract_all_possible_colour_counts_for_grids(self) -> List[int]:
        possible_counts: set[int] = set()
        for input in self.inputs:
            count = self._get_grid_colour_count(input)
            possible_counts.add(count)

        return list(possible_counts)

    # We make and assumption here that the input and output
    # grid shapes are the same size. We will need to learn
    # use as well for other puzzles
    def _get_possible_i_values_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, input in enumerate(self.inputs):
            possible_values[example_number] = list(range(input.shape[0]))

        def get_possible_i_values(example: dict[str, Any]) -> list:
            # V1 is set as example number
            example_number = example["V1"]
            return possible_values[example_number]

        return get_possible_i_values

    def _get_possible_j_values_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, input in enumerate(self.inputs):
            possible_values[example_number] = list(range(input.shape[1]))

        def get_possible_j_values(example: dict[str, Any]) -> list:
            # V1 is set as example number
            example_number = example["V1"]
            return possible_values[example_number]

        return get_possible_j_values

    def _get_possible_shapes_func(
        self, inputs_shapes: List[List[Shape]], outputs_shapes: List[List[Shape]]
    ) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, _ in enumerate(self.inputs):
            input_shapes = [
                self._generate_shape_name(example_number, True, i)
                for i in range(len(inputs_shapes[example_number]))
            ]
            output_shapes: list[str] = []
            # output_shapes = [
            #     self._generate_shape_name(example_number, False, i)
            #     for i in range(len(outputs_shapes[example_number]))
            # ]
            possible_values[example_number] = input_shapes + output_shapes

        def get_possible_shapes(example: dict[str, Any]) -> list:
            # V1 is set as example number
            example_number = example["V1"]
            return possible_values[example_number]

        return get_possible_shapes

    def _get_top_left_bottom_right_diag_eval_func(self) -> Callable[..., bool]:
        # we assume input grids size equal outputs
        test_number_offset = 100
        is_square: dict[int, bool] = {}
        for ex, input_grid in enumerate(self.inputs):
            if input_grid.shape[0] == input_grid.shape[1]:
                is_square[ex] = True
            else:
                is_square[ex] = False

        for ex, input_grid in enumerate(self.test_inputs):
            if input_grid.shape[0] == input_grid.shape[1]:
                is_square[ex + test_number_offset] = True
            else:
                is_square[ex + test_number_offset] = False

        return lambda ex_number, i, j: is_square[ex_number] and i == j

    def _get_bottom_left_top_right_diag_eval_func(self) -> Callable[..., bool]:
        # we assume input grids size equal outputs
        test_number_offset = 100
        is_square: dict[int, bool] = {}
        heights: dict[int, int] = {}
        for ex, input_grid in enumerate(self.inputs):
            heights[ex] = input_grid.shape[0]
            if input_grid.shape[0] == input_grid.shape[1]:
                is_square[ex] = True
            else:
                is_square[ex] = False

        for ex, input_grid in enumerate(self.test_inputs):
            heights[ex + test_number_offset] = input_grid.shape[0]
            if input_grid.shape[0] == input_grid.shape[1]:
                is_square[ex + test_number_offset] = True
            else:
                is_square[ex + test_number_offset] = False

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
        for test_number, input_grid in enumerate(self.test_inputs):
            offset_test_number = test_number + self._TEST_EX_NUMBER_OFFSET
            test_output = np.zeros_like(input_grid)
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    for possible_colour in possible_colours:
                        if foil.predict(
                            {
                                variables.v1.name: offset_test_number,
                                variables.v2.name: possible_colour,
                                variables.v3.name: i,
                                variables.v4.name: j,
                            }
                        ):
                            test_output[i, j] = possible_colour.value
                            break
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
                    else:
                        background_knowledge[predicates.input_pred.name].add(
                            (offset_ex_number, Colour(value), i, j)
                        )

    def _append_background_knowledge_for_shapes(
        self,
        background_knowledge: BackgroundKnowledgeType,
        outputs: list[NDArray[np.int16]],
        inputs_shapes: list[list[Shape]],
        predicates: Predicates,
        ex_number_offset: int = 0,
    ) -> None:
        for ex_number, (input_shapes, output_grid) in enumerate(
            zip(inputs_shapes, outputs)
        ):
            ex_test_number = ex_number + ex_number_offset
            for i in range(output_grid.shape[0]):
                for j in range(output_grid.shape[1]):
                    for input_shape_index, input_shape in enumerate(input_shapes):
                        # for now lets ignore pixels
                        if input_shape.shape_type == ShapeType.PIXEL:
                            continue

                        input_shape_name = self._generate_shape_name(
                            ex_test_number, True, input_shape_index
                        )

                        self._append_background_knowledge_for_shape(
                            background_knowledge,
                            i,
                            j,
                            input_shape,
                            ex_test_number,
                            input_shape_name,
                            predicates,
                        )

    def _append_background_knowledge_for_shape(
        self,
        background_knowledge: BackgroundKnowledgeType,
        output_i: int,
        output_j: int,
        input_shape: Shape,
        ex_number: int,
        input_shape_name: str,
        predicates: Predicates,
    ) -> None:
        if input_shape.is_above_i(output_i):
            background_knowledge[predicates.above_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_below_i(output_i):
            background_knowledge[predicates.below_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_left_of_j(output_j):
            background_knowledge[predicates.left_of_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_right_of_j(output_j):
            background_knowledge[predicates.right_of_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_horizontally_above_right_ij(output_i, output_j):
            background_knowledge[
                predicates.inline_horizontally_above_right_pred.name
            ].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_horizontally_above_left_ij(output_i, output_j):
            background_knowledge[
                predicates.inline_horizontally_above_left_pred.name
            ].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_horizontally_below_right_ij(output_i, output_j):
            background_knowledge[
                predicates.inline_horizontally_below_right_pred.name
            ].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_horizontally_below_left_ij(output_i, output_j):
            background_knowledge[
                predicates.inline_horizontally_below_left_pred.name
            ].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_above_vertically_ij(output_i, output_j):
            background_knowledge[predicates.inline_above_vertically_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_below_vertically_ij(output_i, output_j):
            background_knowledge[predicates.inline_below_vertically_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_left_horizontally_ij(output_i, output_j):
            background_knowledge[predicates.inline_left_horizontally_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_inline_right_horizontally_ij(output_i, output_j):
            background_knowledge[predicates.inline_right_horizontally_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if input_shape.is_ij_inside(output_i, output_j):
            background_knowledge[predicates.inside_pred.name].add(
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

        if input_shape.is_ij_inside_not_overlapping(output_i, output_j):
            background_knowledge[predicates.inside_not_overlapping_pred.name].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        # if input_shape.shape_type == ShapeType.MIXED_COLOUR:
        #     background_knowledge[self._MIXED_COLOUR_SHAPE_PRED_NAME].add(
        #         (input_shape_name,)
        #     )

        # if input_shape.shape_type == ShapeType.SINGLE_COLOUR:
        #     background_knowledge[self._SINGLE_COLOUR_SHAPE_PRED_NAME].add(
        #         (input_shape_name,)
        #     )

        if input_shape.colour is not None:
            for colour_pred in predicates.shape_colour_predicates:
                if (
                    self._generate_shape_colour_pred_name(input_shape.colour)
                    == colour_pred.name
                ):
                    background_knowledge[colour_pred.name].add((input_shape_name,))

        for shape_colour_pred in predicates.shape_colour_count_predicates:
            if (
                self._generate_shape_colour_count_pred_name(input_shape.colour_count)
                == shape_colour_pred.name
            ):
                background_knowledge[shape_colour_pred.name].add(
                    (ex_number, input_shape_name)
                )

        for shape_size_pred in predicates.shape_size_predicates:
            if (
                self._generate_shape_size_pred_name(input_shape.num_of_coloured_pixels)
                == shape_size_pred.name
            ):
                background_knowledge[shape_size_pred.name].add(
                    (ex_number, input_shape_name)
                )

    def _create_args_types(
        self,
        possible_colours: list[Colour],
        inputs_shapes: list[list[Shape]],
        outputs_shapes: list[list[Shape]],
    ) -> ArgTypes:
        # args in target predicate & body predicates
        colour_type_arg = ArgType("colour", possible_colours)
        example_number_arg = ArgType("example_number", list(range(len(inputs_shapes))))
        i_arg = ArgType("i", possible_values_fn=self._get_possible_i_values_func())
        j_arg = ArgType("j", possible_values_fn=self._get_possible_j_values_func())

        # args not in target but in possible body predicates
        shape_arg = ArgType(
            "shape",
            possible_values_fn=self._get_possible_shapes_func(
                inputs_shapes, outputs_shapes
            ),
        )  # noqa: E501

        return self.ArgTypes(
            colour_type_arg=colour_type_arg,
            example_number_arg=example_number_arg,
            i_arg=i_arg,
            j_arg=j_arg,
            shape_arg=shape_arg,
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
        inputs_shapes: list[list[Shape]],
    ) -> Predicates:
        ex_num_arg = arg_types.example_number_arg
        colour_type_arg = arg_types.colour_type_arg
        i_arg = arg_types.i_arg
        j_arg = arg_types.j_arg
        shape_arg = arg_types.shape_arg

        input_pred = Predicate(
            "input", 4, [ex_num_arg, colour_type_arg, i_arg, j_arg]
        )  # noqa: E501
        empty_pred = Predicate("empty", 3, [ex_num_arg, i_arg, j_arg])
        above_pred = Predicate("above", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        below_pred = Predicate("below", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        left_of_pred = Predicate("left-of", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        right_of_pred = Predicate("right-of", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        inline_horizontally_above_right_pred = Predicate(
            "inline-horizontally-above-right",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_horizontally_above_left_pred = Predicate(
            "inline-horizontally-above-left",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_horizontally_below_right_pred = Predicate(
            "inline-horizontally-below-right",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
        inline_horizontally_below_left_pred = Predicate(
            "inline-horizontally-below-left",
            4,
            [ex_num_arg, i_arg, j_arg, shape_arg],
        )  # noqa: E501
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
        )  # noqa: E501
        inside_pred = Predicate("inside", 4, [ex_num_arg, i_arg, j_arg, shape_arg])
        inside_not_overlapping_pred = Predicate(
            "inside-not-overlapping",
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
        for size in self._extract_all_possible_sizes_for_shapes(inputs_shapes):
            shape_size_predicates.append(
                Predicate(
                    self._generate_shape_size_pred_name(size),
                    2,
                    [ex_num_arg, shape_arg],
                )
            )

        max_is: set[int] = set()
        max_js: set[int] = set()
        for output in self.outputs:
            max_is.add(output.shape[0])
            max_js.add(output.shape[1])

        max_i = max(max_is)
        max_j = max(max_js)

        inequality_predicates: list[RuleBasedPredicate] = []
        for i in range(max_i):
            if i != max_i:
                inequality_predicates.append(
                    RuleBasedPredicate(
                        f"i-greater-than-{i}",
                        1,
                        [i_arg],
                        self._get_more_than_eval_func(i),
                    )
                )

            if i != 0:
                inequality_predicates.append(
                    RuleBasedPredicate(
                        f"i-less-than-{i}",
                        1,
                        [i_arg],
                        self._get_less_than_eval_func(i),
                    )
                )

        for j in range(max_j):
            if i != max_j:
                inequality_predicates.append(
                    RuleBasedPredicate(
                        f"j-greater-than-{j}",
                        1,
                        [j_arg],
                        self._get_more_than_eval_func(j),
                    )
                )

            if j != 0:
                inequality_predicates.append(
                    RuleBasedPredicate(
                        f"j-less-than-{j}",
                        1,
                        [j_arg],
                        self._get_less_than_eval_func(j),
                    )
                )

        shape_colour_count_predicates: list[Predicate] = []
        for possible_count in self._extract_all_possible_colour_counts_for_shapes(
            inputs_shapes
        ):
            shape_colour_count_predicates.append(
                Predicate(
                    self._generate_shape_colour_count_pred_name(possible_count),
                    1,
                    [shape_arg],
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

        return self.Predicates(
            input_pred=input_pred,
            empty_pred=empty_pred,
            above_pred=above_pred,
            below_pred=below_pred,
            left_of_pred=left_of_pred,
            right_of_pred=right_of_pred,
            inline_horizontally_above_right_pred=inline_horizontally_above_right_pred,
            inline_horizontally_above_left_pred=inline_horizontally_above_left_pred,
            inline_horizontally_below_right_pred=inline_horizontally_below_right_pred,
            inline_horizontally_below_left_pred=inline_horizontally_below_left_pred,
            inline_above_vertically_pred=inline_above_vertically_pred,
            inline_below_vertically_pred=inline_below_vertically_pred,
            inline_left_horizontally_pred=inline_left_horizontally_pred,
            inline_right_horizontally_pred=inline_right_horizontally_pred,
            mask_overlapping_pred=mask_overlapping_pred,
            inside_pred=inside_pred,
            inside_not_overlapping_pred=inside_not_overlapping_pred,
            top_left_bottom_right_diag_pred=top_left_bottom_right_diag_pred,
            bottom_left_top_right_diag_pred=bottom_left_top_right_diag_pred,
            shape_colour_predicates=shape_colour_predicates,
            colour_predicates=colour_predicates,
            shape_size_predicates=shape_size_predicates,
            inequality_predicates=inequality_predicates,
            shape_colour_count_predicates=shape_colour_count_predicates,
            grid_colour_count_predicates=grid_colour_count_predicates,
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
        inputs_shapes: list[list[Shape]],
        test_inputs_shapes: list[list[Shape]],
    ) -> BackgroundKnowledgeType:
        background_knowledge: Solver.BackgroundKnowledgeType = defaultdict(set)
        # for predicate in predicates.to_list():
        #     background_knowledge[predicate.name] = set()

        # add grid bk
        self._append_background_knowledge_for_grids(
            self.inputs, predicates.grid_colour_count_predicates, background_knowledge
        )

        # bk relating input shapes and i,j outputs
        self._append_background_knowledge_for_shapes(
            background_knowledge, self.outputs, inputs_shapes, predicates
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
        # We assume here input and output grid are the same size
        # so we pass the input grid to the output paramater
        self._append_background_knowledge_for_shapes(
            background_knowledge,
            self.test_inputs,
            test_inputs_shapes,
            predicates,
            ex_number_offset=self._TEST_EX_NUMBER_OFFSET,
        )

        self._append_background_knowledge_with_raw_input(
            background_knowledge,
            self.test_inputs,
            predicates,
            ex_number_offset=self._TEST_EX_NUMBER_OFFSET,
        )

        return background_knowledge

    @staticmethod
    def _extract_all_possible_colours(
        inputs_shapes: List[List[Shape]], outputs_shapes: List[List[Shape]]
    ) -> List[Colour]:
        possible_colours: set[Colour] = set()
        for input_shapes in inputs_shapes:
            for input_shape in input_shapes:
                if input_shape.colour is not None:
                    possible_colours.add(input_shape.colour)

        for output_shapes in outputs_shapes:
            for output_shape in output_shapes:
                if output_shape.colour is not None:
                    possible_colours.add(output_shape.colour)

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
    def _get_grid_colour_count(grid: NDArray[np.int16]):
        unique_elements = np.unique(grid)
        # remove 0
        unique_elements = [x for x in unique_elements if x != 0]
        return len(unique_elements)

    @staticmethod
    def _get_colour_eval_func(colour: Colour) -> Callable[..., bool]:
        return lambda colour_to_check: colour_to_check == colour

    @staticmethod
    def _get_less_than_eval_func(value: int) -> Callable[..., bool]:
        return lambda x: x < value

    @staticmethod
    def _get_more_than_eval_func(value: int) -> Callable[..., bool]:
        return lambda x: x > value

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
    def _generate_shape_name(example_number: int, is_input: bool, index: int) -> str:
        inp_str = "input" if is_input else "output"
        return f"{example_number}_{inp_str}_{index}"
