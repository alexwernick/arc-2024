from typing import Any, Callable, List

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
    inputs: list[NDArray[np.int16]]
    outputs: list[NDArray[np.int16]]
    test_inputs: list[NDArray[np.int16]]

    _INPUT_PRED_NAME = "input"
    _EMPTY_PRED_NAME = "empty"
    _ABOVE_PRED_NAME = "above"
    _BELOW_PRED_NAME = "below"
    _LEFT_OF_PRED_NAME = "left-of"
    _RIGHT_OF_PRED_NAME = "right-of"
    _INLINE_HORIZONTALLY_ABOVE_RIGHT_PRED_NAME = "inline-horizontally-above-right"
    _INLINE_HORIZONTALLY_ABOVE_LEFT_PRED_NAME = "inline-horizontally-above-left"
    _INLINE_HORIZONTALLY_BELOW_RIGHT_PRED_NAME = "inline-horizontally-below-right"
    _INLINE_HORIZONTALLY_BELOW_LEFT_PRED_NAME = "inline-horizontally-below-left"
    _INLINE_ABOVE_VERTICALLY_PRED_NAME = "inline-above-vertically"
    _INLINE_BELOW_VERTICALLY_PRED_NAME = "inline-below-vertically"
    _INLINE_LEFT_HORIZONTALLY_PRED_NAME = "inline-left-horizontally"
    _INLINE_RIGHT_HORIZONTALLY_PRED_NAME = "inline-right-horizontally"
    _MASK_OVERLAPPING_PRED_NAME = "mask-overlapping"
    _INSIDE_PRED_NAME = "inside"
    _INSIDE_NOT_OVERLAPPING_PRED_NAME = "inside-not-overlapping"
    _TOP_LEFT_BOTTOM_RIGHT_DIAG_PRED_NAME = "top-left-bottom-right-diag"
    _BOTTOM_LEFT_TOP_RIGHT_DIAG_PRED_NAME = "bottom-left-top-right-diag"
    _MIXED_COLOUR_SHAPE_PRED_NAME = "mixed-shape"
    _SINGLE_COLOUR_SHAPE_PRED_NAME = "single-colour-shape"

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
        # First we look at possible arg types
        possible_colours: list[Colour] = self._extract_all_possible_colours(
            inputs_shapes, outputs_shapes
        )  # noqa: E501

        # Args for target literal
        colour_type_arg = ArgType("colour", possible_colours)
        example_number_arg = ArgType("example_number", list(range(len(inputs_shapes))))
        i_arg = ArgType("i", possible_values_fn=self._get_possible_i_values_func())
        j_arg = ArgType("j", possible_values_fn=self._get_possible_j_values_func())

        # Other args for predicates to be searched for the body
        shape_arg = ArgType(
            "shape",
            possible_values_fn=self._get_possible_shapes_func(
                inputs_shapes, outputs_shapes
            ),
        )  # noqa: E501
        # colour_count_arg = ArgType("colour_count", possible_colour_counts)

        target_predicate = Predicate(
            "output", 4, [example_number_arg, colour_type_arg, i_arg, j_arg]
        )  # noqa: E501

        V1 = Variable("V1", example_number_arg)
        V2 = Variable("V2", colour_type_arg)
        V3 = Variable("V3", i_arg)
        V4 = Variable("V4", j_arg)

        target_literal = Literal(predicate=target_predicate, args=[V1, V2, V3, V4])

        # Available predicates
        input_pred = Predicate(
            self._INPUT_PRED_NAME,
            4,
            [example_number_arg, colour_type_arg, i_arg, j_arg],
        )  # noqa: E501
        empty_pred = Predicate(
            self._EMPTY_PRED_NAME, 3, [example_number_arg, i_arg, j_arg]
        )
        # above_pred = Predicate(
        #     self._ABOVE_PRED_NAME, 4, [example_number_arg, i_arg, j_arg, shape_arg]
        # )
        # below_pred = Predicate(
        #     self._BELOW_PRED_NAME, 4, [example_number_arg, i_arg, j_arg, shape_arg]
        # )
        # left_of_pred = Predicate(
        #     self._LEFT_OF_PRED_NAME, 4, [example_number_arg, i_arg, j_arg, shape_arg]
        # )
        # right_of_pred = Predicate(
        #     self._RIGHT_OF_PRED_NAME, 4, [example_number_arg, i_arg, j_arg, shape_arg]
        # )
        # inline_horizontally_above_right_pred = Predicate(
        #     self._INLINE_HORIZONTALLY_ABOVE_RIGHT_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inline_horizontally_above_left_pred = Predicate(
        #     self._INLINE_HORIZONTALLY_ABOVE_LEFT_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inline_horizontally_below_right_pred = Predicate(
        #     self._INLINE_HORIZONTALLY_BELOW_RIGHT_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inline_horizontally_below_left_pred = Predicate(
        #     self._INLINE_HORIZONTALLY_BELOW_LEFT_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inline_above_vertically_pred = Predicate(
        #     self._INLINE_ABOVE_VERTICALLY_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inline_below_vertically_pred = Predicate(
        #     self._INLINE_BELOW_VERTICALLY_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inline_left_horizontally_pred = Predicate(
        #     self._INLINE_LEFT_HORIZONTALLY_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inline_right_horizontally_pred = Predicate(
        #     self._INLINE_RIGHT_HORIZONTALLY_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # mask_overlapping_pred = Predicate(
        #     self._MASK_OVERLAPPING_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )  # noqa: E501
        # inside_prod = Predicate(
        #     self._INSIDE_PRED_NAME, 4, [example_number_arg, i_arg, j_arg, shape_arg]
        # )
        # inside_not_overlapping_pred = Predicate(
        #     self._INSIDE_NOT_OVERLAPPING_PRED_NAME,
        #     4,
        #     [example_number_arg, i_arg, j_arg, shape_arg],
        # )
        top_left_bottom_right_diag_pred = RuleBasedPredicate(
            self._TOP_LEFT_BOTTOM_RIGHT_DIAG_PRED_NAME,
            3,
            [example_number_arg, i_arg, j_arg],
            self._get_top_left_bottom_right_diag_eval_func(),
        )
        bottom_left_top_right_diag_pred = RuleBasedPredicate(
            self._BOTTOM_LEFT_TOP_RIGHT_DIAG_PRED_NAME,
            3,
            [example_number_arg, i_arg, j_arg],
            self._get_bottom_left_top_right_diag_eval_func(),
        )
        # mixed_colour_shape_pred = Predicate(
        #     self._MIXED_COLOUR_SHAPE_PRED_NAME, 1, [shape_arg]
        # )
        # single_colour_shape_pred = Predicate(
        #     self._SINGLE_COLOUR_SHAPE_PRED_NAME, 1, [shape_arg]
        # )

        shape_colour_predicates: list[Predicate] = []
        colour_predicates: list[Predicate] = []

        for colour in possible_colours:
            shape_colour_predicates.append(
                Predicate(f"shape-colour-{colour.name}", 1, [shape_arg])
            )

            colour_predicates.append(
                RuleBasedPredicate(
                    f"colour-{colour.name}",
                    1,
                    [colour_type_arg],
                    self._get_colour_eval_func(colour),
                )
            )

        # Once we know the grid size of the outputs we need to get the biggest here
        max_is: set[int] = set()
        max_js: set[int] = set()
        for input in self.inputs:
            max_is.add(input.shape[0])
            max_js.add(input.shape[1])

        max_i = max(max_is)
        max_j = max(max_js)

        inequality_predicates: list[Predicate] = []
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
                Predicate(f"shape-colour-count-{possible_count}", 1, [shape_arg])
            )

        grid_colour_count_predicates: list[Predicate] = []
        for possible_count in self._extract_all_possible_colour_counts_for_grids():
            grid_colour_count_predicates.append(
                Predicate(
                    f"grid-colour-count-{possible_count}", 1, [example_number_arg]
                )
            )

        predicates = [
            input_pred,
            empty_pred,
            # inside_prod,
            # above_pred,
            # below_pred,
            # left_of_pred,
            # right_of_pred,
            # inline_horizontally_above_right_pred,
            # inline_horizontally_above_left_pred,
            # inline_horizontally_below_right_pred,
            # inline_horizontally_below_left_pred,
            # inline_above_vertically_pred,
            # inline_below_vertically_pred,
            # inline_left_horizontally_pred,
            # inline_right_horizontally_pred,
            # mask_overlapping_pred,
            # inside_not_overlapping_pred,
            bottom_left_top_right_diag_pred,
            top_left_bottom_right_diag_pred,
            # I don't think these have any vlaue as we have
            # the colour count which is more informative
            # mixed_colour_shape_pred,
            # single_colour_shape_pred
        ]

        # predicates.extend(shape_colour_predicates)
        predicates.extend(colour_predicates)
        predicates.extend(shape_colour_count_predicates)
        predicates.extend(grid_colour_count_predicates)
        predicates.extend(inequality_predicates)

        # Examples
        examples: list[tuple[bool, dict[str, Any]]] = []
        for ex_number, example_grid in enumerate(self.outputs):
            for i in range(example_grid.shape[0]):
                for j in range(example_grid.shape[1]):
                    value: int = int(example_grid[i, j])
                    if value != 0:
                        positive_example = (
                            True,
                            {
                                V1.name: ex_number,
                                V2.name: Colour(value),
                                V3.name: i,
                                V4.name: j,
                            },
                        )
                        examples.append(positive_example)

                    # negative examples
                    for possible_colour in possible_colours:
                        if possible_colour.value != value:
                            negative_example = (
                                False,
                                {
                                    V1.name: ex_number,
                                    V2.name: possible_colour,
                                    V3.name: i,
                                    V4.name: j,
                                },
                            )
                            examples.append(negative_example)

        # Background facts for predicates
        background_knowledge: dict[str, set[tuple]] = {}
        for predicate in predicates:
            background_knowledge[predicate.name] = set()

        # add grid style bk
        self._append_background_knowledge_for_grids(
            self.inputs, grid_colour_count_predicates, background_knowledge
        )

        # bk relating input and output shapes
        # bk relating shapes
        # We assume here input and output grid are the same size
        for ex_number, (input_shapes, output_grid) in enumerate(
            zip(inputs_shapes, self.outputs)
        ):
            for i in range(output_grid.shape[0]):
                for j in range(output_grid.shape[1]):
                    for input_shape_index, input_shape in enumerate(input_shapes):
                        # for now lets ignore pixels
                        if input_shape.shape_type == ShapeType.PIXEL:
                            continue

                        input_shape_name = self._generate_shape_name(
                            ex_number, True, input_shape_index
                        )

                        self._append_background_knowledge_for_shapes(
                            background_knowledge,
                            i,
                            j,
                            input_shape,
                            ex_number,
                            input_shape_name,
                            shape_colour_predicates,
                            shape_colour_count_predicates,
                        )

        # Raw input bk
        for ex_number, example_grid in enumerate(self.inputs):
            for i in range(example_grid.shape[0]):
                for j in range(example_grid.shape[1]):
                    value = int(example_grid[i, j])
                    if value == 0:
                        background_knowledge[empty_pred.name].add((ex_number, i, j))
                    else:
                        background_knowledge[input_pred.name].add(
                            (ex_number, Colour(value), i, j)
                        )

        foil = FOIL(target_literal, predicates, background_knowledge)
        foil.fit(examples)

        return self._calculate_results(
            foil,
            test_inputs_shapes,
            shape_colour_predicates,
            shape_colour_count_predicates,
            grid_colour_count_predicates,
            possible_colours,
        )

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

    def _extract_all_possible_colour_counts_for_grids(self) -> List[int]:
        possible_counts: set[int] = set()
        for input in self.inputs:
            count = self._get_grid_colour_count(input)
            possible_counts.add(count)

        return list(possible_counts)

    @staticmethod
    def _get_grid_colour_count(grid: NDArray[np.int16]):
        unique_elements = np.unique(grid)
        # remove 0
        unique_elements = [x for x in unique_elements if x != 0]
        return len(unique_elements)

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

    @staticmethod
    def _get_colour_eval_func(colour: Colour) -> Callable[..., bool]:
        return lambda colour_to_check: colour_to_check == colour

    @staticmethod
    def _get_less_than_eval_func(value: int) -> Callable[..., bool]:
        return lambda x: x < value

    @staticmethod
    def _get_more_than_eval_func(value: int) -> Callable[..., bool]:
        return lambda x: x > value

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
        test_inputs_shapes: List[List[Shape]],
        shape_colour_predicates: list[Predicate],
        shape_colour_count_predicates: list[Predicate],
        grid_colour_count_predicates: list[Predicate],
        possible_colours: list[Colour],
    ) -> List[NDArray[np.int16]]:
        # We offset test numbers by 100 to avoid conflicts with the examples
        test_number_offset = 100

        for test_number, test_grid in enumerate(self.test_inputs):
            offset_test_number = test_number + test_number_offset
            for i in range(test_grid.shape[0]):
                for j in range(test_grid.shape[1]):
                    value = int(test_grid[i, j])
                    if value == 0:
                        foil.background_knowledge[self._EMPTY_PRED_NAME].add(
                            (offset_test_number, i, j)
                        )
                    else:
                        foil.background_knowledge[self._INPUT_PRED_NAME].add(
                            (offset_test_number, Colour(value), i, j)
                        )

        # bk for grids
        self._append_background_knowledge_for_grids(
            self.test_inputs,
            grid_colour_count_predicates,
            foil.background_knowledge,
            test_number_offset=test_number_offset,
        )

        # bk relating shapes
        # We assume here input and output grid are the same size
        for test_number, (input_shapes, input_grid) in enumerate(
            zip(test_inputs_shapes, self.test_inputs)
        ):
            offset_test_number = test_number + test_number_offset
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    for input_shape_index, input_shape in enumerate(input_shapes):
                        input_shape_name = self._generate_shape_name(
                            offset_test_number, True, input_shape_index
                        )
                        self._append_background_knowledge_for_shapes(
                            foil.background_knowledge,
                            i,
                            j,
                            input_shape,
                            offset_test_number,
                            input_shape_name,
                            shape_colour_predicates,
                            shape_colour_count_predicates,
                        )

        # we iteratively populate the test outputs
        test_outputs: List[NDArray[np.int16]] = []
        for test_number, input_grid in enumerate(self.test_inputs):
            offset_test_number = test_number + test_number_offset
            test_output = np.zeros_like(input_grid)
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    for possible_colour in possible_colours:
                        if foil.predict(
                            {
                                "V1": offset_test_number,
                                "V2": possible_colour,
                                "V3": i,
                                "V4": j,
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
        test_number_offset=0,
    ):
        for ex, input_grid in enumerate(inputs):
            unique_colour_count = self._get_grid_colour_count(input_grid)
            for grid_colour_count_pred in grid_colour_count_predicates:
                # Do this better
                if (
                    f"grid-colour-count-{unique_colour_count}"
                    == grid_colour_count_pred.name
                    and grid_colour_count_pred.name in background_knowledge
                ):
                    background_knowledge[grid_colour_count_pred.name].add(
                        (ex + test_number_offset,)
                    )

    def _append_background_knowledge_for_shapes(
        self,
        background_knowledge: dict[str, set[tuple]],
        output_i: int,
        output_j: int,
        input_shape: Shape,
        ex_number: int,
        input_shape_name: str,
        shape_colour_predicates: list[Predicate],
        shape_colour_count_predicates: list[Predicate],
    ) -> None:
        if (
            input_shape.is_above_i(output_i)
            and self._ABOVE_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._ABOVE_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_below_i(output_i)
            and self._BELOW_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._BELOW_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_left_of_j(output_j)
            and self._LEFT_OF_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._LEFT_OF_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_right_of_j(output_j)
            and self._RIGHT_OF_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._RIGHT_OF_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_horizontally_above_right_ij(output_i, output_j)
            and self._INLINE_HORIZONTALLY_ABOVE_RIGHT_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_HORIZONTALLY_ABOVE_RIGHT_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_horizontally_above_left_ij(output_i, output_j)
            and self._INLINE_HORIZONTALLY_ABOVE_LEFT_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_HORIZONTALLY_ABOVE_LEFT_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_horizontally_below_right_ij(output_i, output_j)
            and self._INLINE_HORIZONTALLY_BELOW_RIGHT_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_HORIZONTALLY_BELOW_RIGHT_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_horizontally_below_left_ij(output_i, output_j)
            and self._INLINE_HORIZONTALLY_BELOW_LEFT_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_HORIZONTALLY_BELOW_LEFT_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_above_vertically_ij(output_i, output_j)
            and self._INLINE_ABOVE_VERTICALLY_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_ABOVE_VERTICALLY_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_below_vertically_ij(output_i, output_j)
            and self._INLINE_BELOW_VERTICALLY_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_BELOW_VERTICALLY_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_left_horizontally_ij(output_i, output_j)
            and self._INLINE_LEFT_HORIZONTALLY_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_LEFT_HORIZONTALLY_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_inline_right_horizontally_ij(output_i, output_j)
            and self._INLINE_RIGHT_HORIZONTALLY_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INLINE_RIGHT_HORIZONTALLY_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_ij_inside(output_i, output_j)
            and self._INSIDE_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INSIDE_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_mask_overlapping_ij(output_i, output_j)
            and self._MASK_OVERLAPPING_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._MASK_OVERLAPPING_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.is_ij_inside_not_overlapping(output_i, output_j)
            and self._INSIDE_NOT_OVERLAPPING_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._INSIDE_NOT_OVERLAPPING_PRED_NAME].add(
                (
                    ex_number,
                    output_i,
                    output_j,
                    input_shape_name,
                )
            )

        if (
            input_shape.shape_type == ShapeType.MIXED_COLOUR
            and self._MIXED_COLOUR_SHAPE_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._MIXED_COLOUR_SHAPE_PRED_NAME].add(
                (input_shape_name,)
            )

        if (
            input_shape.shape_type == ShapeType.SINGLE_COLOUR
            and self._SINGLE_COLOUR_SHAPE_PRED_NAME in background_knowledge
        ):
            background_knowledge[self._SINGLE_COLOUR_SHAPE_PRED_NAME].add(
                (input_shape_name,)
            )

        if input_shape.colour is not None:
            for colour_pred in shape_colour_predicates:
                # Do this better
                if (
                    f"shape-colour-{input_shape.colour.name}" == colour_pred.name
                    and colour_pred.name in background_knowledge
                ):
                    background_knowledge[colour_pred.name].add((input_shape_name,))

        for shape_colour_pred in shape_colour_count_predicates:
            # Do this better
            if (
                f"shape-colour-count-{input_shape.colour_count}"
                == shape_colour_pred.name
                and shape_colour_pred.name in background_knowledge
            ):
                background_knowledge[shape_colour_pred.name].add((input_shape_name,))

    @staticmethod
    def _generate_shape_name(example_number: int, is_input: bool, index: int) -> str:
        inp_str = "input" if is_input else "output"
        return f"{example_number}_{inp_str}_{index}"
