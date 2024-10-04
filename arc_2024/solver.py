from typing import Any, Callable, List

import numpy as np
from numpy.typing import NDArray

from arc_2024.inductive_logic_programming.first_order_logic import (
    ArgType,
    Literal,
    Predicate,
    Variable,
)
from arc_2024.inductive_logic_programming.FOIL import FOIL
from arc_2024.representations.colour import Colour
from arc_2024.representations.interpreter import Interpreter
from arc_2024.representations.shape import Shape


class Solver:
    inputs: list[NDArray[np.int16]]
    outputs: list[NDArray[np.int16]]
    test_inputs: list[NDArray[np.int16]]

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
            test_input_shapes,
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
            "input", 4, [example_number_arg, colour_type_arg, i_arg, j_arg]
        )  # noqa: E501
        empty_pred = Predicate("empty", 3, [example_number_arg, i_arg, j_arg])
        in_shape_pred = Predicate("in-shape", 3, [i_arg, j_arg, shape_arg])
        exact_match_pred = Predicate("exact-match", 2, [shape_arg, shape_arg])
        above_pred = Predicate("above", 2, [shape_arg, shape_arg])
        below_pred = Predicate("below", 2, [shape_arg, shape_arg])
        left_of_pred = Predicate("left-of", 2, [shape_arg, shape_arg])
        right_of_pred = Predicate("right-of", 2, [shape_arg, shape_arg])
        inline_horizontally_above_right_pred = Predicate(
            "inline-horizontally-above-right", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        inline_horizontally_above_left_pred = Predicate(
            "inline-horizontally-above-left", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        inline_horizontally_below_right_pred = Predicate(
            "inline-horizontally-below-right", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        inline_horizontally_below_left_pred = Predicate(
            "inline-horizontally-below-left", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        inline_above_vertically_pred = Predicate(
            "inline-above-vertically", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        inline_below_vertically_pred = Predicate(
            "inline-below-vertically", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        inline_left_horizontally_pred = Predicate(
            "inline-left-horizontally", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        inline_right_horizontally_pred = Predicate(
            "inline-right-horizontally", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        mask_overlapping_pred = Predicate(
            "mask-overlapping", 2, [shape_arg, shape_arg]
        )  # noqa: E501
        #  is_inside_pred = Predicate("is-inside", 2, [shape_arg, shape_arg])
        same_colour_pred = Predicate("same-colour", 2, [shape_arg, shape_arg])

        predicates = [
            exact_match_pred,
            above_pred,
            below_pred,
            left_of_pred,
            right_of_pred,
            inline_horizontally_above_right_pred,
            inline_horizontally_above_left_pred,
            inline_horizontally_below_right_pred,
            inline_horizontally_below_left_pred,
            inline_above_vertically_pred,
            inline_below_vertically_pred,
            inline_left_horizontally_pred,
            inline_right_horizontally_pred,
            mask_overlapping_pred,
            same_colour_pred,
            in_shape_pred,
            input_pred,
            empty_pred,
        ]

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

        # bk relating input and output shapes
        for ex_number, (input_shapes, output_shapes) in enumerate(
            zip(inputs_shapes, outputs_shapes)
        ):
            for i, input_shape in enumerate(input_shapes):
                for j, output_shape in enumerate(output_shapes):
                    input_shape_name = self._generate_shape_name(ex_number, True, i)
                    output_shape_name = self._generate_shape_name(ex_number, False, j)

                    if input_shape.is_exact_match(output_shape):
                        background_knowledge[exact_match_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_above(output_shape):
                        background_knowledge[above_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_below(output_shape):
                        background_knowledge[below_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_left_of(output_shape):
                        background_knowledge[left_of_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_right_of(output_shape):
                        background_knowledge[right_of_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_inline_horizontally_above_right(output_shape):
                        background_knowledge[
                            inline_horizontally_above_right_pred.name
                        ].add((input_shape_name, output_shape_name))

                    if input_shape.is_inline_horizontally_above_left(output_shape):
                        background_knowledge[
                            inline_horizontally_above_left_pred.name
                        ].add((input_shape_name, output_shape_name))

                    if input_shape.is_inline_horizontally_below_right(output_shape):
                        background_knowledge[
                            inline_horizontally_below_right_pred.name
                        ].add((input_shape_name, output_shape_name))

                    if input_shape.is_inline_horizontally_below_left(output_shape):
                        background_knowledge[
                            inline_horizontally_below_left_pred.name
                        ].add((input_shape_name, output_shape_name))

                    if input_shape.is_inline_above_vertically(output_shape):
                        background_knowledge[inline_above_vertically_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_inline_below_vertically(output_shape):
                        background_knowledge[inline_below_vertically_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_inline_left_horizontally(output_shape):
                        background_knowledge[inline_left_horizontally_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_inline_right_horizontally(output_shape):
                        background_knowledge[inline_right_horizontally_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_mask_overlapping(output_shape):
                        background_knowledge[mask_overlapping_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    if input_shape.is_same_colour(output_shape):
                        background_knowledge[same_colour_pred.name].add(
                            (input_shape_name, output_shape_name)
                        )

                    for pixel in input_shape.all_pixels():
                        background_knowledge[in_shape_pred.name].add(
                            (pixel[0], pixel[1], input_shape_name)
                        )

                    for pixel in output_shape.all_pixels():
                        background_knowledge[in_shape_pred.name].add(
                            (pixel[0], pixel[1], output_shape_name)
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

        return self.inputs

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

    # We make and assumption here that the input and output
    # grid shapes are the same size. We will need to learn
    # use as well for other puzzles
    def _get_possible_i_values_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, input in enumerate(self.inputs):
            possible_values[example_number] = list(range(input.shape[0]))

        def get_possible_i_values(example: dict[str, Any]):
            # V1 is set as example number
            example_number = example["V1"]
            possible_values[example_number]

        return get_possible_i_values

    def _get_possible_j_values_func(self) -> Callable[[dict[str, Any]], list]:
        possible_values = {}
        for example_number, input in enumerate(self.inputs):
            possible_values[example_number] = list(range(input.shape[1]))

        def get_possible_j_values(example: dict[str, Any]):
            # V1 is set as example number
            example_number = example["V1"]
            possible_values[example_number]

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
            output_shapes = [
                self._generate_shape_name(example_number, False, i)
                for i in range(len(outputs_shapes[example_number]))
            ]
            possible_values[example_number] = input_shapes + output_shapes

        def get_possible_shapes(example: dict[str, Any]):
            # V1 is set as example number
            example_number = example["V1"]
            possible_values[example_number]

        return get_possible_shapes

    @staticmethod
    def _generate_shape_name(example_number: int, is_input: bool, index: int) -> str:
        inp_str = "input" if is_input else "output"
        return f"{example_number}_{inp_str}_{index}"
