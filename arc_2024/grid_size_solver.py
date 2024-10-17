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
from arc_2024.representations.shape import Shape
from arc_2024.representations.shape_type import ShapeType


class GridSizeSolver:
    inputs: list[NDArray[np.int16]]
    outputs: list[NDArray[np.int16]]
    test_inputs: list[NDArray[np.int16]]
    inputs_shapes: List[List[Shape]]
    outputs_shapes: List[List[Shape]]
    test_inputs_shapes: List[List[Shape]]

    class ArgTypes(NamedTuple):
        example_number_arg: ArgType
        height_arg: ArgType
        width_arg: ArgType
        shape_arg: ArgType

    class Variables(NamedTuple):
        v1: Variable  # example number
        v2: Variable  # height
        v3: Variable  # width

    class Predicates(NamedTuple):
        input_pred: Predicate
        shape_width_pred: RuleBasedPredicate
        shape_height_pred: RuleBasedPredicate
        shape_dimensions_pred: RuleBasedPredicate
        rotated_shape_dimensions_pred: RuleBasedPredicate
        grid_width_pred: RuleBasedPredicate
        grid_height_pred: RuleBasedPredicate
        multiplyer_predicates: list[RuleBasedPredicate]
        divider_predicates: list[RuleBasedPredicate]
        number_value_predicates: list[RuleBasedPredicate]
        shape_group_predicates: list[RuleBasedPredicate]

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
        inputs_shapes: List[List[Shape]],
        outputs_shapes: List[List[Shape]],
        test_inputs_shapes: List[List[Shape]],
    ):
        for arr in inputs + outputs + test_inputs:
            if arr.ndim != 2:
                raise ValueError(
                    "Arrays in lists inputs, outputs & test_inputs must be 2D"
                )

        self.inputs = inputs
        self.outputs = outputs
        self.test_inputs = test_inputs
        self.inputs_shapes = inputs_shapes
        self.outputs_shapes = outputs_shapes
        self.test_inputs_shapes = test_inputs_shapes

    def solve(self, beam_width: int = 1) -> List[NDArray[np.int16]]:
        """
        This function solves the task.
        """
        arg_types = self._create_args_types()
        variables = self._create_variables(arg_types)
        target_literal = self._create_target_literal(arg_types, variables)
        predicates = self._create_predicates(arg_types)
        predicate_list = predicates.to_list()
        examples = self._create_examples(variables)
        background_knowledge = self._create_background_knowledge(predicates)

        foil = FOIL(
            target_literal,
            predicate_list,
            background_knowledge,
            beam_width=beam_width,
            type_extension_limit={
                arg_types.example_number_arg: 1,
                arg_types.width_arg: 1,
                arg_types.height_arg: 1,
                arg_types.shape_arg: 3,
            },
            max_clause_length=4,
        )
        foil.fit(examples)

        return self._calculate_results(foil, variables)

    def _get_possible_shapes_func(self) -> Callable[[dict[str, Any]], list]:
        shapes = {}
        for example_number, input_shapes in enumerate(self.inputs_shapes):
            shapes[example_number] = [
                shape for shape in input_shapes if shape.shape_type != ShapeType.PIXEL
            ]

        for example_number, input_shapes in enumerate(self.test_inputs_shapes):
            shapes[example_number + self._TEST_EX_NUMBER_OFFSET] = [
                shape for shape in input_shapes if shape.shape_type != ShapeType.PIXEL
            ]

        def get_possible_shapes(example: dict[str, Any]) -> list:
            # V1 is set as example number
            return shapes[example["V1"]]

        return get_possible_shapes

    def _create_background_knowledge(
        self,
        predicates: Predicates,
    ) -> BackgroundKnowledgeType:
        background_knowledge: GridSizeSolver.BackgroundKnowledgeType = defaultdict(set)

        # Raw input bk
        self._append_background_knowledge_with_raw_input(
            background_knowledge, self.inputs, predicates
        )

        # Now we add the bk for the test inputs
        self._append_background_knowledge_with_raw_input(
            background_knowledge,
            self.test_inputs,
            predicates,
            ex_number_offset=self._TEST_EX_NUMBER_OFFSET,
        )

        return background_knowledge

    def _append_background_knowledge_with_raw_input(
        self,
        background_knowledge: BackgroundKnowledgeType,
        inputs: list[NDArray[np.int16]],
        predicates: Predicates,
        ex_number_offset=0,
    ) -> None:
        for ex_number, example_grid in enumerate(inputs):
            offset_ex_number = ex_number + ex_number_offset
            background_knowledge[predicates.input_pred.name].add(
                (offset_ex_number, example_grid.shape[0], example_grid.shape[1])
            )

    def _calculate_results(
        self,
        foil: FOIL,
        variables: Variables,
    ) -> List[NDArray[np.int16]]:
        # we iteratively populate the test outputs
        test_outputs: List[NDArray[np.int16]] = []
        for test_number in range(len(self.test_inputs)):
            offset_test_number = test_number + self._TEST_EX_NUMBER_OFFSET
            found = False
            for i in range(1, 31):
                for j in range(1, 31):
                    if foil.predict(
                        {
                            variables.v1.name: offset_test_number,
                            variables.v2.name: i,
                            variables.v3.name: j,
                        }
                    ):
                        test_outputs.append(np.zeros((i, j)))
                        found = True
                        break
                if found:
                    break

            if not found:
                raise Exception("No solution found for grid size")

        return test_outputs

    def _create_args_types(self) -> ArgTypes:
        # args in target predicate & body predicates
        example_number_arg = ArgType(
            "example_number", list(range(len(self.inputs_shapes)))
        )
        height_arg = ArgType("height", list(range(1, 31)))
        width_arg = ArgType("width", list(range(1, 31)))

        # args not in target but in possible body predicates
        shape_arg = ArgType(
            "shape",
            possible_values_fn=self._get_possible_shapes_func(),
        )

        return self.ArgTypes(
            example_number_arg=example_number_arg,
            height_arg=height_arg,
            width_arg=width_arg,
            shape_arg=shape_arg,
        )

    def _create_variables(self, arg_types: ArgTypes) -> Variables:
        return self.Variables(
            v1=Variable("V1", arg_types.example_number_arg),
            v2=Variable("V2", arg_types.height_arg),
            v3=Variable("V3", arg_types.width_arg),
        )

    def _create_target_literal(
        self, arg_types: ArgTypes, variables: "GridSizeSolver.Variables"
    ) -> Literal:
        target_predicate = Predicate(
            "output",
            3,
            [
                arg_types.example_number_arg,
                arg_types.height_arg,
                arg_types.width_arg,
            ],
        )  # noqa: E501
        return Literal(
            predicate=target_predicate,
            args=[variables.v1, variables.v2, variables.v3],
        )

    def _create_predicates(
        self,
        arg_types: ArgTypes,
    ) -> Predicates:
        ex_num_arg = arg_types.example_number_arg
        height_arg = arg_types.height_arg
        width_arg = arg_types.width_arg
        shape_arg = arg_types.shape_arg

        input_pred = Predicate(
            "input", 3, [ex_num_arg, height_arg, width_arg]
        )  # noqa: E501

        shape_width_pred = RuleBasedPredicate(
            "shape-width",
            3,
            [ex_num_arg, shape_arg, width_arg],
            lambda _, shape, width: shape.width == width,
        )

        shape_height_pred = RuleBasedPredicate(
            "shape-height",
            3,
            [ex_num_arg, shape_arg, height_arg],
            lambda _, shape, height: shape.height == height,
        )

        shape_dimensions_pred = RuleBasedPredicate(
            "shape-dimensions",
            4,
            [ex_num_arg, shape_arg, height_arg, width_arg],
            lambda _, shape, height, width: shape.height == height
            and shape.width == width,
        )

        rotated_shape_dimensions_pred = RuleBasedPredicate(
            "rotated-shape-dimensions",
            4,
            [ex_num_arg, shape_arg, height_arg, width_arg],
            lambda _, shape, height, width: shape.height == width
            and shape.width == height,
        )

        grid_width_pred = RuleBasedPredicate(
            "grid-width", 2, [ex_num_arg, width_arg], self._get_is_grid_width_func()
        )

        grid_height_pred = RuleBasedPredicate(
            "grid-height", 2, [ex_num_arg, height_arg], self._get_is_grid_height_func()
        )

        # equal_pred = RuleBasedPredicate(
        #     "equal",
        #     2,
        #     [numeric_arg, numeric_arg],
        #     lambda num_1, num_2 : num_1 == num_2
        # )

        shape_group_predicates: list[RuleBasedPredicate] = []
        for shape_group in self._extract_all_possible_shape_groups(self.inputs_shapes):
            shape_group_pred = RuleBasedPredicate(
                f"shape-group-{shape_group}",
                2,
                [ex_num_arg, shape_arg],
                self._get_is_shape_group_func(shape_group),
            )
            shape_group_predicates.append(shape_group_pred)

        multiplyer_predicates: list[RuleBasedPredicate] = []
        for multiplyer in list(range(2, 31)):
            # multiplyer_predicates.append(RuleBasedPredicate(
            #     f"multiply-by-{multiplyer}",
            #     2,
            #     [numeric_arg, numeric_arg],
            #     self._get_multiplyer_eval_func(multiplyer)
            # ))

            multiplyer_predicates.append(
                RuleBasedPredicate(
                    f"shape-width-multiply-by-{multiplyer}",
                    3,
                    [ex_num_arg, shape_arg, width_arg],
                    self._get_shape_width_multiplyer_eval_func(multiplyer),
                )
            )

            multiplyer_predicates.append(
                RuleBasedPredicate(
                    f"shape-height-multiply-by-{multiplyer}",
                    3,
                    [ex_num_arg, shape_arg, height_arg],
                    self._get_shape_height_multiplyer_eval_func(multiplyer),
                )
            )

            multiplyer_predicates.append(
                RuleBasedPredicate(
                    f"grid-width-multiply-by-{multiplyer}",
                    2,
                    [ex_num_arg, width_arg],
                    self._get_grid_width_multiplyer_eval_func(multiplyer),
                )
            )

            multiplyer_predicates.append(
                RuleBasedPredicate(
                    f"grid-height-multiply-by-{multiplyer}",
                    2,
                    [ex_num_arg, height_arg],
                    self._get_grid_height_multiplyer_eval_func(multiplyer),
                )
            )

        divider_predicates: list[RuleBasedPredicate] = []
        for divider in list(range(2, 31)):
            # divider_predicates.append(RuleBasedPredicate(
            #     f"divide-by-{divider}",
            #     2,
            #     [numeric_arg, numeric_arg],
            #     self._get_divider_eval_func(divider)
            # ))
            divider_predicates.append(
                RuleBasedPredicate(
                    f"shape-width-divide-by-{divider}",
                    3,
                    [ex_num_arg, shape_arg, width_arg],
                    self._get_shape_width_divider_eval_func(divider),
                )
            )

            divider_predicates.append(
                RuleBasedPredicate(
                    f"shape-height-divide-by-{divider}",
                    3,
                    [ex_num_arg, shape_arg, height_arg],
                    self._get_shape_height_divider_eval_func(divider),
                )
            )

            divider_predicates.append(
                RuleBasedPredicate(
                    f"grid-width-divide-by-{divider}",
                    2,
                    [ex_num_arg, width_arg],
                    self._get_grid_width_divider_eval_func(divider),
                )
            )

            divider_predicates.append(
                RuleBasedPredicate(
                    f"grid-height-divide-by-{divider}",
                    2,
                    [ex_num_arg, height_arg],
                    self._get_grid_height_divider_eval_func(divider),
                )
            )

        number_value_predicates: list[RuleBasedPredicate] = []
        for number in list(range(1, 31)):
            number_value_predicates.append(
                RuleBasedPredicate(
                    f"number-width-{number}",
                    1,
                    [width_arg],
                    self._get_number_eval_func(number),
                )
            )
            number_value_predicates.append(
                RuleBasedPredicate(
                    f"number-height-{number}",
                    1,
                    [height_arg],
                    self._get_number_eval_func(number),
                )
            )

        return self.Predicates(
            input_pred=input_pred,
            shape_width_pred=shape_width_pred,
            shape_height_pred=shape_height_pred,
            shape_dimensions_pred=shape_dimensions_pred,
            rotated_shape_dimensions_pred=rotated_shape_dimensions_pred,
            grid_width_pred=grid_width_pred,
            grid_height_pred=grid_height_pred,
            multiplyer_predicates=multiplyer_predicates,
            divider_predicates=divider_predicates,
            number_value_predicates=number_value_predicates,
            shape_group_predicates=shape_group_predicates,
        )

    def _create_examples(
        self, variables: Variables
    ) -> list[tuple[bool, dict[str, Any]]]:
        examples: list[tuple[bool, dict[str, Any]]] = []
        for ex_number, example_grid in enumerate(self.outputs):
            for i in list(range(1, 31)):
                for j in list(range(1, 31)):
                    examples.append(
                        (
                            i == example_grid.shape[0] and j == example_grid.shape[1],
                            {
                                variables.v1.name: ex_number,
                                variables.v2.name: i,
                                variables.v3.name: j,
                            },
                        )
                    )
        return examples

    def _get_is_grid_height_func(self) -> Callable[..., bool]:
        def is_grid_height(example_number: int, height: int) -> bool:
            if example_number >= self._TEST_EX_NUMBER_OFFSET:
                return (
                    self.test_inputs[
                        example_number - self._TEST_EX_NUMBER_OFFSET
                    ].shape[0]
                    == height
                )
            return self.inputs[example_number].shape[0] == height

        return is_grid_height

    def _get_is_grid_width_func(self) -> Callable[..., bool]:
        def is_grid_width(example_number: int, width: int) -> bool:
            if example_number >= self._TEST_EX_NUMBER_OFFSET:
                return (
                    self.test_inputs[
                        example_number - self._TEST_EX_NUMBER_OFFSET
                    ].shape[1]
                    == width
                )
            return self.inputs[example_number].shape[1] == width

        return is_grid_width

    def _get_grid_height_multiplyer_eval_func(
        self, multiplyer: int
    ) -> Callable[..., bool]:
        def is_grid_height(example_number: int, num: int) -> bool:
            if example_number >= self._TEST_EX_NUMBER_OFFSET:
                return (
                    self.test_inputs[
                        example_number - self._TEST_EX_NUMBER_OFFSET
                    ].shape[0]
                    == num * multiplyer
                )
            return self.inputs[example_number].shape[0] == num * multiplyer

        return is_grid_height

    def _get_grid_width_multiplyer_eval_func(
        self, multiplyer: int
    ) -> Callable[..., bool]:
        def is_grid_width(example_number: int, num: int) -> bool:
            if example_number >= self._TEST_EX_NUMBER_OFFSET:
                return (
                    self.test_inputs[
                        example_number - self._TEST_EX_NUMBER_OFFSET
                    ].shape[1]
                    == num * multiplyer
                )
            return self.inputs[example_number].shape[1] == num * multiplyer

        return is_grid_width

    def _get_grid_height_divider_eval_func(self, divider: int) -> Callable[..., bool]:
        def is_grid_height(example_number: int, num: int) -> bool:
            if example_number >= self._TEST_EX_NUMBER_OFFSET:
                return (
                    self.test_inputs[
                        example_number - self._TEST_EX_NUMBER_OFFSET
                    ].shape[0]
                    == num / divider
                )
            return self.inputs[example_number].shape[0] == num / divider

        return is_grid_height

    def _get_grid_width_divider_eval_func(self, divider: int) -> Callable[..., bool]:
        def is_grid_width(example_number: int, num: int) -> bool:
            if example_number >= self._TEST_EX_NUMBER_OFFSET:
                return (
                    self.test_inputs[
                        example_number - self._TEST_EX_NUMBER_OFFSET
                    ].shape[1]
                    == num / divider
                )
            return self.inputs[example_number].shape[1] == num / divider

        return is_grid_width

    @staticmethod
    def _get_number_eval_func(number: int) -> Callable[..., bool]:
        return lambda number_to_check: number_to_check == number

    @staticmethod
    def _get_multiplyer_eval_func(multiplyer: int) -> Callable[..., bool]:
        return (
            lambda number_to_check, number_to_multiply: number_to_check
            == multiplyer * number_to_multiply
        )

    @staticmethod
    def _get_shape_width_multiplyer_eval_func(multiplyer: int) -> Callable[..., bool]:
        return lambda _, shape, num: shape.width * multiplyer == num

    @staticmethod
    def _get_shape_height_multiplyer_eval_func(multiplyer: int) -> Callable[..., bool]:
        return lambda _, shape, num: shape.height * multiplyer == num

    @staticmethod
    def _get_shape_width_divider_eval_func(divider: int) -> Callable[..., bool]:
        return lambda _, shape, num: shape.width / divider == num

    @staticmethod
    def _get_shape_height_divider_eval_func(divider: int) -> Callable[..., bool]:
        return lambda _, shape, num: shape.height / divider == num

    @staticmethod
    def _get_divider_eval_func(divider: int) -> Callable[..., bool]:
        return (
            lambda number_to_check, number_to_divide: number_to_check
            == number_to_divide / divider
        )

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
    def _get_is_shape_group_func(shape_group: str) -> Callable[..., bool]:
        return lambda _, shape: shape_group in shape.shape_groups
