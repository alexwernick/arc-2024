import json
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import pytest

from arc_2024.representations.interpreter import Interpreter
from arc_2024.representations.rotatable_mask_shape import RotatableMaskShape
from arc_2024.representations.shape import Shape, ShapeType


class InterpreterTestCase(NamedTuple):
    interpreter: Interpreter
    expected_input_shapes: List[List[Shape]]
    expected_output_shapes: List[List[Shape]]
    expected_test_input_shapes: List[List[Shape]]


@pytest.fixture
def interpreter() -> Interpreter:
    return create_interpreter_for_task("6d75e8bb")


@pytest.fixture
def interpreter_dictiorary() -> Dict[str, InterpreterTestCase]:
    interpreter_dictiorary: Dict[str, InterpreterTestCase] = {}
    interpreter_dictiorary["6d75e8bb"] = create_test_case_6d75e8bb()
    interpreter_dictiorary["6e19193c"] = create_test_case_6e19193c()

    return interpreter_dictiorary


def create_test_case_6d75e8bb() -> InterpreterTestCase:
    expected_input_shapes: List[List[Shape]] = [[] for _ in range(3)]
    expected_output_shapes: List[List[Shape]] = [[] for _ in range(3)]
    expected_test_input_shapes: List[List[Shape]] = [[] for _ in range(1)]

    # example 1
    mask = np.array(
        [
            [8, 8, 8, 0],
            [8, 0, 0, 0],
            [8, 8, 8, 8],
            [8, 8, 0, 0],
            [8, 8, 8, 0],
            [8, 0, 0, 0],
            [8, 8, 8, 0],
            [8, 8, 8, 0],
            [8, 8, 0, 0],
        ],
        dtype=np.int16,
    )

    expected_input_shapes[0].append(
        Shape((2, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )
    expected_output_shapes[0].append(
        Shape((2, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[0, 0, 2], [2, 2, 2]], dtype=np.int16)

    expected_output_shapes[0].append(
        Shape((2, 2), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array(
        [[0, 2, 2], [0, 0, 2], [2, 2, 2], [0, 0, 2], [0, 0, 2], [0, 2, 2]],
        dtype=np.int16,
    )

    expected_output_shapes[0].append(
        Shape((5, 2), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array(
        [
            [8, 8, 8, 2],
            [8, 2, 2, 2],
            [8, 8, 8, 8],
            [8, 8, 2, 2],
            [8, 8, 8, 2],
            [8, 2, 2, 2],
            [8, 8, 8, 2],
            [8, 8, 8, 2],
            [8, 8, 2, 2],
        ],
        dtype=np.int16,
    )

    expected_output_shapes[0].append(
        Shape(
            (2, 1),
            mask,
            shape_type=ShapeType.MIXED_COLOUR,
        )
    )

    # example 2
    mask = np.array(
        [
            [8, 8, 8, 8, 8, 8],
            [8, 0, 8, 8, 0, 8],
            [8, 0, 8, 0, 0, 8],
            [0, 0, 8, 0, 8, 8],
        ],
        dtype=np.int16,
    )

    expected_input_shapes[1].append(
        Shape((1, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )
    expected_output_shapes[1].append(
        Shape((1, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[0, 2], [0, 2], [2, 2]], dtype=np.int16)

    expected_output_shapes[1].append(
        Shape((2, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[0, 2], [2, 2], [2, 0]], dtype=np.int16)

    expected_output_shapes[1].append(
        Shape((2, 4), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array(
        [
            [8, 8, 8, 8, 8, 8],
            [8, 2, 8, 8, 2, 8],
            [8, 2, 8, 2, 2, 8],
            [2, 2, 8, 2, 8, 8],
        ],
        dtype=np.int16,
    )

    expected_output_shapes[1].append(
        Shape((1, 1), mask, shape_type=ShapeType.MIXED_COLOUR)
    )

    # example 3
    mask = np.array(
        [
            [8, 8, 8, 8, 8],
            [0, 0, 8, 0, 8],
            [0, 8, 8, 8, 8],
            [0, 0, 8, 8, 8],
            [0, 0, 0, 8, 8],
            [0, 0, 8, 8, 8],
        ],
        dtype=np.int16,
    )

    expected_input_shapes[2].append(
        Shape((1, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )
    expected_output_shapes[2].append(
        Shape((1, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array(
        [[2, 2, 0], [2, 0, 0], [2, 2, 0], [2, 2, 2], [2, 2, 0]], dtype=np.int16
    )

    expected_output_shapes[2].append(
        Shape((2, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[2]], dtype=np.int16)
    expected_output_shapes[2].append(
        Shape((2, 4), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array(
        [
            [8, 8, 8, 8, 8],
            [2, 2, 8, 2, 8],
            [2, 8, 8, 8, 8],
            [2, 2, 8, 8, 8],
            [2, 2, 2, 8, 8],
            [2, 2, 8, 8, 8],
        ],
        dtype=np.int16,
    )

    expected_output_shapes[2].append(
        Shape((1, 1), mask, shape_type=ShapeType.MIXED_COLOUR)
    )

    # test input 1
    mask = np.array(
        [
            [8, 0, 0, 0, 0, 0, 0],
            [8, 0, 0, 0, 8, 8, 0],
            [8, 0, 8, 0, 0, 8, 0],
            [8, 8, 8, 0, 0, 8, 0],
            [8, 8, 8, 8, 0, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
        ],
        dtype=np.int16,
    )

    expected_test_input_shapes[0].append(
        Shape((2, 2), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    return InterpreterTestCase(
        interpreter=create_interpreter_for_task("6d75e8bb"),
        expected_input_shapes=expected_input_shapes,
        expected_output_shapes=expected_output_shapes,
        expected_test_input_shapes=expected_test_input_shapes,
    )


def create_test_case_6e19193c() -> InterpreterTestCase:
    expected_input_shapes: List[List[Shape]] = [[] for _ in range(2)]
    expected_output_shapes: List[List[Shape]] = [[] for _ in range(2)]
    expected_test_input_shapes: List[List[Shape]] = [[] for _ in range(1)]

    # example 1
    mask = np.array([[7, 0], [7, 7]], dtype=np.int16)
    rot_mask = mask.astype(bool)

    expected_input_shapes[0].append(
        Shape((2, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )
    expected_output_shapes[0].append(
        Shape((2, 1), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    expected_input_shapes[0].append(
        RotatableMaskShape((2, 1), mask, rot_mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[7, 7], [0, 7]], dtype=np.int16)

    expected_input_shapes[0].append(
        Shape((4, 6), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )
    expected_output_shapes[0].append(
        Shape((4, 6), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    expected_input_shapes[0].append(
        RotatableMaskShape((4, 6), mask, rot_mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[0, 7], [7, 0]], dtype=np.int16)

    expected_output_shapes[0].append(
        Shape((0, 3), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array(
        [[0, 0, 0, 7], [0, 0, 7, 0], [0, 7, 0, 0], [7, 0, 0, 0]], dtype=np.int16
    )

    expected_output_shapes[0].append(
        Shape((6, 2), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    # example 2
    mask = np.array([[9, 0], [9, 9]], dtype=np.int16)

    expected_input_shapes[1].append(
        Shape((6, 3), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )
    expected_output_shapes[1].append(
        Shape((6, 3), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    expected_input_shapes[1].append(
        RotatableMaskShape((6, 3), mask, rot_mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[9, 9], [0, 9]], dtype=np.int16)

    expected_input_shapes[1].append(
        Shape((1, 3), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )
    expected_output_shapes[1].append(
        Shape((1, 3), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    expected_input_shapes[1].append(
        RotatableMaskShape((1, 3), mask, rot_mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array(
        [
            [0, 0, 0, 0, 9],
            [0, 0, 0, 9, 0],
            [0, 0, 9, 0, 0],
            [0, 9, 0, 0, 0],
            [9, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    expected_output_shapes[1].append(
        Shape((1, 5), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[0, 0, 9], [0, 9, 0], [9, 0, 0]], dtype=np.int16)

    expected_output_shapes[1].append(
        Shape((3, 0), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    # test input 1
    mask = np.array([[8, 8], [0, 8]], dtype=np.int16)

    expected_test_input_shapes[0].append(
        Shape((6, 2), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    expected_test_input_shapes[0].append(
        RotatableMaskShape((6, 2), mask, rot_mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[8, 8], [8, 0]], dtype=np.int16)

    expected_test_input_shapes[0].append(
        Shape((4, 7), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    expected_test_input_shapes[0].append(
        RotatableMaskShape((4, 7), mask, rot_mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    mask = np.array([[0, 8], [8, 8]], dtype=np.int16)

    expected_test_input_shapes[0].append(
        Shape((2, 3), mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    expected_test_input_shapes[0].append(
        RotatableMaskShape((2, 3), mask, rot_mask, shape_type=ShapeType.SINGLE_COLOUR)
    )

    return InterpreterTestCase(
        interpreter=create_interpreter_for_task("6e19193c"),
        expected_input_shapes=expected_input_shapes,
        expected_output_shapes=expected_output_shapes,
        expected_test_input_shapes=expected_test_input_shapes,
    )


def create_interpreter_for_task(task_id: str):
    file_path = Path(__file__).parent / f"test_data/{task_id}.json"
    with open(str(file_path), "r") as file:
        data = json.load(file)

    inputs = [np.array(item["input"]) for item in data["train"]]
    outputs = [np.array(item["output"]) for item in data["train"]]
    test_inputs = [np.array(item["input"]) for item in data["test"]]
    return Interpreter(inputs, outputs, test_inputs)
