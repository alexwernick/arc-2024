import json
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import pytest

from arc_2024.representations.colour import Color
from arc_2024.representations.interpreter import Interpreter
from arc_2024.representations.shape import Shape


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
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int16,
    )

    expected_input_shapes[0].append(Shape(Color(8), (2, 1), mask))
    expected_output_shapes[0].append(Shape(Color(8), (2, 1), mask))

    mask = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int16)

    expected_output_shapes[0].append(Shape(Color(2), (2, 2), mask))

    mask = np.array(
        [[0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1]],
        dtype=np.int16,
    )

    expected_output_shapes[0].append(Shape(Color(2), (5, 2), mask))

    # example 2
    mask = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 1],
        ],
        dtype=np.int16,
    )

    expected_input_shapes[1].append(Shape(Color(8), (1, 1), mask))
    expected_output_shapes[1].append(Shape(Color(8), (1, 1), mask))

    mask = np.array([[0, 1], [0, 1], [1, 1]], dtype=np.int16)

    expected_output_shapes[1].append(Shape(Color(2), (2, 1), mask))

    mask = np.array([[0, 1], [1, 1], [1, 0]], dtype=np.int16)

    expected_output_shapes[1].append(Shape(Color(2), (2, 4), mask))

    # example 3
    mask = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.int16,
    )

    expected_input_shapes[2].append(Shape(Color(8), (1, 1), mask))
    expected_output_shapes[2].append(Shape(Color(8), (1, 1), mask))

    mask = np.array(
        [[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=np.int16
    )

    expected_output_shapes[2].append(Shape(Color(2), (2, 1), mask))

    mask = np.array([[1]], dtype=np.int16)
    expected_output_shapes[2].append(Shape(Color(2), (2, 4), mask))

    # test input 1
    mask = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.int16,
    )

    expected_test_input_shapes[0].append(Shape(Color(8), (2, 2), mask))

    return InterpreterTestCase(
        interpreter=create_interpreter_for_task("6d75e8bb"),
        expected_input_shapes=expected_input_shapes,
        expected_output_shapes=expected_output_shapes,
        expected_test_input_shapes=expected_test_input_shapes,
    )


def create_test_case_6e19193c() -> InterpreterTestCase:
    expected_input_shapes: List[List[Shape]] = [[] for _ in range(3)]
    expected_output_shapes: List[List[Shape]] = [[] for _ in range(3)]
    expected_test_input_shapes: List[List[Shape]] = [[] for _ in range(1)]

    # example 1
    mask = np.array([[1, 0], [1, 1]], dtype=np.int16)

    expected_input_shapes[0].append(Shape(Color(7), (2, 1), mask))
    expected_output_shapes[0].append(Shape(Color(7), (2, 1), mask))

    mask = np.array([[1, 1], [0, 1]], dtype=np.int16)

    expected_output_shapes[0].append(Shape(Color(7), (4, 6), mask))
    expected_output_shapes[0].append(Shape(Color(7), (4, 6), mask))

    mask = np.array([[0, 1], [1, 0]], dtype=np.int16)

    expected_output_shapes[0].append(Shape(Color(7), (0, 3), mask))

    mask = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.int16
    )

    expected_output_shapes[0].append(Shape(Color(7), (6, 2), mask))

    # example 2
    mask = np.array([[1, 0], [1, 1]], dtype=np.int16)

    expected_input_shapes[1].append(Shape(Color(9), (6, 3), mask))
    expected_output_shapes[1].append(Shape(Color(9), (6, 3), mask))

    mask = np.array([[1, 1], [0, 1]], dtype=np.int16)

    expected_input_shapes[1].append(Shape(Color(9), (1, 3), mask))
    expected_output_shapes[1].append(Shape(Color(9), (1, 3), mask))

    mask = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    expected_output_shapes[1].append(Shape(Color(9), (1, 5), mask))

    mask = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int16)

    expected_output_shapes[1].append(Shape(Color(9), (3, 0), mask))

    # test input 1
    mask = np.array([[1, 1], [0, 1]], dtype=np.int16)

    expected_test_input_shapes[0].append(Shape(Color(8), (6, 2), mask))

    mask = np.array([[1, 1], [1, 0]], dtype=np.int16)

    expected_test_input_shapes[0].append(Shape(Color(8), (4, 7), mask))

    mask = np.array([[0, 1], [1, 1]], dtype=np.int16)

    expected_test_input_shapes[0].append(Shape(Color(8), (2, 3), mask))

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
