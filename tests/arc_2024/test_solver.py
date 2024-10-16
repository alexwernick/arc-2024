from pathlib import Path

import numpy as np
import pytest

from arc_2024.data_management.data_manager import DataManager
from arc_2024.representations.interpreter import Interpreter
from arc_2024.solver import Solver


@pytest.mark.parametrize(
    "task_id,empty_test_outputs",
    [
        ("6e02f1e3", [np.zeros((3, 3), dtype=np.int16)]),
        ("6d75e8bb", [np.zeros((9, 11), dtype=np.int16)]),
        ("6e82a1ae", [np.zeros((10, 10), dtype=np.int16)]),
        ("6e19193c", [np.zeros((10, 10), dtype=np.int16)]),
        ("6f8cd79b", [np.zeros((7, 6), dtype=np.int16)]),
        ("00d62c1b", [np.zeros((20, 20), dtype=np.int16)]),
    ],
)
def test_solver(task_id, empty_test_outputs):
    data_manager = DataManager(
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
    )

    inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)

    interpreter = Interpreter(inputs, outputs, test_inputs)
    interpretations = interpreter.interpret_shapes()

    for interpretation in interpretations:
        try:
            (
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
                _,
            ) = interpretation

            solver = Solver(
                inputs,
                outputs,
                test_inputs,
                empty_test_outputs,
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
            )

            results = solver.solve(beam_width=2)

            for result, test_output in zip(results, test_outputs):
                assert np.array_equal(
                    result, test_output
                ), f"Assertion failed for task_id: {task_id}"

            return
        except Exception as e:
            print(f"Error: {e}")

    assert False, f"No solution found for task_id: {task_id}"
