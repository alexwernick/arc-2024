import os
from pathlib import Path

import pytest

from arc_2024.data_management.data_manager import DataManager
from arc_2024.grid_size_solver import GridSizeSolver
from arc_2024.representations.interpreter import Interpreter


def get_file_names_without_suffix(directory):
    file_names = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            name, _ = os.path.splitext(file)
            file_names.append(name)
    return file_names


@pytest.mark.parametrize(
    "task_id", get_file_names_without_suffix(Path(__file__).parent / "test_data")
)
def test_grid_size_solver(task_id):
    data_manager = DataManager(
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
    )

    inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)

    interpreter = Interpreter(inputs, outputs, test_inputs)
    interpretations = interpreter.interpret_shapes()
    exceptions = []

    for interpretation in interpretations:
        try:
            (
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
                _,
            ) = interpretation

            solver = GridSizeSolver(
                inputs,
                outputs,
                test_inputs,
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
            )

            results = solver.solve(beam_width=2)

            for result, test_output in zip(results, test_outputs):
                assert (
                    result.shape == test_output.shape
                ), f"Assertion failed for task_id: {task_id}"

            return

        except Exception as e:
            exceptions.append(e)
            print(f"Error: {e}")

    if len(exceptions) > 0:
        raise exceptions[0]
    assert False, f"No solution found for task_id: {task_id}"
