from pathlib import Path

import numpy as np
import pytest

from arc_2024.data_management.data_manager import DataManager
from arc_2024.solver import Solver


@pytest.mark.parametrize(
    "task_id", [("6e02f1e3"), ("6d75e8bb"), ("6e82a1ae"), ("6e19193c")]
)
def test_solver(task_id):
    data_manager = DataManager(
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
    )

    inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)

    solver = Solver(inputs, outputs, test_inputs)

    results = solver.solve(beam_width=2)

    for result, test_output in zip(results, test_outputs):
        assert np.array_equal(result, test_output)
