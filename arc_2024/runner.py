import numpy as np
from numpy.typing import NDArray

from arc_2024.data_management.data_manager import DataManager
from arc_2024.grid_size_solver import GridSizeSolver
from arc_2024.representations.interpreter import Interpreter
from arc_2024.solver import Solver


def verify_solution(
    results: NDArray[np.int16], test_outputs: list[NDArray[np.int16]], task_id: str
):
    for result, test_output in zip(results, test_outputs):
        correct_solution = np.array_equal(result, test_output)
        if correct_solution:
            print(f"Task {task_id} was solved correctly")
        else:
            print(f"Task {task_id} was solved incorrectly")


def run(
    input_data_path: str,
    output_data_path: str,
    temp_data_path: str,
    test_file_name: str,
    split_tasks: bool,
    verify_solutions: bool,
    submit_empty_solutions: bool = False,
):
    data_manager = DataManager(input_data_path, output_data_path, temp_data_path)

    # Split the tasks into individual files (on kaggle server)
    if split_tasks:
        data_manager.split_tasks_to_individual_files(test_file_name)

    sumbission_file_name = "submission.json"
    data_manager.create_valid_empty_solution_file(sumbission_file_name)

    # This is just to test format on kaggle server
    if submit_empty_solutions:
        return

    task_ids: list[str] = data_manager.get_all_task_ids()
    grid_size_solutions: dict[str, list[NDArray[np.int16]]] = {}

    for task_id in task_ids:
        inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)
        interpreter = Interpreter(inputs, outputs, test_inputs)
        interpretations = interpreter.interpret_shapes()

        for interpretation in interpretations:
            if task_id not in grid_size_solutions:
                (
                    inputs_shapes,
                    outputs_shapes,
                    test_inputs_shapes,
                    interpret_type,
                ) = interpretation
                grid_size_solver = GridSizeSolver(
                    inputs,
                    outputs,
                    test_inputs,
                    inputs_shapes,
                    outputs_shapes,
                    test_inputs_shapes,
                )
                try:
                    grid_size_solutions[task_id] = grid_size_solver.solve(beam_width=2)
                    print(
                        f"Task {task_id} grid size was solved using interpret_type {interpret_type.name}"  # noqa: E501
                    )
                except Exception as e:
                    print(
                        f"Task {task_id} grid size was not solved due to exception {e} using interpret_type {interpret_type.name}"  # noqa: E501
                    )

        if task_id not in grid_size_solutions:
            continue

        for interpretation in interpretations:
            (
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
                interpret_type,
            ) = interpretation
            solver = Solver(
                inputs,
                outputs,
                test_inputs,
                grid_size_solutions[task_id],
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
            )
            try:
                results = solver.solve(beam_width=2)
                if verify_solutions:
                    verify_solution(results, test_outputs, task_id)
                data_manager.update_solution_file(
                    sumbission_file_name, task_id, results
                )
            except Exception as e:
                print(
                    f"Task {task_id} was not solved due to exception {e} using interpret_type {interpret_type.name}"  # noqa: E501
                )
