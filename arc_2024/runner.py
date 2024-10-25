import time

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
    max_run_time_for_solutions: int = 36000,  # 10 hours
    skip_training_data: bool = False,
):
    start_time = time.time()
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
    grid_size_solutions: list[
        tuple[str, list[NDArray[np.int16]], Interpreter.InterpretType]
    ] = []

    # If skip_training_data is True we check if we are in the
    # training env and we return
    # We know task 36d67576 is in the training data
    if skip_training_data and "36d67576" in task_ids:
        return

    # we give 10% of time to solve grid size and 90% to solve the task
    run_time_per_task = max_run_time_for_solutions / len(task_ids)
    grid_size_timeout_seconds = int(run_time_per_task * 0.1)
    task_timeout_seconds = int(run_time_per_task * 0.9)

    for task_id in task_ids:
        inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)
        interpreter = Interpreter(inputs, outputs, test_inputs)
        interpretations = interpreter.interpret_shapes()

        for interpretation in interpretations:
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
                grid_size_solutions.append(
                    (
                        task_id,
                        grid_size_solver.solve(
                            beam_width=2,
                            max_clause_length=4,
                            timeout_seconds=grid_size_timeout_seconds,
                        ),
                        interpret_type,
                    )
                )
                print(
                    f"Task {task_id} grid size was solved using interpret_type {interpret_type.name}"  # noqa: E501
                )
                break
            except Exception as e:
                print(
                    f"Task {task_id} grid size was not solved due to exception {e} using interpret_type {interpret_type.name}"  # noqa: E501
                )

    def get_max_grid_size(grids: list[NDArray[np.int16]]):
        max_grid_size = 0
        for grid in grids:
            max_grid_size = max(max_grid_size, grid.shape[0] * grid.shape[1])
        return max_grid_size

    # Do smaller puzzles first
    grid_size_solutions = sorted(
        grid_size_solutions, key=lambda x: get_max_grid_size(x[1])
    )

    max_time_hit: bool = False
    for task_id, empty_test_outputs, interpret_type in grid_size_solutions:
        if max_time_hit:
            break

        inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)
        interpreter = Interpreter(inputs, outputs, test_inputs)
        interpretations = interpreter.interpret_shapes()
        interpretations = sorted(
            interpretations, key=lambda x: x.interpret_type != interpret_type
        )

        for interpretation in interpretations:
            (
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
                interpret_type,
            ) = interpretation

            elapsed_time = time.time() - start_time
            if elapsed_time > max_run_time_for_solutions:
                print(f"Max time hit. Ending runner after {elapsed_time}s")
                max_time_hit = True
                break

            solver = Solver(
                inputs,
                outputs,
                test_inputs,
                empty_test_outputs,
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
            )
            try:
                results = solver.solve(
                    beam_width=2,
                    max_clause_length=6,
                    timeout_seconds=task_timeout_seconds,
                )

                print(
                    f"Task {task_id} was solved using interpret_type {interpret_type.name}"  # noqa: E501
                )

                if verify_solutions:
                    verify_solution(results, test_outputs, task_id)
                data_manager.update_solution_file(
                    sumbission_file_name, task_id, results
                )
                break
            except Exception as e:
                print(
                    f"Task {task_id} was not solved due to exception {e} using interpret_type {interpret_type.name}"  # noqa: E501
                )
