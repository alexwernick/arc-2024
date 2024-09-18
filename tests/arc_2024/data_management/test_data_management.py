import json
import os
from pathlib import Path

from arc_2024.data_management.data_manager import DataManager


def test_split_tasks_to_individual_files(data_manager: DataManager):
    # Call the split_tasks_to_individual_files method
    file_name = "sample_challenges.json"
    data_manager.split_tasks_to_individual_files(file_name)

    # Check if the files were created
    assert os.path.exists(Path(data_manager._temp_dir) / "007bbfb7.json")
    assert os.path.exists(Path(data_manager._temp_dir) / "00d62c1b.json")


def test_create_solution_file(data_manager: DataManager):
    # Create individual solution files
    os.makedirs(data_manager._temp_dir, exist_ok=True)
    solution_data = {
        "007bbfb7": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
        "00d62c1b": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    }
    for task_id, task_data in solution_data.items():
        output_file_path = os.path.join(
            data_manager._temp_dir, f"solution_{task_id}.json"
        )
        with open(output_file_path, "w") as output_file:
            json.dump(task_data, output_file, indent=4)

    # Call the create_solution_file method
    file_name = "sample_solutions.json"
    data_manager.create_solution_file(file_name)

    # Check if the solution file was created
    assert os.path.exists(Path(data_manager._output_dir) / file_name)
    with open(Path(data_manager._output_dir) / file_name, "r") as file:
        data = json.load(file)
        assert data == solution_data


def test_save_individual_solution(data_manager: DataManager):
    # Prepare test data
    solution_data = {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
    task_id = "007bbfb7"

    # Call the save_individual_solution method
    data_manager.save_individual_solution(solution_data, task_id)

    # Check if the solution file was created
    output_file_path = os.path.join(data_manager._temp_dir, f"solution_{task_id}.json")
    assert os.path.exists(output_file_path)
    with open(output_file_path, "r") as file:
        data = json.load(file)
        assert data == solution_data


def test_get_unsolved_tasks(data_manager: DataManager):
    # Prepare test data
    file_name = "sample_challenges.json"
    data_manager.split_tasks_to_individual_files(file_name)
    solution_data = {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
    task_id = "007bbfb7"

    # Save an individual solution
    data_manager.save_individual_solution(solution_data, task_id)

    # Call the get_unsolved_tasks method
    unsolved_tasks = data_manager.get_unsolved_tasks()

    # Check if the unsolved tasks are as expected
    assert (
        task_id not in unsolved_tasks
    )  # Assuming get_unsolved_tasks returns a list of unsolved task IDs
    assert "00d62c1b" in unsolved_tasks  # This task should still be unsolved
