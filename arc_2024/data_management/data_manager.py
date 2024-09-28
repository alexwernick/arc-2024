import json
import os
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from numpy.typing import NDArray


class DataManager:
    def __init__(self, input_dir, output_dir, temp_dir):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._temp_dir = temp_dir
        self._solution_file_prefix = "solution_"

    def split_tasks_to_individual_files(self, file_name):
        """
        The format of the competition gives all data in one file.
        This function splits the data into individual files
        """
        # Load the JSON content
        json_file_path = (
            Path(self._input_dir) / file_name
        )  # Replace with the actual path to your JSON file
        with open(json_file_path, "r") as file:
            data = json.load(file)

        # Create the 'test' directory
        os.makedirs(self._temp_dir, exist_ok=True)

        # Split the JSON content into individual files
        for task_id, task_data in data.items():
            output_file_path = os.path.join(self._temp_dir, f"{task_id}.json")
            with open(output_file_path, "w") as output_file:
                json.dump(task_data, output_file)

    def create_solution_file(self, file_name):
        """
        Creates a single solution file from individual
        solution files in the 'temp' directory
        """
        # Create the 'output' directory
        os.makedirs(self._output_dir, exist_ok=True)

        # Load the individual solution files
        solution_data = {}
        for file_path in Path(self._temp_dir).glob(
            f"{self._solution_file_prefix}*.json"
        ):
            with open(file_path, "r") as file:
                data = json.load(file)
                task_id = file_path.stem.replace(self._solution_file_prefix, "")
                solution_data[task_id] = data

        # Write the solution file
        output_file_path = os.path.join(self._output_dir, file_name)
        with open(output_file_path, "w") as output_file:
            json.dump(solution_data, output_file)

    def save_individual_solution(self, solution_data, task_id):
        """
        Saves a single solution file in the 'temp' directory
        """
        os.makedirs(self._temp_dir, exist_ok=True)
        output_file_path = os.path.join(
            self._temp_dir, f"{self._solution_file_prefix}{task_id}.json"
        )
        with open(output_file_path, "w") as output_file:
            json.dump(solution_data, output_file)

    def get_unsolved_tasks(self):
        """
        Returns a list of unsolved tasks
        """
        tasks = []
        solutions = []
        for file_path in Path(self._temp_dir).glob("*.json"):
            if not file_path.name.startswith(self._solution_file_prefix):
                tasks.append(file_path.stem)
            else:
                solutions.append(file_path.stem.replace(self._solution_file_prefix, ""))

        return list(set(tasks) - set(solutions))  # Return unsolved tasks

    class TaskData(NamedTuple):
        inputs: List[NDArray[np.int16]]
        outputs: List[NDArray[np.int16]]
        test_inputs: List[NDArray[np.int16]]
        test_outputs: List[NDArray[np.int16]]

    def get_task_data(self, task_id: str) -> TaskData:
        """
        Returns the inputs, outputs, test_inputs and test_outputs for a given task
        """
        with open(Path(self._temp_dir) / f"{task_id}.json", "r") as file:
            data = json.load(file)

        inputs = [np.array(item["input"], dtype=np.int16) for item in data["train"]]
        outputs = [np.array(item["output"], dtype=np.int16) for item in data["train"]]
        test_inputs = [np.array(item["input"], dtype=np.int16) for item in data["test"]]
        test_outputs = [
            np.array(item["output"], dtype=np.int16) for item in data["test"]
        ]

        return self.TaskData(inputs, outputs, test_inputs, test_outputs)
