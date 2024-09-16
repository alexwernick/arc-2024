import json
import os
from pathlib import Path


class DataManager:
    def __init__(self, input_dir, output_dir, temp_dir):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._temp_dir = temp_dir

    def split_tasks_to_individual_files(self, file_name):
        """
        The format of the competition gives all date in one file.
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
                json.dump(task_data, output_file, indent=4)

    def create_solution_file(self, file_name):
        """
        Creates a single solution file from individual
        solution files in the 'temp' directory
        """
        # Create the 'output' directory
        os.makedirs(self._output_dir, exist_ok=True)

        # Load the individual solution files
        solution_data = {}
        solution_file_prefix = "solution_"
        for file_path in Path(self._temp_dir).glob(f"{solution_file_prefix}*.json"):
            with open(file_path, "r") as file:
                data = json.load(file)
                task_id = file_path.stem.replace(solution_file_prefix, "")
                solution_data[task_id] = data

        # Write the solution file
        output_file_path = os.path.join(self._output_dir, file_name)
        with open(output_file_path, "w") as output_file:
            json.dump(solution_data, output_file, indent=4)
