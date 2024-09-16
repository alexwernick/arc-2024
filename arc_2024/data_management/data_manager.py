import json
import os
from pathlib import Path


class DataManager:
    def __init__(self, input_dir, temp_dir):
        self._input_dir = input_dir
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
