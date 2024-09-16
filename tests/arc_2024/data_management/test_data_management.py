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
