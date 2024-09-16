import os
from pathlib import Path

import pytest

from arc_2024.data_management.data_manager import DataManager


@pytest.fixture
def data_manager():
    # Setup code
    input_dir = Path(__file__).parent / "test_data"
    output_dir = Path(__file__).parent / "test_output"
    temp_dir = Path(__file__).parent / "test_temp"
    data_manager = DataManager(input_dir, output_dir, temp_dir)
    yield data_manager
    # Cleanup code
    delete_dir(temp_dir)
    delete_dir(output_dir)


def delete_dir(dir: Path):
    if dir.exists() and dir.is_dir():
        for file in dir.glob("*"):
            file.unlink()
        os.rmdir(dir)
