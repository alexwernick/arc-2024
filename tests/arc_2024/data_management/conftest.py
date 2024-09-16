import os
from pathlib import Path

import pytest

from arc_2024.data_management.data_manager import DataManager


@pytest.fixture
def data_manager():
    # Setup code
    input_dir = Path(__file__).parent / "test_data"
    temp_dir = Path(__file__).parent / "test_temp"
    data_manager = DataManager(input_dir, temp_dir)
    yield data_manager
    # Cleanup code
    for file in temp_dir.glob("*"):
        file.unlink()

    os.rmdir(temp_dir)
