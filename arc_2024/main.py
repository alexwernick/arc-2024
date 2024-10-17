import cProfile
import os
import pstats
from pathlib import Path

from dotenv import load_dotenv

from arc_2024.runner import run


def main():
    # load config variables from .env file
    load_dotenv()
    INPUT_DATA_PATH = os.getenv("INPUT_DATA_PATH")
    OUTPUT_DATA_PATH = os.getenv("OUTPUT_DATA_PATH")
    TEMP_DATA_PATH = os.getenv("TEMP_DATA_PATH")
    TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")

    input_data_path = Path(__file__).parent / INPUT_DATA_PATH
    output_data_path = Path(__file__).parent / OUTPUT_DATA_PATH
    temp_data_path = Path(__file__).parent / TEMP_DATA_PATH
    test_file_name = TEST_FILE_NAME

    run(
        input_data_path,
        output_data_path,
        temp_data_path,
        test_file_name,
        split_tasks=False,
        verify_solutions=True,
    )


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()
