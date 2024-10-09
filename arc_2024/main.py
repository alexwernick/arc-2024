import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from arc_2024.data_management.data_manager import DataManager
from arc_2024.solver import Solver


def main():
    # load config variables from .env file
    load_dotenv()
    INPUT_DATA_PATH = os.getenv("INPUT_DATA_PATH")
    OUTPUT_DATA_PATH = os.getenv("OUTPUT_DATA_PATH")
    TEMP_DATA_PATH = os.getenv("TEMP_DATA_PATH")
    # TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")

    data_manager = DataManager(
        Path(__file__).parent / INPUT_DATA_PATH,
        Path(__file__).parent / OUTPUT_DATA_PATH,
        Path(__file__).parent / TEMP_DATA_PATH,
    )
    # Split the tasks into individual files
    # data_manager.split_tasks_to_individual_files(TEST_FILE_NAME)

    task_id = "6e82a1ae"  # 6e02f1e3 6d75e8bb 6e82a1ae
    inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)

    solver = Solver(inputs, outputs, test_inputs)

    results = solver.solve()

    for result, test_output in zip(results, test_outputs):
        correct_solution = np.array_equal(result, test_output)
        if correct_solution:
            print(f"Task {task_id} was solved correctly")
        else:
            print(f"Task {task_id} was solved incorrectly")

    # for unsolved_task in data_manager.get_unsolved_tasks():
    #     with open(
    #         Path(__file__).parent / TEMP_DATA_PATH / f"{unsolved_task}.json", "r"
    #     ) as file:
    #         data = json.load(file)
    #     # Count the number of elements
    #     element_count = len(data["test"])
    #     solution = []
    #     for _ in range(element_count):
    #         # Here you would implement your solution logic
    #         solution.append(
    #             {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
    #         )

    #     data_manager.save_individual_solution(solution, unsolved_task)

    # data_manager.create_solution_file("submission.json")


if __name__ == "__main__":
    main()
