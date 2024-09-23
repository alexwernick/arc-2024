import json
import os
from pathlib import Path

from dotenv import load_dotenv

from arc_2024.data_management.data_manager import DataManager
from arc_2024.popper.loop import learn_solution
from arc_2024.popper.util import Settings


def main():
    # load config variables from .env file
    load_dotenv()
    INPUT_DATA_PATH = os.getenv("INPUT_DATA_PATH")
    OUTPUT_DATA_PATH = os.getenv("OUTPUT_DATA_PATH")
    TEMP_DATA_PATH = os.getenv("TEMP_DATA_PATH")
    TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")

    data_manager = DataManager(
        Path(__file__).parent / INPUT_DATA_PATH,
        Path(__file__).parent / OUTPUT_DATA_PATH,
        Path(__file__).parent / TEMP_DATA_PATH,
    )
    # Split the tasks into individual files
    data_manager.split_tasks_to_individual_files(TEST_FILE_NAME)

    for unsolved_task in data_manager.get_unsolved_tasks():
        with open(
            Path(__file__).parent / TEMP_DATA_PATH / f"{unsolved_task}.json", "r"
        ) as file:
            data = json.load(file)
        # Count the number of elements
        element_count = len(data["test"])
        solution = []
        for _ in range(element_count):
            # Here you would implement your solution logic
            solution.append(
                {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
            )

        data_manager.save_individual_solution(solution, unsolved_task)

    data_manager.create_solution_file("submission.json")

    # try popper
    ex_file = str(Path(__file__).parent / "data/arc_parsed_popper/253bf280/exs.pl")
    bk_file = str(Path(__file__).parent / "data/arc_parsed_popper/253bf280/bk.pl")
    bias_file = str(Path(__file__).parent / "data/arc_parsed_popper/253bf280/bias.pl")
    settings = Settings(ex_file=ex_file, bk_file=bk_file, bias_file=bias_file)

    prog, score, stats = learn_solution(settings)
    if prog != None:
        settings.print_prog_score(prog, score)
    else:
        print("NO SOLUTION")
    if settings.show_stats:
        stats.show()


if __name__ == "__main__":
    main()
