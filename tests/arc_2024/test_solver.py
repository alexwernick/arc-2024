from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from arc_2024.data_management.data_manager import DataManager
from arc_2024.inductive_logic_programming.first_order_logic import Predicate
from arc_2024.representations.colour import Colour
from arc_2024.representations.interpreter import Interpreter
from arc_2024.representations.shape import Shape
from arc_2024.representations.shape_type import ShapeType
from arc_2024.solver import Solver


@pytest.mark.parametrize(
    "task_id,empty_test_outputs",
    [
        ("6e02f1e3", [np.zeros((3, 3), dtype=np.int16)]),
        ("6d75e8bb", [np.zeros((9, 11), dtype=np.int16)]),
        ("6e82a1ae", [np.zeros((10, 10), dtype=np.int16)]),
        ("6e19193c", [np.zeros((10, 10), dtype=np.int16)]),
        ("6f8cd79b", [np.zeros((7, 6), dtype=np.int16)]),
        ("00d62c1b", [np.zeros((20, 20), dtype=np.int16)]),
        # ("06df4c85", [np.zeros((20, 26), dtype=np.int16)]),
        # ("0962bcdd", [np.zeros((12, 12), dtype=np.int16)]),
        # ("0ca9ddb6", [np.zeros((9, 9), dtype=np.int16)]),
        # ("0d3d703e", [np.zeros((3, 3), dtype=np.int16)]),
        # ("178fcbfb", [np.zeros((12, 11), dtype=np.int16)]),
        ("1caeab9d", [np.zeros((10, 10), dtype=np.int16)]),
        ("1a07d186", [np.zeros((19, 26), dtype=np.int16)]),
        ("0520fde7", [np.zeros((3, 3), dtype=np.int16)]),
        ("1b2d62fb", [np.zeros((5, 3), dtype=np.int16)]),
        ("08ed6ac7", [np.zeros((9, 9), dtype=np.int16)]),
        ("7c008303", [np.zeros((6, 6), dtype=np.int16)]),
        ("007bbfb7", [np.zeros((9, 9), dtype=np.int16)]),
    ],
)
def test_solver(task_id, empty_test_outputs):
    data_manager = DataManager(
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_data",
    )

    inputs, outputs, test_inputs, test_outputs = data_manager.get_task_data(task_id)

    interpreter = Interpreter(inputs, outputs, test_inputs)
    interpretations = interpreter.interpret_shapes()
    exceptions = []

    for interpretation in interpretations:
        try:
            (
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
                _,
            ) = interpretation

            solver = Solver(
                inputs,
                outputs,
                test_inputs,
                empty_test_outputs,
                inputs_shapes,
                outputs_shapes,
                test_inputs_shapes,
            )

            results = solver.solve(beam_width=2, max_clause_length=8)

            for result, test_output in zip(results, test_outputs):
                assert np.array_equal(
                    result, test_output
                ), f"Assertion failed for task_id: {task_id}"

            return
        except Exception as e:
            exceptions.append(e)
            print(f"Error: {e}")

    if len(exceptions) > 0:
        raise exceptions[0]
    assert False, f"No solution found for task_id: {task_id}"


def test_append_background_knowledge_for_mask_inline_with_shape_ex1():
    shape_1 = Shape(
        (3, 6), np.array([[4], [4]], dtype=np.int16), ShapeType.SINGLE_COLOUR
    )
    shape_2 = Shape(
        (2, 1), np.array([[1], [1]], dtype=np.int16), ShapeType.SINGLE_COLOUR
    )
    shape_name_1 = "shape_1"
    shape_name_2 = "shape_2"
    ex_num = 1
    mask_overlapping_top_inline_top_pred = Predicate(
        "mask-overlapping-top-inline-top-shape",
        0,
        [],
    )
    mask_overlapping_bot_inline_bot_pred = Predicate(
        "mask-overlapping-bot-inline-bot-shape",
        0,
        [],
    )
    mask_overlapping_left_inline_left_pred = Predicate(
        "mask-overlapping-left-inline-left-shape",
        0,
        [],
    )
    mask_overlapping_right_inline_right_pred = Predicate(
        "mask-overlapping-right-inline-right-shape",
        0,
        [],
    )
    mask_overlapping_bot_top_touching_pred = Predicate(
        "mask-overlapping-bot-top-touching-shape",
        0,
        [],
    )
    mask_overlapping_top_bot_touching_pred = Predicate(
        "mask-overlapping-top-bot-touching-shape",
        0,
        [],
    )
    mask_overlapping_right_left_touching_pred = Predicate(
        "mask-overlapping-right-left-touching-shape",
        0,
        [],
    )
    mask_overlapping_left_right_touching_pred = Predicate(
        "mask-overlapping-left-right-touching-shape",
        0,
        [],
    )

    background_knowledge = defaultdict(set)

    for i in range(5):
        for j in range(10):
            Solver._append_background_knowledge_for_mask_inline_with_shape(
                background_knowledge,
                i,
                j,
                shape_1,
                shape_name_1,
                shape_2,
                shape_name_2,
                ex_num,
                mask_overlapping_top_inline_top_pred,
                mask_overlapping_bot_inline_bot_pred,
                mask_overlapping_left_inline_left_pred,
                mask_overlapping_right_inline_right_pred,
                mask_overlapping_bot_top_touching_pred,
                mask_overlapping_top_bot_touching_pred,
                mask_overlapping_right_left_touching_pred,
                mask_overlapping_left_right_touching_pred,
            )

    mask_overlapping_top_inline_top_set = {
        (ex_num, 2, 6, shape_name_1, shape_name_2),
        (ex_num, 3, 6, shape_name_1, shape_name_2),
    }
    mask_overlapping_bot_inline_bot_set = {
        (ex_num, 2, 6, shape_name_1, shape_name_2),
        (ex_num, 3, 6, shape_name_1, shape_name_2),
    }
    mask_overlapping_left_inline_left_set = {
        (ex_num, 3, 1, shape_name_1, shape_name_2),
        (ex_num, 4, 1, shape_name_1, shape_name_2),
    }
    mask_overlapping_right_inline_right_set = {
        (ex_num, 3, 1, shape_name_1, shape_name_2),
        (ex_num, 4, 1, shape_name_1, shape_name_2),
    }
    mask_overlapping_bot_top_touching_set = {
        (ex_num, 0, 6, shape_name_1, shape_name_2),
        (ex_num, 1, 6, shape_name_1, shape_name_2),
    }
    mask_overlapping_top_bot_touching_set = {(ex_num, 4, 6, shape_name_1, shape_name_2)}
    mask_overlapping_right_left_touching_set = {
        (ex_num, 3, 0, shape_name_1, shape_name_2),
        (ex_num, 4, 0, shape_name_1, shape_name_2),
    }
    mask_overlapping_left_right_touching_set = {
        (ex_num, 3, 2, shape_name_1, shape_name_2),
        (ex_num, 4, 2, shape_name_1, shape_name_2),
    }

    assert (
        background_knowledge[mask_overlapping_top_inline_top_pred.name]
        == mask_overlapping_top_inline_top_set
    )
    assert (
        background_knowledge[mask_overlapping_bot_inline_bot_pred.name]
        == mask_overlapping_bot_inline_bot_set
    )
    assert (
        background_knowledge[mask_overlapping_left_inline_left_pred.name]
        == mask_overlapping_left_inline_left_set
    )
    assert (
        background_knowledge[mask_overlapping_right_inline_right_pred.name]
        == mask_overlapping_right_inline_right_set
    )
    assert (
        background_knowledge[mask_overlapping_bot_top_touching_pred.name]
        == mask_overlapping_bot_top_touching_set
    )
    assert (
        background_knowledge[mask_overlapping_top_bot_touching_pred.name]
        == mask_overlapping_top_bot_touching_set
    )
    assert (
        background_knowledge[mask_overlapping_right_left_touching_pred.name]
        == mask_overlapping_right_left_touching_set
    )
    assert (
        background_knowledge[mask_overlapping_left_right_touching_pred.name]
        == mask_overlapping_left_right_touching_set
    )


def test_append_background_knowledge_for_mask_inline_with_shape_ex2():
    shape_1 = Shape(
        (2, 1), np.array([[2, 2, 2]], dtype=np.int16), ShapeType.SINGLE_COLOUR
    )
    shape_2 = Shape(
        (5, 4),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int16),
        ShapeType.SINGLE_COLOUR,
    )
    shape_name_1 = "shape_1"
    shape_name_2 = "shape_2"
    ex_num = 1
    mask_overlapping_top_inline_top_pred = Predicate(
        "mask-overlapping-top-inline-top-shape",
        0,
        [],
    )
    mask_overlapping_bot_inline_bot_pred = Predicate(
        "mask-overlapping-bot-inline-bot-shape",
        0,
        [],
    )
    mask_overlapping_left_inline_left_pred = Predicate(
        "mask-overlapping-left-inline-left-shape",
        0,
        [],
    )
    mask_overlapping_right_inline_right_pred = Predicate(
        "mask-overlapping-right-inline-right-shape",
        0,
        [],
    )
    mask_overlapping_bot_top_touching_pred = Predicate(
        "mask-overlapping-bot-top-touching-shape",
        0,
        [],
    )
    mask_overlapping_top_bot_touching_pred = Predicate(
        "mask-overlapping-top-bot-touching-shape",
        0,
        [],
    )
    mask_overlapping_right_left_touching_pred = Predicate(
        "mask-overlapping-right-left-touching-shape",
        0,
        [],
    )
    mask_overlapping_left_right_touching_pred = Predicate(
        "mask-overlapping-left-right-touching-shape",
        0,
        [],
    )

    background_knowledge = defaultdict(set)

    for i in range(10):
        for j in range(10):
            Solver._append_background_knowledge_for_mask_inline_with_shape(
                background_knowledge,
                i,
                j,
                shape_1,
                shape_name_1,
                shape_2,
                shape_name_2,
                ex_num,
                mask_overlapping_top_inline_top_pred,
                mask_overlapping_bot_inline_bot_pred,
                mask_overlapping_left_inline_left_pred,
                mask_overlapping_right_inline_right_pred,
                mask_overlapping_bot_top_touching_pred,
                mask_overlapping_top_bot_touching_pred,
                mask_overlapping_right_left_touching_pred,
                mask_overlapping_left_right_touching_pred,
            )

    mask_overlapping_top_inline_top_set = {
        (ex_num, 5, 1, shape_name_1, shape_name_2),
        (ex_num, 5, 2, shape_name_1, shape_name_2),
        (ex_num, 5, 3, shape_name_1, shape_name_2),
    }

    mask_overlapping_bot_inline_bot_set = {
        (ex_num, 6, 1, shape_name_1, shape_name_2),
        (ex_num, 6, 2, shape_name_1, shape_name_2),
        (ex_num, 6, 3, shape_name_1, shape_name_2),
    }

    mask_overlapping_left_inline_left_set = {
        (ex_num, 2, 4, shape_name_1, shape_name_2),
        (ex_num, 2, 5, shape_name_1, shape_name_2),
        (ex_num, 2, 6, shape_name_1, shape_name_2),
    }

    mask_overlapping_right_inline_right_set = {
        (ex_num, 2, 4, shape_name_1, shape_name_2),
        (ex_num, 2, 5, shape_name_1, shape_name_2),
        (ex_num, 2, 6, shape_name_1, shape_name_2),
    }

    mask_overlapping_bot_top_touching_set = {
        (ex_num, 4, 1, shape_name_1, shape_name_2),
        (ex_num, 4, 2, shape_name_1, shape_name_2),
        (ex_num, 4, 3, shape_name_1, shape_name_2),
    }

    mask_overlapping_top_bot_touching_set = {
        (ex_num, 7, 1, shape_name_1, shape_name_2),
        (ex_num, 7, 2, shape_name_1, shape_name_2),
        (ex_num, 7, 3, shape_name_1, shape_name_2),
    }

    mask_overlapping_right_left_touching_set = {
        (ex_num, 2, 1, shape_name_1, shape_name_2),
        (ex_num, 2, 2, shape_name_1, shape_name_2),
        (ex_num, 2, 3, shape_name_1, shape_name_2),
    }

    mask_overlapping_left_right_touching_set = {
        (ex_num, 2, 7, shape_name_1, shape_name_2),
        (ex_num, 2, 8, shape_name_1, shape_name_2),
        (ex_num, 2, 9, shape_name_1, shape_name_2),
    }

    assert (
        background_knowledge[mask_overlapping_top_inline_top_pred.name]
        == mask_overlapping_top_inline_top_set
    )
    assert (
        background_knowledge[mask_overlapping_bot_inline_bot_pred.name]
        == mask_overlapping_bot_inline_bot_set
    )
    assert (
        background_knowledge[mask_overlapping_left_inline_left_pred.name]
        == mask_overlapping_left_inline_left_set
    )
    assert (
        background_knowledge[mask_overlapping_right_inline_right_pred.name]
        == mask_overlapping_right_inline_right_set
    )
    assert (
        background_knowledge[mask_overlapping_bot_top_touching_pred.name]
        == mask_overlapping_bot_top_touching_set
    )
    assert (
        background_knowledge[mask_overlapping_top_bot_touching_pred.name]
        == mask_overlapping_top_bot_touching_set
    )
    assert (
        background_knowledge[mask_overlapping_right_left_touching_pred.name]
        == mask_overlapping_right_left_touching_set
    )
    assert (
        background_knowledge[mask_overlapping_left_right_touching_pred.name]
        == mask_overlapping_left_right_touching_set
    )


@pytest.mark.parametrize(
    "shape_1,expected_indicies",
    [
        (
            Shape(
                (3, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            {(3, 2)},
        ),
        (
            Shape(
                (4, 6),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            {(4, 4)},
        ),
        (
            Shape(
                (8, 6),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            set(),
        ),
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            set(),
        ),
        (
            Shape(
                (0, 3),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            {(1, 3)},
        ),
        (
            Shape(
                (8, 3),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            {(8, 3)},
        ),
    ],
)
def test_append_background_knowledge_for_mask_gravity_to_shape(
    shape_1, expected_indicies
):
    solver = Solver([], [], [], [], [], [], [])
    shape_2 = Shape(
        (2, 3),
        np.array([[4], [4], [4], [4], [4], [4]], dtype=np.int16),
        ShapeType.SINGLE_COLOUR,
    )

    shape_name_1 = "shape_1"
    shape_name_2 = "shape_2"
    ex_num = 1
    mask_overlapping_gravity_to_shape_pred = Predicate(
        "mask-overlapping-to-shape",
        0,
        [],
    )

    background_knowledge = defaultdict(set)

    for i in range(10):
        for j in range(10):
            solver._append_background_knowledge_for_mask_gravity_to_shape(
                background_knowledge,
                i,
                j,
                shape_1,
                shape_name_1,
                shape_2,
                shape_name_2,
                ex_num,
                mask_overlapping_gravity_to_shape_pred,
            )

    mask_overlapping_gravity_to_shape_set = {
        (ex_num, ind[0], ind[1], shape_name_1, shape_name_2)
        for ind in expected_indicies
    }

    assert (
        background_knowledge[mask_overlapping_gravity_to_shape_pred.name]
        == mask_overlapping_gravity_to_shape_set
    )


def test_append_background_knowledge_for_mask_gravity_to_shape_ex_2():
    solver = Solver([], [], [], [], [], [], [])

    shape_1 = Shape(
        (0, 0),
        np.array([[4, 4, 0], [4, 4, 4], [4, 4, 0]], dtype=np.int16),
        ShapeType.SINGLE_COLOUR,
    )

    shape_2 = Shape(
        (0, 4),
        np.array([[4, 4, 4], [0, 4, 4], [4, 4, 4]], dtype=np.int16),
        ShapeType.SINGLE_COLOUR,
    )

    shape_name_1 = "shape_1"
    shape_name_2 = "shape_2"
    ex_num = 1
    mask_overlapping_gravity_to_shape_pred = Predicate(
        "mask-overlapping-to-shape",
        0,
        [],
    )

    background_knowledge = defaultdict(set)

    for i in range(10):
        for j in range(10):
            solver._append_background_knowledge_for_mask_gravity_to_shape(
                background_knowledge,
                i,
                j,
                shape_1,
                shape_name_1,
                shape_2,
                shape_name_2,
                ex_num,
                mask_overlapping_gravity_to_shape_pred,
            )

    mask_overlapping_gravity_to_shape_set = {
        (ex_num, 0, 2, shape_name_1, shape_name_2),
        (ex_num, 0, 3, shape_name_1, shape_name_2),
        (ex_num, 1, 2, shape_name_1, shape_name_2),
        (ex_num, 1, 3, shape_name_1, shape_name_2),
        (ex_num, 1, 4, shape_name_1, shape_name_2),
        (ex_num, 2, 2, shape_name_1, shape_name_2),
        (ex_num, 2, 3, shape_name_1, shape_name_2),
    }

    assert (
        background_knowledge[mask_overlapping_gravity_to_shape_pred.name]
        == mask_overlapping_gravity_to_shape_set
    )


def test_append_background_knowledge_for_expandable_shapes_ex1():
    input_shape = Shape(
        (8, 8),
        np.array([[2, 1], [4, 7]], dtype=np.int16),
        ShapeType.MIXED_COLOUR,
    )

    input_shape_name = "shape"
    ex_num = 1
    mask_overlapping_expanded_to_grid_pred = Predicate(
        "mask-overlapping-expanded-to-grid",
        0,
        [],
    )

    mask_overlapping_and_colour_expanded_to_grid_pred = Predicate(
        "mask-overlapping-and-colour-expanded-to-grid",
        0,
        [],
    )

    background_knowledge = defaultdict(set)

    possible_colours = [colour for colour in Colour]

    for i in range(10):
        for j in range(10):
            Solver._append_background_knowledge_for_expandable_shapes(
                background_knowledge,
                i,
                j,
                input_shape,
                np.zeros((6, 6)),
                ex_num,
                input_shape_name,
                mask_overlapping_expanded_to_grid_pred,
                mask_overlapping_and_colour_expanded_to_grid_pred,
                possible_colours,
            )

    mask_overlapping_and_colour_expanded_to_grid_set = {
        (ex_num, Colour(2), 0, 0, input_shape_name),
        (ex_num, Colour(2), 0, 1, input_shape_name),
        (ex_num, Colour(2), 0, 2, input_shape_name),
        (ex_num, Colour(2), 1, 0, input_shape_name),
        (ex_num, Colour(2), 1, 1, input_shape_name),
        (ex_num, Colour(2), 1, 2, input_shape_name),
        (ex_num, Colour(2), 2, 0, input_shape_name),
        (ex_num, Colour(2), 2, 1, input_shape_name),
        (ex_num, Colour(2), 2, 2, input_shape_name),
        (ex_num, Colour(4), 3, 0, input_shape_name),
        (ex_num, Colour(4), 3, 1, input_shape_name),
        (ex_num, Colour(4), 3, 2, input_shape_name),
        (ex_num, Colour(4), 4, 0, input_shape_name),
        (ex_num, Colour(4), 4, 1, input_shape_name),
        (ex_num, Colour(4), 4, 2, input_shape_name),
        (ex_num, Colour(4), 5, 0, input_shape_name),
        (ex_num, Colour(4), 5, 1, input_shape_name),
        (ex_num, Colour(4), 5, 2, input_shape_name),
        (ex_num, Colour(1), 0, 3, input_shape_name),
        (ex_num, Colour(1), 0, 4, input_shape_name),
        (ex_num, Colour(1), 0, 5, input_shape_name),
        (ex_num, Colour(1), 1, 3, input_shape_name),
        (ex_num, Colour(1), 1, 4, input_shape_name),
        (ex_num, Colour(1), 1, 5, input_shape_name),
        (ex_num, Colour(1), 2, 3, input_shape_name),
        (ex_num, Colour(1), 2, 4, input_shape_name),
        (ex_num, Colour(1), 2, 5, input_shape_name),
        (ex_num, Colour(7), 3, 3, input_shape_name),
        (ex_num, Colour(7), 3, 4, input_shape_name),
        (ex_num, Colour(7), 3, 5, input_shape_name),
        (ex_num, Colour(7), 4, 3, input_shape_name),
        (ex_num, Colour(7), 4, 4, input_shape_name),
        (ex_num, Colour(7), 4, 5, input_shape_name),
        (ex_num, Colour(7), 5, 3, input_shape_name),
        (ex_num, Colour(7), 5, 4, input_shape_name),
        (ex_num, Colour(7), 5, 5, input_shape_name),
    }

    mask_overlapping_expanded_to_grid_set = {
        (ex_num, i, j, input_shape_name) for i in range(6) for j in range(6)
    }

    assert (
        background_knowledge[mask_overlapping_and_colour_expanded_to_grid_pred.name]
        == mask_overlapping_and_colour_expanded_to_grid_set
    )

    assert (
        background_knowledge[mask_overlapping_expanded_to_grid_pred.name]
        == mask_overlapping_expanded_to_grid_set
    )


def test_append_background_knowledge_for_expandable_shapes_ex2():
    input_shape = Shape(
        (2, 1),
        np.array([[2, 1, 0], [4, 7, 7]], dtype=np.int16),
        ShapeType.MIXED_COLOUR,
    )

    input_shape_name = "shape"
    ex_num = 1
    mask_overlapping_expanded_to_grid_pred = Predicate(
        "mask-overlapping-expanded-to-grid",
        0,
        [],
    )

    mask_overlapping_and_colour_expanded_to_grid_pred = Predicate(
        "mask-overlapping-and-colour-expanded-to-grid",
        0,
        [],
    )

    background_knowledge = defaultdict(set)

    possible_colours = [colour for colour in Colour]

    for i in range(10):
        for j in range(10):
            Solver._append_background_knowledge_for_expandable_shapes(
                background_knowledge,
                i,
                j,
                input_shape,
                np.zeros((4, 6)),
                ex_num,
                input_shape_name,
                mask_overlapping_expanded_to_grid_pred,
                mask_overlapping_and_colour_expanded_to_grid_pred,
                possible_colours,
            )

    mask_overlapping_and_colour_expanded_to_grid_set = {
        (ex_num, Colour(2), 0, 0, input_shape_name),
        (ex_num, Colour(2), 0, 1, input_shape_name),
        (ex_num, Colour(2), 1, 0, input_shape_name),
        (ex_num, Colour(2), 1, 1, input_shape_name),
        (ex_num, Colour(4), 2, 0, input_shape_name),
        (ex_num, Colour(4), 2, 1, input_shape_name),
        (ex_num, Colour(4), 3, 0, input_shape_name),
        (ex_num, Colour(4), 3, 1, input_shape_name),
        (ex_num, Colour(1), 0, 2, input_shape_name),
        (ex_num, Colour(1), 0, 3, input_shape_name),
        (ex_num, Colour(1), 1, 2, input_shape_name),
        (ex_num, Colour(1), 1, 3, input_shape_name),
        (ex_num, Colour(7), 2, 2, input_shape_name),
        (ex_num, Colour(7), 2, 3, input_shape_name),
        (ex_num, Colour(7), 2, 4, input_shape_name),
        (ex_num, Colour(7), 2, 5, input_shape_name),
        (ex_num, Colour(7), 3, 2, input_shape_name),
        (ex_num, Colour(7), 3, 3, input_shape_name),
        (ex_num, Colour(7), 3, 4, input_shape_name),
        (ex_num, Colour(7), 3, 5, input_shape_name),
    }

    mask_overlapping_expanded_to_grid_set = {
        (ex[0], ex[2], ex[3], ex[4])
        for ex in mask_overlapping_and_colour_expanded_to_grid_set
    }

    assert (
        background_knowledge[mask_overlapping_and_colour_expanded_to_grid_pred.name]
        == mask_overlapping_and_colour_expanded_to_grid_set
    )

    assert (
        background_knowledge[mask_overlapping_expanded_to_grid_pred.name]
        == mask_overlapping_expanded_to_grid_set
    )


def test_append_background_knowledge_for_repeatable_shapes_ex1():
    input_shape = Shape(
        (8, 6),
        np.array([[0, 7, 7], [7, 7, 7], [0, 6, 6]], dtype=np.int16),
        ShapeType.MIXED_COLOUR,
    )

    input_shape_name = "shape"
    ex_num = 1
    mask_overlapping_repeated_grid_pred = Predicate(
        "mask-overlapping-repeated-grid",
        0,
        [],
    )

    mask_overlapping_and_colour_repeated_grid_pred = Predicate(
        "mask-overlapping-and-colour-repeated-grid",
        0,
        [],
    )

    background_knowledge = defaultdict(set)

    possible_colours = [colour for colour in Colour]

    for i in range(6):
        for j in range(6):
            Solver._append_background_knowledge_for_repeatable_shapes(
                background_knowledge,
                i,
                j,
                input_shape,
                np.zeros((6, 6)),
                ex_num,
                input_shape_name,
                mask_overlapping_repeated_grid_pred,
                mask_overlapping_and_colour_repeated_grid_pred,
                possible_colours,
            )

    mask_overlapping_and_colour_repeated_grid_set = {
        (ex_num, Colour(7), 0, 1, input_shape_name),
        (ex_num, Colour(7), 0, 2, input_shape_name),
        (ex_num, Colour(7), 1, 0, input_shape_name),
        (ex_num, Colour(7), 1, 1, input_shape_name),
        (ex_num, Colour(7), 1, 2, input_shape_name),
        (ex_num, Colour(6), 2, 1, input_shape_name),
        (ex_num, Colour(6), 2, 2, input_shape_name),
        (ex_num, Colour(7), 0, 4, input_shape_name),
        (ex_num, Colour(7), 0, 5, input_shape_name),
        (ex_num, Colour(7), 1, 3, input_shape_name),
        (ex_num, Colour(7), 1, 4, input_shape_name),
        (ex_num, Colour(7), 1, 5, input_shape_name),
        (ex_num, Colour(6), 2, 4, input_shape_name),
        (ex_num, Colour(6), 2, 5, input_shape_name),
        (ex_num, Colour(7), 3, 4, input_shape_name),
        (ex_num, Colour(7), 3, 5, input_shape_name),
        (ex_num, Colour(7), 4, 3, input_shape_name),
        (ex_num, Colour(7), 4, 4, input_shape_name),
        (ex_num, Colour(7), 4, 5, input_shape_name),
        (ex_num, Colour(6), 5, 4, input_shape_name),
        (ex_num, Colour(6), 5, 5, input_shape_name),
        (ex_num, Colour(7), 3, 1, input_shape_name),
        (ex_num, Colour(7), 3, 2, input_shape_name),
        (ex_num, Colour(7), 4, 0, input_shape_name),
        (ex_num, Colour(7), 4, 1, input_shape_name),
        (ex_num, Colour(7), 4, 2, input_shape_name),
        (ex_num, Colour(6), 5, 1, input_shape_name),
        (ex_num, Colour(6), 5, 2, input_shape_name),
    }

    mask_overlapping_repeated_grid_set = {
        (ex[0], ex[2], ex[3], ex[4])
        for ex in mask_overlapping_and_colour_repeated_grid_set
    }

    assert (
        background_knowledge[mask_overlapping_and_colour_repeated_grid_pred.name]
        == mask_overlapping_and_colour_repeated_grid_set
    )

    assert (
        background_knowledge[mask_overlapping_repeated_grid_pred.name]
        == mask_overlapping_repeated_grid_set
    )
