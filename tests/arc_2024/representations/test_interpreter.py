import json
from pathlib import Path

import numpy as np
import pytest

from arc_2024.representations.colour import Colour
from arc_2024.representations.interpreter import Interpreter
from arc_2024.representations.rotatable_mask_shape import RotatableMaskShape
from arc_2024.representations.shape import Shape
from arc_2024.representations.shape_type import ShapeType


def create_interpreter_for_task(task_id: str) -> Interpreter:
    file_path = Path(__file__).parent / f"test_data/{task_id}.json"
    with open(str(file_path), "r") as file:
        data = json.load(file)

    inputs = [np.array(item["input"]) for item in data["train"]]
    outputs = [np.array(item["output"]) for item in data["train"]]
    test_inputs = [np.array(item["input"]) for item in data["test"]]
    return Interpreter(inputs, outputs, test_inputs)


def test_interprets_all_individual_pixels_of_colour(interpreter: Interpreter):
    # Exercise code
    interpretations = interpreter.interpret_shapes()
    interpreted_shapes = next(
        (
            x
            for x in interpretations
            if x.interpret_type == Interpreter.InterpretType.LOCAL_SEARCH
        ),
        None,
    )

    # Verify code
    assert interpreted_shapes is not None
    inputs_shapes = interpreted_shapes.inputs
    outputs_shapes = interpreted_shapes.outputs
    test_inputs_shapes = interpreted_shapes.test_inputs

    # Define the condition to check
    def pixel_exists_in_shapes(shape: Shape, color: Colour, j: int, k: int) -> bool:
        return (
            shape.num_of_coloured_pixels == 1
            and shape.height == 1
            and shape.width == 1
            and shape.colour == color
            and shape.position == (j, k)
            and shape.mask == [[color.value]]
            and shape.shape_type.name == "PIXEL"
        )

    def check_shapes(raw_data, shapes):
        for i, input in enumerate(raw_data):
            for j in range(input.shape[0]):
                for k in range(input.shape[1]):
                    if input[j, k] == 0:
                        continue

                    color = Colour(input[j, k])
                    matching_shapes = [
                        shape
                        for shape in shapes[i]
                        if pixel_exists_in_shapes(shape, color, j, k)
                    ]
                    assert len(matching_shapes) == 1

    check_shapes(interpreter.inputs, inputs_shapes)
    check_shapes(interpreter.outputs, outputs_shapes)
    check_shapes(interpreter.test_inputs, test_inputs_shapes)


@pytest.mark.parametrize("task_id", [("6d75e8bb"), ("6e19193c")])
def test_interprets_all_shapes(interpreter_dictiorary, task_id):
    # Setup
    interpreter = interpreter_dictiorary[task_id].interpreter
    expected_input_shapes = interpreter_dictiorary[task_id].expected_input_shapes
    expected_output_shapes = interpreter_dictiorary[task_id].expected_output_shapes
    expected_test_input_shapes = interpreter_dictiorary[
        task_id
    ].expected_test_input_shapes

    # Exercise code
    interpretations = interpreter.interpret_shapes()
    interpreted_shapes = next(
        (
            x
            for x in interpretations
            if x.interpret_type == Interpreter.InterpretType.LOCAL_SEARCH
        ),
        None,
    )

    # Verify code
    assert interpreted_shapes is not None
    inputs_shapes = interpreted_shapes.inputs
    outputs_shapes = interpreted_shapes.outputs
    test_inputs_shapes = interpreted_shapes.test_inputs

    def check_shapes(shapes, expected_shapes):
        for i, expected_shapes_per_example in enumerate(expected_shapes):
            # we don't care about non pixel shapes
            non_pixel_shapes = [
                shape for shape in shapes[i] if shape.shape_type.name != "PIXEL"
            ]
            assert len(non_pixel_shapes) == len(expected_shapes_per_example)
            for expected_shape in expected_shapes_per_example:
                matching_shapes = [
                    shape for shape in non_pixel_shapes if expected_shape == shape
                ]
                assert len(matching_shapes) == 1

    check_shapes(inputs_shapes, expected_input_shapes)
    check_shapes(outputs_shapes, expected_output_shapes)
    check_shapes(test_inputs_shapes, expected_test_input_shapes)


# def test_interprets_colour_count_shape_groups(interpreter: Interpreter):
#     interpreter = create_interpreter_for_task("0b148d64")
#     # Exercise code
#     interpretations = interpreter.interpret_shapes()
#     interpreted_shapes = next(
#         (
#             x
#             for x in interpretations
#             if x.interpret_type == Interpreter.InterpretType.SEPERATOR
#         ),
#         None,
#     )

#     # Verify code
#     assert interpreted_shapes is not None
#     inputs_shapes = interpreted_shapes.inputs
#     test_inputs_shapes = interpreted_shapes.test_inputs

#     for shape in inputs_shapes[0]:
#         if shape.shape_type.name != "SINGLE_COLOUR":
#             continue

#         if shape.colour == Colour(8):
#             assert "GROUP_COLOUR_COUNT-3" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-0" in shape.shape_groups
#         elif shape.colour == Colour(2):
#             assert "GROUP_COLOUR_COUNT-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-0" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-1" in shape.shape_groups

#     for shape in inputs_shapes[1]:
#         if shape.shape_type.name != "SINGLE_COLOUR":
#             continue

#         if shape.colour == Colour(2):
#             assert "GROUP_COLOUR_COUNT-3" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-0" in shape.shape_groups
#         elif shape.colour == Colour(3):
#             assert "GROUP_COLOUR_COUNT-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-0" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-1" in shape.shape_groups

#     for shape in inputs_shapes[2]:
#         if shape.shape_type.name != "SINGLE_COLOUR":
#             continue

#         if shape.colour == Colour(1):
#             assert "GROUP_COLOUR_COUNT-3" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-0" in shape.shape_groups
#         elif shape.colour == Colour(4):
#             assert "GROUP_COLOUR_COUNT-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-0" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-1" in shape.shape_groups

#     for shape in test_inputs_shapes[0]:
#         if shape.shape_type.name != "SINGLE_COLOUR":
#             continue

#         if shape.colour == Colour(3):
#             assert "GROUP_COLOUR_COUNT-3" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-0" in shape.shape_groups
#         elif shape.colour == Colour(1):
#             assert "GROUP_COLOUR_COUNT-1" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_ASC-0" in shape.shape_groups
#             assert "GROUP_COLOUR_COUNT_DESC-1" in shape.shape_groups


def test_split_array_on_zeros_with_indices():
    array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 4, 0, 4, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [4, 4, 4, 4, 0, 4, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [4, 0, 4, 0, 0, 4, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 4, 4, 4, 4, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            [4, 4, 4, 0, 4, 4, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 4, 4, 4, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 4, 4, 4, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [4, 4, 0, 4, 0, 4, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    sub_arrays_with_indices = Interpreter._split_array_on_zeros_with_indices(array)

    assert len(sub_arrays_with_indices) == 4

    array_1 = np.array(
        [
            [0, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=np.int16,
    )

    array_2 = np.array(
        [
            [1, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 1],
        ],
        dtype=np.int16,
    )

    array_3 = np.array(
        [
            [4, 0, 0, 4, 0, 4],
            [4, 4, 4, 4, 0, 4],
            [4, 0, 4, 0, 0, 4],
            [0, 4, 4, 4, 4, 0],
            [4, 4, 4, 0, 4, 4],
            [0, 4, 4, 4, 4, 0],
            [0, 4, 4, 4, 0, 4],
            [0, 4, 0, 0, 0, 0],
            [4, 4, 0, 4, 0, 4],
        ],
        dtype=np.int16,
    )

    array_4 = np.array(
        [
            [1, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 0],
        ],
        dtype=np.int16,
    )

    assert np.array_equal(sub_arrays_with_indices[0][1], array_1)
    assert np.array_equal(sub_arrays_with_indices[1][1], array_2)
    assert np.array_equal(sub_arrays_with_indices[2][1], array_3)
    assert np.array_equal(sub_arrays_with_indices[3][1], array_4)
    assert sub_arrays_with_indices[0][0] == (1, 0)
    assert sub_arrays_with_indices[1][0] == (1, 8)
    assert sub_arrays_with_indices[2][0] == (11, 0)
    assert sub_arrays_with_indices[3][0] == (11, 8)


def test_interprets_seperator_shapes_ex1():
    # Setup
    interpreter = create_interpreter_for_task("7c008303")

    interpretations = interpreter.interpret_shapes()
    interpreted_shapes = next(
        (
            x
            for x in interpretations
            if x.interpret_type == Interpreter.InterpretType.SEPERATOR
        ),
        None,
    )

    expected_inputs_shapes = [set() for _ in range(3)]
    expected_test_inputs_shapes = [set() for _ in range(1)]

    # ex 1
    expected_inputs_shapes[0].add(
        Shape(
            (0, 0),
            np.array(
                [[8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8]], dtype=np.int16
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[0].add(
        Shape(
            (0, 7),
            np.array(
                [[8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8]], dtype=np.int16
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[0].add(
        Shape(
            (0, 11),
            np.array(
                [[8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8]], dtype=np.int16
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[0].add(
        Shape(
            (0, 3),
            np.array(
                [[8], [8], [8], [8], [8], [8], [8], [8], [8], [8], [8]], dtype=np.int16
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[0].add(
        Shape(
            (3, 0),
            np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[0].add(
        Shape(
            (0, 0),
            np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[0].add(
        Shape(
            (10, 0),
            np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[0].add(
        Shape(
            (1, 1),
            np.array([[2, 4], [1, 6]], dtype=np.int16),
            shape_type=ShapeType.MIXED_COLOUR,
        )
    )

    expected_inputs_shapes[0].add(
        Shape(
            (4, 4),
            np.array(
                [[0, 3, 0], [3, 3, 3], [0, 3, 0], [0, 3, 0], [3, 3, 3], [0, 3, 0]],
                dtype=np.int16,
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[0].add(
        Shape(
            (4, 8),
            np.array(
                [[0, 3, 0], [3, 3, 3], [0, 3, 0], [0, 3, 0], [3, 3, 3], [0, 3, 0]],
                dtype=np.int16,
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # ex 2
    expected_inputs_shapes[1].add(
        Shape(
            (0, 6),
            np.array([[8], [8], [8], [8], [8], [8], [8], [8], [8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[1].add(
        Shape(
            (2, 0),
            np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[1].add(
        Shape(
            (0, 7),
            np.array([[1, 2], [4, 1]], dtype=np.int16),
            shape_type=ShapeType.MIXED_COLOUR,
        )
    )

    expected_inputs_shapes[1].add(
        Shape(
            (3, 0),
            np.array(
                [
                    [0, 0, 3, 3, 0, 3],
                    [3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 3, 0, 3],
                    [0, 0, 0, 0, 3, 0],
                    [3, 3, 3, 3, 3, 3],
                    [0, 0, 0, 0, 3, 0],
                ],
                dtype=np.int16,
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # ex 3
    expected_inputs_shapes[2].add(
        Shape(
            (0, 2),
            np.array([[8], [8], [8], [8], [8], [8], [8], [8], [8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_inputs_shapes[2].add(
        Shape(
            (6, 0),
            np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[2].add(
        Shape(
            (7, 0),
            np.array([[2, 4], [6, 5]], dtype=np.int16),
            shape_type=ShapeType.MIXED_COLOUR,
        )
    )

    expected_inputs_shapes[2].add(
        Shape(
            (0, 3),
            np.array(
                [
                    [0, 0, 3, 0, 0, 3],
                    [0, 0, 3, 0, 0, 3],
                    [3, 3, 0, 3, 3, 0],
                    [0, 0, 0, 0, 3, 0],
                    [0, 3, 0, 3, 0, 0],
                    [0, 3, 0, 0, 0, 3],
                ],
                dtype=np.int16,
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # test ex 1
    expected_test_inputs_shapes[0].add(
        Shape(
            (0, 6),
            np.array([[8], [8], [8], [8], [8], [8], [8], [8], [8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )
    expected_test_inputs_shapes[0].add(
        Shape(
            (6, 0),
            np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_test_inputs_shapes[0].add(
        Shape(
            (7, 7),
            np.array([[2, 1], [4, 7]], dtype=np.int16),
            shape_type=ShapeType.MIXED_COLOUR,
        )
    )

    expected_test_inputs_shapes[0].add(
        Shape(
            (0, 0),
            np.array(
                [
                    [0, 0, 0, 3, 0, 0],
                    [3, 3, 0, 3, 0, 3],
                    [0, 3, 0, 3, 0, 3],
                    [0, 3, 3, 3, 0, 0],
                    [0, 3, 0, 0, 0, 3],
                    [0, 0, 3, 0, 0, 0],
                ],
                dtype=np.int16,
            ),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # Verify code
    assert interpreted_shapes is not None
    inputs_shapes = interpreted_shapes.inputs
    test_inputs_shapes = interpreted_shapes.test_inputs

    for input_shapes, expected_input_shapes in zip(
        inputs_shapes, expected_inputs_shapes
    ):
        inps = set(
            [
                shape
                for shape in input_shapes
                if not isinstance(shape, RotatableMaskShape)
            ]
        )
        assert inps == expected_input_shapes

    for input_shapes, expected_input_shapes in zip(
        test_inputs_shapes, expected_test_inputs_shapes
    ):
        inps = set(
            [
                shape
                for shape in input_shapes
                if not isinstance(shape, RotatableMaskShape)
            ]
        )
        assert inps == expected_input_shapes


def test_interprets_seperator_shapes_ex2():
    # Setup
    interpreter = create_interpreter_for_task("0520fde7")

    interpretations = interpreter.interpret_shapes()
    interpreted_shapes = next(
        (
            x
            for x in interpretations
            if x.interpret_type == Interpreter.InterpretType.SEPERATOR
        ),
        None,
    )

    expected_inputs_shapes = [set() for _ in range(3)]
    expected_test_inputs_shapes = [set() for _ in range(1)]

    # ex 1
    expected_inputs_shapes[0].add(
        Shape(
            (0, 3),
            np.array([[5], [5], [5]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[0].add(
        Shape(
            (0, 0),
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[0].add(
        Shape(
            (0, 4),
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # ex 2
    expected_inputs_shapes[1].add(
        Shape(
            (0, 3),
            np.array([[5], [5], [5]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[1].add(
        Shape(
            (0, 0),
            np.array([[1, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[1].add(
        Shape(
            (0, 4),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # ex 3
    expected_inputs_shapes[2].add(
        Shape(
            (0, 3),
            np.array([[5], [5], [5]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[2].add(
        Shape(
            (0, 0),
            np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_inputs_shapes[2].add(
        Shape(
            (0, 4),
            np.array([[0, 0, 0], [1, 0, 1], [1, 0, 1]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # test ex 1
    expected_test_inputs_shapes[0].add(
        Shape(
            (0, 3),
            np.array([[5], [5], [5]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_test_inputs_shapes[0].add(
        Shape(
            (0, 0),
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    expected_test_inputs_shapes[0].add(
        Shape(
            (0, 4),
            np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=np.int16),
            shape_type=ShapeType.SINGLE_COLOUR,
        )
    )

    # Verify code
    assert interpreted_shapes is not None
    inputs_shapes = interpreted_shapes.inputs
    test_inputs_shapes = interpreted_shapes.test_inputs

    for input_shapes, expected_input_shapes in zip(
        inputs_shapes, expected_inputs_shapes
    ):
        inps = set(
            [
                shape
                for shape in input_shapes
                if not isinstance(shape, RotatableMaskShape)
            ]
        )
        assert inps == expected_input_shapes

    for input_shapes, expected_input_shapes in zip(
        test_inputs_shapes, expected_test_inputs_shapes
    ):
        inps = set(
            [
                shape
                for shape in input_shapes
                if not isinstance(shape, RotatableMaskShape)
            ]
        )
        assert inps == expected_input_shapes
