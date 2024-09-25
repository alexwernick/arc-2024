import pytest

from arc_2024.representations.colour import Color
from arc_2024.representations.interpreter import Interpreter
from arc_2024.representations.shape import Shape


def test_interprets_all_individual_pixels_of_colour(interpreter: Interpreter):
    # Exercise code
    interpreted_shapes = interpreter.interpret_shapes()

    # Verify code
    inputs_shapes = interpreted_shapes.inputs
    outputs_shapes = interpreted_shapes.outputs
    test_inputs_shapes = interpreted_shapes.test_inputs

    # Define the condition to check
    def pixel_exists_in_shapes(shape: Shape, color: Color, j: int, k: int) -> bool:
        return (
            shape.num_of_coloured_pixels == 1
            and shape.height == 1
            and shape.width == 1
            and shape.colour == color
            and shape.position == (j, k)
            and shape.mask == [[1]]
        )

    def check_shapes(raw_data, shapes):
        for i, input in enumerate(raw_data):
            for j in range(input.shape[0]):
                for k in range(input.shape[1]):
                    if input[j, k] == 0:
                        continue

                    color = Color(input[j, k])
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
    interpreted_shapes = interpreter.interpret_shapes()

    # Verify code
    inputs_shapes = interpreted_shapes.inputs
    outputs_shapes = interpreted_shapes.outputs
    test_inputs_shapes = interpreted_shapes.test_inputs

    def check_shapes(shapes, expected_shapes):
        for i, expected_shapes_per_example in enumerate(expected_shapes):
            for expected_shape in expected_shapes_per_example:
                matching_shapes = [
                    shape for shape in shapes[i] if expected_shape == shape
                ]
                assert len(matching_shapes) == 1

    check_shapes(inputs_shapes, expected_input_shapes)
    check_shapes(outputs_shapes, expected_output_shapes)
    check_shapes(test_inputs_shapes, expected_test_input_shapes)
