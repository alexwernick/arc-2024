from typing import List

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.interpreter import Interpreter


class Solver:
    inputs: list[NDArray[np.int16]]
    outputs: list[NDArray[np.int16]]
    test_inputs: list[NDArray[np.int16]]

    def __init__(
        self,
        inputs: list[NDArray[np.int16]],
        outputs: list[NDArray[np.int16]],
        test_inputs: list[NDArray[np.int16]],
    ):
        for arr in inputs + outputs + test_inputs:
            if arr.ndim != 2:
                raise ValueError(
                    "Arrays in lists inputs, outputs & test_inputs must be 2D"
                )

        self.inputs = inputs
        self.outputs = outputs
        self.test_inputs = test_inputs

    def solve(self) -> List[NDArray[np.int16]]:
        """
        This function solves the task.
        """
        interpreter = Interpreter(self.inputs, self.outputs, self.test_inputs)
        (
            inputs_shapes,
            outputs_shapes,
            test_input_shapes,
        ) = interpreter.interpret_shapes()

        # Find all realtionshonships between shapes

        # look at outputs relations to inputs and then compare
        relationships: list[set] = [set() for _ in range(len(inputs_shapes))]

        for example_index, (input_shapes, output_shapes) in enumerate(
            zip(inputs_shapes, outputs_shapes)
        ):
            for input_index, input_shape in enumerate(input_shapes):
                for output_index, output_shape in enumerate(output_shapes):
                    for (
                        relationship_name,
                        relationship,
                    ) in output_shape.relationships.items():
                        relationships[example_index].add(
                            (
                                input_index,
                                output_index,
                                relationship_name,
                                relationship(input_shape),
                            )
                        )

        return self.inputs
