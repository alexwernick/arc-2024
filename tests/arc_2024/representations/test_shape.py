import numpy as np
import pytest

from arc_2024.representations.rotatable_mask_shape import RotatableMaskShape
from arc_2024.representations.shape import Shape
from arc_2024.representations.shape_type import ShapeType


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[2]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
    ],
)
def test_is_above(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_above(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (1, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
    ],
)
def test_is_below(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_below(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
    ],
)
def test_is_left_of(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_left_of(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 1),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
    ],
)
def test_is_right_of(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_right_of(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (1, 3),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 3),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (1, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
    ],
)
def test_is_inline_diagonally_above_right(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_diagonally_above_right(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (1, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
    ],
)
def test_is_inline_diagonally_above_left(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_diagonally_above_left(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (3, 3),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (3, 3),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (3, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
    ],
)
def test_is_inline_diagonally_below_right(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_diagonally_below_right(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (3, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (3, 1),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (3, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 2),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
    ],
)
def test_is_inline_diagonally_below_left(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_diagonally_below_left(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 1),
                np.array([[1, 1], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            False,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (1, 1),
                np.array([[1, 1], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (0, 0),
                np.array([[1, 1], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
        (
            Shape(
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            Shape(
                (2, 0),
                np.array([[1, 1], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            True,
        ),
    ],
)
def test_is_mask_overlapping(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_mask_overlapping(shape2)

    # Verify code
    assert result == expected_result


def test_all_pixels():
    # Setup code
    shape = Shape(
        (2, 3),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
        shape_type=ShapeType.SINGLE_COLOUR,
    )

    # Exercise code
    result = shape.all_pixels()

    # Verify code
    assert result == [(2, 3), (3, 3), (4, 3), (4, 4), (4, 5)]


def test_rotatable_mask_shape():
    # Setup code
    shape = RotatableMaskShape(
        (0, 0),
        np.array([[1, 1], [1, 0]], dtype=np.int16),
        np.array([[1, 0], [1, 1]], dtype=np.bool),
        shape_type=ShapeType.SINGLE_COLOUR,
    )

    # Assert
    assert shape.is_inline_diagonally_below_left_ij(3, 3)


@pytest.mark.parametrize(
    "j,shape,expected_result",
    [
        (
            3,
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3,
        ),
        (
            3,
            Shape(
                (4, 6),
                np.array([[1, 1], [1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3.5,
        ),
    ],
)
def test_horizontal_distance_from_center(j, shape, expected_result):
    # Exercise code
    result = shape.horizontal_distance_from_center_j(j)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "i,shape,expected_result",
    [
        (
            3,
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3,
        ),
        (
            3,
            Shape(
                (6, 4),
                np.array([[1, 1], [1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3.5,
        ),
    ],
)
def test_vertical_distance_from_center(i, shape, expected_result):
    # Exercise code
    result = shape.vertical_distance_from_center_i(i)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "j,shape,expected_result",
    [
        (
            3,
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3,
        ),
        (
            3,
            Shape(
                (4, 6),
                np.array([[1, 1], [1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3,
        ),
        (
            6,
            Shape(
                (4, 1),
                np.array([[1, 1], [1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            4,
        ),
    ],
)
def test_horizontal_distance_from_edge(j, shape, expected_result):
    # Exercise code
    result = shape.horizontal_distance_from_edge_j(j)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "i,shape,expected_result",
    [
        (
            3,
            Shape(
                (0, 0),
                np.array([[1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3,
        ),
        (
            3,
            Shape(
                (6, 4),
                np.array([[1, 1], [1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            3,
        ),
        (
            6,
            Shape(
                (1, 4),
                np.array([[1, 1], [1, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            4,
        ),
        (
            5,
            Shape(
                (4, 6),
                np.array([[1, 1], [0, 1]], dtype=np.int16),
                shape_type=ShapeType.SINGLE_COLOUR,
            ),
            0,
        ),
    ],
)
def test_vertical_distance_from_edge(i, shape, expected_result):
    # Exercise code
    result = shape.vertical_distance_from_edge_i(i)

    # Verify code
    assert result == expected_result
