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
    result = shape.horizontal_distance_from_center_ij(0, j)

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
    result = shape.vertical_distance_from_center_ij(i, 0)

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
    result = shape.horizontal_distance_from_edge_ij(0, j)

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
    result = shape.vertical_distance_from_edge_ij(i, 0)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    (
        "rot_number,is_above,is_below,is_left_of,is_right_of,"
        "is_inline_diagonally_above_right,"
        "is_inline_diagonally_above_left,"
        "is_inline_diagonally_below_right,"
        "is_inline_diagonally_below_left"
    ),
    [
        (0, False, True, False, True, False, False, True, False),
        (1, True, False, False, True, True, False, False, False),
        (2, True, False, True, False, False, True, False, False),
        (3, False, True, True, False, False, False, False, True),
    ],
)
def test_rotatable_mask_shape(
    rot_number,
    is_above,
    is_below,
    is_left_of,
    is_right_of,
    is_inline_diagonally_above_right,
    is_inline_diagonally_above_left,
    is_inline_diagonally_below_right,
    is_inline_diagonally_below_left,
):
    # Setup code
    i = 0
    j = 0
    mask = np.array([[1, 1], [1, 0]], dtype=np.int16)
    fixed_mask = np.rot90(mask.astype(bool), rot_number)

    shape = RotatableMaskShape(
        (2, 2),
        mask,
        fixed_mask,
        shape_type=ShapeType.SINGLE_COLOUR,
    )

    # Assert
    assert shape.is_above_ij(i, j) == is_above
    assert shape.is_below_ij(i, j) == is_below
    assert shape.is_left_of_ij(i, j) == is_left_of
    assert shape.is_right_of_ij(i, j) == is_right_of
    assert (
        shape.is_inline_diagonally_above_right_ij(i, j)
        == is_inline_diagonally_above_right
    )
    assert (
        shape.is_inline_diagonally_above_left_ij(i, j)
        == is_inline_diagonally_above_left
    )
    assert (
        shape.is_inline_diagonally_below_right_ij(i, j)
        == is_inline_diagonally_below_right
    )
    assert (
        shape.is_inline_diagonally_below_left_ij(i, j)
        == is_inline_diagonally_below_left
    )
    assert shape.horizontal_distance_from_center_ij(i, j) == 2.5
    assert shape.vertical_distance_from_center_ij(i, j) == 2.5
    assert shape.horizontal_distance_from_edge_ij(i, j) == 2
    assert shape.vertical_distance_from_edge_ij(i, j) == 2


def test_find_internal_blank_spaces_in_mask():
    mask = np.array(
        [
            [3, 3, 3, 3, 0, 0, 0, 0],
            [3, 0, 0, 3, 0, 0, 0, 0],
            [3, 0, 0, 3, 0, 3, 0, 0],
            [3, 0, 3, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 0, 0, 3, 0],
            [0, 3, 0, 0, 0, 3, 3, 0],
            [0, 3, 3, 0, 0, 3, 0, 3],
            [0, 3, 0, 3, 0, 0, 3, 0],
            [0, 0, 3, 0, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    shape = Shape(
        (1, 2),
        mask,
        shape_type=ShapeType.SINGLE_COLOUR,
    )

    internal_blank_spaces = shape._find_internal_blank_spaces_in_mask()

    assert len(internal_blank_spaces) == 3

    for internal_blank_space in internal_blank_spaces:
        if internal_blank_space[0] == (2, 3):
            assert np.array_equal(
                internal_blank_space[1],
                np.array(
                    [
                        [True, True],
                        [True, True],
                        [True, False],
                    ],
                    dtype=np.bool_,
                ),
            )
        elif internal_blank_space[0] == (8, 4):
            assert np.array_equal(
                internal_blank_space[1], np.full((1, 1), True, dtype=np.bool_)
            )
        elif internal_blank_space[0] == (7, 8):
            assert np.array_equal(
                internal_blank_space[1], np.full((1, 1), True, dtype=np.bool_)
            )
        else:
            assert False


def test_is_ij_inside_blank_space():
    true_positions = [(2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (8, 4), (7, 8)]
    mask = np.array(
        [
            [3, 3, 3, 3, 0, 0, 0, 0],
            [3, 0, 0, 3, 0, 0, 0, 0],
            [3, 0, 0, 3, 0, 3, 0, 0],
            [3, 0, 3, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 0, 0, 3, 0],
            [0, 3, 0, 0, 0, 3, 3, 0],
            [0, 3, 3, 0, 0, 3, 0, 3],
            [0, 3, 0, 3, 0, 0, 3, 0],
            [0, 0, 3, 0, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    shape = Shape(
        (1, 2),
        mask,
        shape_type=ShapeType.SINGLE_COLOUR,
    )

    for i in range(10):
        for j in range(10):
            if (i, j) in true_positions:
                result = shape.is_ij_inside_blank_space(i, j)
                assert result
            else:
                result = shape.is_ij_inside_blank_space(i, j)
                assert not result
