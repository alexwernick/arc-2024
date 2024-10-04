import numpy as np
import pytest

from arc_2024.representations.shape import Shape


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int16)),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(
                None,
                (1, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
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
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int16)),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (1, 0), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(
                None,
                (1, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
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
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0], [0, 0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0], [0, 0]], dtype=np.int16)),
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (0, 2), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(
                None,
                (0, 1),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
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
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0], [0, 0]], dtype=np.int16)),
            Shape(None, (0, 0), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (0, 0), np.array([[0, 0], [0, 0]], dtype=np.int16)),
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int16),
            ),
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(
                None,
                (0, 1),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
            ),
            Shape(
                None,
                (0, 0),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int16),
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
            Shape(None, (1, 3), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            Shape(None, (1, 3), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (1, 2), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            False,
        ),
    ],
)
def test_is_inline_horizontally_above_right(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_horizontally_above_right(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(None, (1, 1), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            Shape(None, (1, 1), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (0, 1), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            False,
        ),
    ],
)
def test_is_inline_horizontally_above_left(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_horizontally_above_left(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(None, (3, 3), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            Shape(None, (3, 3), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (3, 2), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            False,
        ),
    ],
)
def test_is_inline_horizontally_below_right(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_horizontally_below_right(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(None, (3, 1), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            True,
        ),
        (
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            Shape(None, (3, 1), np.array([[0]], dtype=np.int16)),
            False,
        ),
        (
            Shape(None, (3, 2), np.array([[0]], dtype=np.int16)),
            Shape(None, (2, 2), np.array([[0]], dtype=np.int16)),
            False,
        ),
    ],
)
def test_is_inline_horizontally_below_left(shape1, shape2, expected_result):
    # Exercise code
    result = shape1.is_inline_horizontally_below_left(shape2)

    # Verify code
    assert result == expected_result


@pytest.mark.parametrize(
    "shape1,shape2,expected_result",
    [
        (
            Shape(
                None,
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
            ),
            Shape(None, (0, 1), np.array([[1, 1], [0, 1]], dtype=np.int16)),
            False,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
            ),
            Shape(None, (1, 1), np.array([[1, 1], [0, 1]], dtype=np.int16)),
            True,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
            ),
            Shape(None, (0, 0), np.array([[1, 1], [0, 1]], dtype=np.int16)),
            True,
        ),
        (
            Shape(
                None,
                (0, 0),
                np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16),
            ),
            Shape(None, (2, 0), np.array([[1, 1], [0, 1]], dtype=np.int16)),
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
        None, (2, 3), np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int16)
    )

    # Exercise code
    result = shape.all_pixels()

    # Verify code
    assert result == [(2, 3), (3, 3), (4, 3), (4, 4), (4, 5)]
