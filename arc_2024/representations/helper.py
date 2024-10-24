import numpy as np


def surrounding_coordinates(
    i: int, j: int, i_max: int, j_max: int, include_diagonals: bool = True
) -> set:
    directions = [
        (i - 1, j),
        (i, j - 1),
        (i, j + 1),
        (i + 1, j),
    ]

    diagonal_directions = [
        (i - 1, j - 1),
        (i - 1, j + 1),
        (i + 1, j - 1),
        (i + 1, j + 1),
    ]

    if include_diagonals:
        directions.extend(diagonal_directions)

    return {
        (x, y) for x, y in directions if x >= 0 and y >= 0 and x < i_max and y < j_max
    }


def find_smallest_indices_greater_than_q(arr: np.ndarray, q) -> tuple[int, int]:
    rows_greater_than_q = np.any(arr > q, axis=1)
    cols_greater_than_q = np.any(arr > q, axis=0)

    min_i = np.where(rows_greater_than_q)[0]
    min_j = np.where(cols_greater_than_q)[0]

    if min_i.size > 0 and min_j.size > 0:
        return int(min_i[0]), int(min_j[0])
    else:
        raise ValueError(f"No row and column are greater than '{q}'")


def find_smallest_indices_equal_to_q(arr: np.ndarray, q) -> tuple[int, int]:
    rows_greater_than_q = np.any(arr == q, axis=1)
    cols_greater_than_q = np.any(arr == q, axis=0)

    min_i = np.where(rows_greater_than_q)[0]
    min_j = np.where(cols_greater_than_q)[0]

    if min_i.size > 0 and min_j.size > 0:
        return int(min_i[0]), int(min_j[0])
    else:
        raise ValueError(f"No row and column contain '{q}'")


def remove_rows_and_cols_with_value_x(arr: np.ndarray, x) -> np.ndarray:
    # Remove rows with all zeros
    arr = arr[~np.all(arr == x, axis=1)]
    # Remove columns with all zeros
    arr = arr[:, ~np.all(arr == x, axis=0)]
    return arr
