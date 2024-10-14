from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from arc_2024.representations.shape import Shape
from arc_2024.representations.shape_type import ShapeType

RelationshipType = Callable[["Shape"], bool]


class RotatableMaskShape(Shape):
    _fixed_mask: NDArray[np.bool]
    _rotate_ij: Callable[[float, float, float, float], Tuple[float, float]]

    def __init__(
        self,
        position: Tuple[int, int],
        mask: NDArray[np.int16],
        fixed_mask: NDArray[np.bool],
        shape_type: ShapeType,
    ):
        super().__init__(position, mask, shape_type)
        self._fixed_mask = fixed_mask

        rot0: NDArray[np.bool] = self.mask.astype(bool)
        rot90 = np.rot90(rot0)
        rot180 = np.rot90(rot90)
        rot270 = np.rot90(rot180)

        if np.array_equal(rot0, fixed_mask):
            self._rotate_ij = self._rotate_0
        elif np.array_equal(rot90, fixed_mask):
            self._rotate_ij = self._rotate_90
        elif np.array_equal(rot180, fixed_mask):
            self._rotate_ij = self._rotate_180
        elif np.array_equal(rot270, fixed_mask):
            self._rotate_ij = self._rotate_270
        else:
            raise ValueError("Fixed mask does not match any rotation of the mask")

    def __eq__(self, other):
        if not isinstance(other, RotatableMaskShape):
            return False
        is_eq = super().__eq__(other)
        mask_equal = np.array_equal(self.fixed_mask, other.fixed_mask)
        return is_eq and mask_equal

    def is_above_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is above i
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_above_ij(rot_i, rot_j)

    def is_below_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is below i
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_below_ij(rot_i, rot_j)

    def is_left_of_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is left of i
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_left_of_ij(rot_i, rot_j)

    def is_right_of_ij(self, i: Union[int, float], j: Union[int, float]) -> bool:
        """
        Returns True if self is right of i
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_right_of_ij(rot_i, rot_j)

    def is_inline_diagonally_above_right_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and above and to the right
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_inline_diagonally_above_right_ij(rot_i, rot_j)

    def is_inline_diagonally_above_left_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and above and to the left
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_inline_diagonally_above_left_ij(rot_i, rot_j)

    def is_inline_diagonally_below_right_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and below and to the right
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_inline_diagonally_below_right_ij(rot_i, rot_j)

    def is_inline_diagonally_below_left_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> bool:
        """
        Returns True if self inline with i,j diagonally and below and to the left
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().is_inline_diagonally_below_left_ij(rot_i, rot_j)

    def horizontal_distance_from_center_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The horizontal distance between the center and the point j
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().horizontal_distance_from_center_ij(rot_i, rot_j)

    def vertical_distance_from_center_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The vertical distance between the center and the point i
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().vertical_distance_from_center_ij(rot_i, rot_j)

    def horizontal_distance_from_edge_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The horizontal distance between the center and the point j
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().horizontal_distance_from_edge_ij(rot_i, rot_j)

    def vertical_distance_from_edge_ij(
        self, i: Union[int, float], j: Union[int, float]
    ) -> Union[int, float]:
        """
        The vertical distance between the edge and the point i
        """
        rot_i, rot_j = self._rotate_ij(i, j, self.centre[0], self.centre[1])
        return super().vertical_distance_from_edge_ij(rot_i, rot_j)

    @staticmethod
    def _rotate_0(
        i: Union[int, float], j: Union[int, float], x: float, y: float
    ) -> Tuple[float, float]:
        return i, j

    @staticmethod
    def _rotate_90(
        i: Union[int, float], j: Union[int, float], x: float, y: float
    ) -> Tuple[float, float]:
        # Translate to origin
        i_translated = i - x
        j_translated = j - y
        # Rotate 90 degrees
        i_rotated = -j_translated
        j_rotated = i_translated
        # Translate back
        i_final = i_rotated + x
        j_final = j_rotated + y
        return i_final, j_final

    @staticmethod
    def _rotate_180(
        i: Union[int, float], j: Union[int, float], x: float, y: float
    ) -> Tuple[float, float]:
        # Translate to origin
        i_translated = i - x
        j_translated = j - y
        # Rotate 180 degrees
        i_rotated = -i_translated
        j_rotated = -j_translated
        # Translate back
        i_final = i_rotated + x
        j_final = j_rotated + y
        return i_final, j_final

    @staticmethod
    def _rotate_270(
        i: Union[int, float], j: Union[int, float], x: float, y: float
    ) -> Tuple[float, float]:
        # Translate to origin
        i_translated = i - x
        j_translated = j - y
        # Rotate 270 degrees
        i_rotated = j_translated
        j_rotated = -i_translated
        # Translate back
        i_final = i_rotated + x
        j_final = j_rotated + y
        return i_final, j_final
