# noinspection SpellCheckingInspection
"""
This file implements the marching algorithm for RFFQM as appears in the paper:

Fan Feng, Xiangxin Dang, Richard D. James, Paul Plucinsky,
The designs and deformations of rigidly and flat-foldable quadrilateral mesh origami
https://doi.org/10.1016/j.jmps.2020.104018

The basic idea behind the algorithm is that given the angles and the lengths of
the left and bottom boundaries, the entire crease pattern is determined
step-by-step
"""
from typing import Optional, List, Tuple

import numpy as np

norm = np.linalg.norm
sin = np.sin
cos = np.cos


class MarchingAlgorithm(object):
    """
    This class implements the marching algorithm
    We utilize the fact that compatibility of the angles can be checked independently
    and before checking compatibility of the lengths

    The computed angles are stored in 2 arrays, alphas and betas
    """

    def __init__(self, angles_left: np.ndarray, angles_bottom: np.ndarray,
                 sigmas: Optional[np.ndarray] = None):
        """
        Constructor
        :param angles_left: Array of shape (2, rows)
        :param angles_bottom: Array of shape (2, cols-1)
        :param sigmas: Array of +- of shape (rows, cols). Default is all -
        """
        # TODO: Fix sigmas to be only at the boundaries
        self._assert_valid_input_angles(angles_left, angles_bottom)

        rows, cols = angles_left.shape[1], angles_bottom.shape[1] + 1
        alphas = np.zeros((rows, cols), dtype=np.float128)
        betas = np.zeros((rows, cols), dtype=np.float128)

        if sigmas is None:
            sigmas = np.ones(alphas.shape) * -1
        else:
            assert sigmas.shape == alphas.shape

        alphas[:, 0] = angles_left[0, :]
        betas[:, 0] = angles_left[1, :]
        alphas[0, 1:] = angles_bottom[0, :]
        betas[0, 1:] = angles_bottom[1, :]

        self.rows = rows
        self.cols = cols
        self.alphas = alphas
        self.betas = betas
        self.sigmas = sigmas

        self._fill_angles()
        self._ls = np.zeros((rows - 1, cols))
        self._cs = np.zeros((rows, cols - 1))

    @staticmethod
    def _assert_valid_input_angles(angles_left: np.ndarray, angles_bottom: np.ndarray):
        if np.any(angles_left < 0):
            raise ValueError("Got a left boundary angle with negative value")
        if np.any(angles_bottom < 0):
            raise ValueError("Got a bottom boundary angle with negative value")
        if np.any(angles_left > np.pi):
            raise ValueError("Got a left boundary angle with value greater then pi")
        if np.any(angles_bottom > np.pi):
            raise ValueError("Got a bottom boundary angle with value greater then pi")

    def _calc_angles(self, i, j):
        """
        Calculate alphas[i,j] and betas[i,j] based on the values at
        (i-1,j) (i,j-1) (i-1,j-1)

        See eq. 63
        """
        # These angles are calculated based on Kawasaki condition
        # see eq. 57 in paper
        alpha_a, beta_a, sigma_a = self.alphas[i - 1, j - 1], self.betas[i - 1, j - 1], self.sigmas[i - 1, j - 1]
        alpha_c = self.betas[i - 1, j]
        beta_c = self.alphas[i - 1, j]
        sigma_c = self.sigmas[i - 1, j]
        alpha_b = np.pi - self.betas[i, j - 1]
        beta_b = np.pi - self.alphas[i, j - 1]
        sigma_b = self.sigmas[i, j - 1]

        angles_sum = alpha_a + alpha_b + alpha_c
        # First check in paper:
        alpha_d = 2 * np.pi - angles_sum
        if alpha_d < 0:
            raise IncompatibleError(
                f"Got a negative angle at index {i},{j}. Angle value: {alpha_d}")
        if alpha_d > np.pi:
            raise IncompatibleError(
                f"Got angle that is beyond pi at index {i},{j}. Angle value: {alpha_d}")

        mu_a = mu1(alpha_a, beta_a, -sigma_a)
        mu_b = mu2(alpha_b, beta_b, sigma_b)
        mu_c = mu2(alpha_c, beta_c, sigma_c)

        mu_abc = mu_a * mu_b * mu_c
        # Second check in paper
        if np.isclose(np.abs(mu_abc), 1):
            raise IncompatibleError(
                f"Got |mu| close to 1 at index {i},{j}. mu value: {mu_abc}")
        sigma_d = -np.sign(mu_abc ** 2 - 1)
        cos_angles = cos(angles_sum)
        beta_d = np.arccos(sigma_d *
                           (2 * mu_abc - (mu_abc ** 2 + 1) * cos_angles) /
                           (2 * mu_abc * cos_angles - (mu_abc ** 2 + 1)))

        self.sigmas[i, j] = sigma_d
        self.alphas[i, j] = np.pi - alpha_d
        self.betas[i, j] = np.pi - beta_d

    def _fill_angles(self):
        for i in range(1, self.rows):
            for j in range(1, self.cols):
                self._calc_angles(i, j)

    def create_dots(self,
                    ls_left: float | List[float] | np.ndarray,
                    cs_bottom: float | List[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the dots that corresponds the crease pattern
        :param ls_left: List of lengths of the left boundary (rows-1 cells)
        :param cs_bottom: List of lengths of the bottom boundary (cols-1 cells)
        :return: (Dots, Indexes)
        """
        rows, cols = self.rows, self.cols
        dots = np.zeros((2, rows * cols), dtype=np.float128)
        indexes = np.arange(rows * cols).reshape((rows, cols))

        if hasattr(ls_left, '__len__'):
            assert len(ls_left) == rows - 1
        else:
            ls_left = np.ones(rows - 1) * ls_left
        if hasattr(cs_bottom, '__len__'):
            assert len(cs_bottom) == cols - 1
        else:
            cs_bottom = np.ones(cols - 1) * cs_bottom
        if isinstance(ls_left, list):
            ls_left = np.array(ls_left)
        if isinstance(cs_bottom, list):
            cs_bottom = np.array(cs_bottom)

        if np.any(ls_left <= 0):
            raise ValueError("Got a non-positive left boundary crease length")
        if np.any(cs_bottom <= 0):
            raise ValueError("Got a non-positive bottom boundary crease length")

        # Calculate position for bottom boundary
        angle = np.pi / 2 - self.alphas[0, 0]
        for j in range(1, cols):
            length = cs_bottom[j - 1]
            self._cs[0, j - 1] = length
            vec = np.array([np.cos(angle), np.sin(angle)])
            dots[:, indexes[0, j]] = dots[:, indexes[0, j - 1]] + vec * length
            angle = angle + np.pi - (self.alphas[0, j] + self.betas[0, j])

        # Calculate position for left boundary
        angle = np.pi / 2
        for i in range(1, rows):
            length = ls_left[i - 1]
            self._ls[i - 1, 0] = length
            vec = np.array([np.cos(angle), np.sin(angle)])
            dots[:, indexes[i, 0]] = dots[:, indexes[i - 1, 0]] + vec * length
            angle = angle - np.pi + (self.alphas[i, 0] + np.pi - self.betas[i, 0])

        for i in range(1, rows):
            for j in range(1, cols):
                self._calc_position(dots, indexes, i, j)

        return dots, indexes

    def _calc_position(self, dots, indexes, i, j):
        # See equation 64
        l_ac = norm(dots[:, indexes[i - 1, j]] - dots[:, indexes[i - 1, j - 1]])
        l_ab = norm(dots[:, indexes[i, j - 1]] - dots[:, indexes[i - 1, j - 1]])

        alpha_a = self.alphas[i - 1, j - 1]
        alpha_b = np.pi - self.betas[i, j - 1]
        alpha_c = self.betas[i - 1, j]
        angles_sum = alpha_a + alpha_b + alpha_c

        matrix = 1 / sin(angles_sum) * np.array([[-sin(alpha_b), sin(alpha_a + alpha_b)],
                                                 [sin(alpha_a + alpha_c), -sin(alpha_c)]])

        l_cd, l_bd = matrix @ np.array([l_ab, l_ac])
        if l_cd <= 0:
            raise IncompatibleError(
                f"Got a non-positive crease line for line cd at index {i},{j}. "
                f"length got: {l_cd}")
        if l_bd <= 0:
            raise IncompatibleError(
                f"Got a non-positive crease line for line bd at index {i},{j}. "
                f"length got: {l_bd}")
        self._cs[i, j - 1] = l_bd
        self._ls[i - 1, j] = l_cd

        vec_ca = dots[:, indexes[i - 1, j - 1]] - dots[:, indexes[i - 1, j]]
        vec_ca = vec_ca / norm(vec_ca)
        angle = -alpha_c
        rotation_matrix = np.array([[cos(angle), -sin(angle)],
                                    [sin(angle), cos(angle)]])
        vec_cd = rotation_matrix @ vec_ca
        dots[:, indexes[i, j]] = dots[:, indexes[i - 1, j]] + vec_cd * l_cd

    def get_ls(self) -> np.ndarray:
        return self._ls

    def get_cs(self) -> np.ndarray:
        return self._cs


def mu1(a, b, s):
    """
    Calculate mu1 based on equation 35 in the paper
    """
    return mu2(a, np.pi - b, s)


def mu2(a, b, s):
    """
    Calculate mu1 based on equation 35 in the paper
    """
    num = -s + cos(a) * cos(b) + sin(a) * sin(b)
    # noinspection SpellCheckingInspection
    denom = cos(b) - s * cos(a)
    return num / denom


class IncompatibleError(Exception):
    pass


def create_miura_angles(ls, cs, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the angles in the L boundary for a basic Miura-Ori pattern
    :param ls: Array of the vertical lengths
    :param cs: Array of the horizontal lengths
    :param angle: The angle of the parallelograms in the Miura-Ori
    :return: 2 Arrays, of shape (2,len(ls)+1); (2,len(cs))
        they represent the left angles and the bottom angles.
        Note that the vertex at the L-shape
    """
    angles_left = np.ones((2, len(ls) + 1), dtype=np.float128) * angle
    angles_bottom = np.ones((2, len(cs)), dtype=np.float128) * angle
    angles_bottom[:, ::2] = np.pi - angle

    return angles_left, angles_bottom
