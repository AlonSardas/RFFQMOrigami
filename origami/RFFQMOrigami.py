# noinspection SpellCheckingInspection
"""
This file implements the kinematics of a general RFFQM. For further reading, see:

Fan Feng, Xiangxin Dang, Richard D. James, Paul Plucinsky,
The designs and deformations of rigidly and flat-foldable quadrilateral mesh origami
https://doi.org/10.1016/j.jmps.2020.104018

We assume a given crease pattern that must be valid as RFFQM. Then calculate the
position of the quadrangles given an activation angle omega
"""
import logging
from typing import Optional

import numpy as np

from origami.quadranglearray import QuadrangleArray
from origami.utils import linalgutils

norm = np.linalg.norm
sin = np.sin
cos = np.cos
PI = np.pi

logger = logging.getLogger('origami')
logger.setLevel(logging.DEBUG)


class RFFQM(object):
    def __init__(self, crease_pattern: QuadrangleArray, sigmas: Optional[np.ndarray] = None):
        self.rows, self.cols = crease_pattern.rows, crease_pattern.cols
        self.initial_dots = crease_pattern
        self.dots = crease_pattern.copy()
        self.indexes = self.initial_dots.indexes
        self.gamma = 0

        if sigmas is None:
            # In a Miura-Ori all the sigmas are -1
            sigmas = -np.ones((self.rows + 1, self.cols + 1))
        else:
            assert sigmas.shape == (self.rows + 1, self.cols + 1)
        self.sigmas = sigmas
        self.angles = self._calc_angles()

    def _calc_angles(self) -> np.ndarray:
        dots = self.initial_dots.dots
        indexes = self.indexes
        angles = np.zeros((2, self.rows * self.cols))
        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                x0 = dots[:, indexes[i, j]]
                t1 = dots[:, indexes[i, j + 1]] - x0
                t2 = dots[:, indexes[i + 1, j]] - x0
                t3 = dots[:, indexes[i, j - 1]] - x0
                angles[0, indexes[i, j]] = linalgutils.calc_angle(t1, t2)
                angles[1, indexes[i, j]] = linalgutils.calc_angle(t2, t3)

        return angles

    def set_gamma(self, gamma, should_center=True) -> QuadrangleArray:
        """
        See eq. 70 and after for the algorithm and notation used here.
        Note that in the paper, they use some imaginary seed values for angles that we
        don't have here. We only use inner vertices to calculate the deformation.
        This makes this implementation slightly different

        In addition, note that they calculate all the deformation gradient prior to
        calculating the kinematics, here we calculate each deformation gradient just before
        we need to use it on the specific panel
        """
        self.gamma = gamma
        self.dots = self.initial_dots.copy()  # Each time we start from a flat configuration
        indexes = self.indexes
        sigmas = self.sigmas
        rows, cols = self.rows, self.cols

        i, j = 1, 1
        alpha, beta = self.angles[:, indexes[i, j]]
        gamma_1, gamma_2, gamma_3, gamma_4 = self._calc_angles_right(
            alpha, beta, sigmas[i, j], gamma)

        # This is used as the seed for the angles for the next rows
        initial_gamma2 = gamma_2

        gamma_1 = gamma

        i = 1
        # We fold first the bottom row
        for j in range(1, cols - 1):
            alpha, beta = self.angles[:, indexes[i, j]]
            gamma_1, gamma_2, gamma_3, gamma_4 = self._calc_angles_right(
                alpha, beta, sigmas[i, j], gamma_1)

            dots_indices = indexes[:2, :j].flat
            logger.debug(
                f'Changing vertical crease at first row. j={j}. angle: {gamma_4:.3f}. '
                f'Indices to change: {np.array(dots_indices)}')
            self._rotate_crease(i, j, 4, gamma_4, dots_indices)

        gamma_1 = gamma

        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                alpha, beta = self.angles[:, indexes[i, j]]
                gamma_1, gamma_2, gamma_3, gamma_4 = self._calc_angles_right(
                    alpha, beta, sigmas[i, j], gamma_1)

                dots_indices = indexes[i + 1, :j].flat
                logger.debug(f'Changing vertical crease at {i},{j}. Angle: {gamma_2}. '
                             f'Indices to change {np.array(dots_indices)}')
                self._rotate_crease(i, j, 2, -gamma_2, dots_indices)

            j = cols - 2
            dots_indices = indexes[:i + 1, :].flat
            logger.debug(f'Changing horizontal crease at row {i}. angle: {gamma_1}. '
                         f'indices to change: {np.array(dots_indices)}')
            self._rotate_crease(i, j, 1, gamma_1, dots_indices)

            if i == rows - 2:
                break  # No need to calculate more angles if we are done

            # See eq. 76 for the angles one step up
            j = 1
            alpha, beta = self.angles[:, indexes[i + 1, j]]
            gamma_1, gamma_2, gamma_3, gamma_4 = self._calc_angles_up(
                alpha, beta, sigmas[i + 1, j], initial_gamma2)

            initial_gamma2 = gamma_2  # Save the seed of a vertical crease to the next row
            gamma_1 = gamma_3  # This is the seed angle for the next row

        if should_center:
            self.dots.rotate_and_center()

        return self.dots

    def calc_gamma_by_omega(self, omega):
        """
        This is for convenient when there when sometime we want to determine gamma,
        and sometime we want to determine omega of the first vertex.
        It happened because of inconsistency in the naming of omega and gamma
        """
        i, j = 1, 1
        alpha, beta = self.angles[:, self.indexes[i, j]]
        omega = calc_gamma1(self.sigmas[i, j], omega, alpha, beta)
        return omega

    def calc_omegas_vs_x(self) -> np.ndarray:
        cols = self.cols
        quads = self.dots
        omegas = np.zeros(cols // 2 - 1)
        for j in range(len(omegas)):
            jj = 2 * j + 1
            base_dot = quads.dots[:, quads.indexes[0, jj]]
            v0 = quads.dots[:, quads.indexes[1, jj]] - base_dot
            v1 = quads.dots[:, quads.indexes[0, jj - 1]] - base_dot
            v2 = quads.dots[:, quads.indexes[0, jj + 1]] - base_dot
            n1 = np.cross(v1, v0)
            n2 = np.cross(v0, v2)
            omegas[j] = linalgutils.calc_angle(n1, n2)
        return omegas

    def calc_gammas_vs_y(self) -> np.ndarray:
        rows = self.rows
        quads = self.dots
        gammas = np.zeros(rows // 2 - 1)
        for i in range(len(gammas)):
            ii = 2 * i + 1
            base_dot = quads.dots[:, quads.indexes[ii, 0]]
            v0 = quads.dots[:, quads.indexes[ii, 1]] - base_dot
            v1 = quads.dots[:, quads.indexes[ii - 1, 0]] - base_dot
            v2 = quads.dots[:, quads.indexes[ii + 1, 0]] - base_dot
            n1 = np.cross(v1, v0)
            n2 = np.cross(v0, v2)
            gammas[i] = linalgutils.calc_angle(n1, n2)
        return gammas

    def _rotate_crease(self, i0, j0, d, angle, indexes_to_change):
        i1, j1 = i0, j0
        if d == 1:
            j1 += 1
        elif d == 2:
            i1 += 1
        elif d == 3:
            j1 -= 1
        elif d == 4:
            i1 -= 1
        else:
            raise RuntimeError("Unknown direction of a crease")

        dots = self.dots.dots
        initial_dots = self.initial_dots.dots
        indexes = self.indexes

        x0 = initial_dots[:, indexes[i0, j0]]
        t = initial_dots[:, indexes[i1, j1]] - x0
        if t[2] != 0:
            raise RuntimeError(
                "Folding around a crease only works when the crease is in the XY-plane. "
                f"Got a crease line {t}")
        R = linalgutils.create_rotation_around_axis(t, angle)
        dots[:, indexes_to_change] = x0[:, np.newaxis] + R @ (dots[:, indexes_to_change] - x0[:, np.newaxis])

    @staticmethod
    def _calc_angles_right(alpha, beta, sigma, gamma_1):
        # See eq. 75 for the angles one step right
        gamma_3 = gamma_1
        gamma_1 = -sigma * gamma_3
        gamma_2 = calc_gamma2(sigma, gamma_1, alpha, beta)
        gamma_4 = sigma * gamma_2
        return gamma_1, gamma_2, gamma_3, gamma_4

    @staticmethod
    def _calc_angles_up(alpha, beta, sigma, gamma_2):
        gamma_4 = gamma_2
        gamma_2 = sigma * gamma_4
        gamma_1 = calc_gamma1(sigma, gamma_2, alpha, beta)
        gamma_3 = -sigma * gamma_1
        return gamma_1, gamma_2, gamma_3, gamma_4


def calc_gamma2(sigma: int, omega: float, alpha: float, beta: float) -> float:
    """
    Calc the angle gamma2
    See eq. 10
    """
    s, o, a, b = sigma, omega, alpha, beta
    nom = (-s + cos(a) * cos(b)) * cos(o) + sin(a) * sin(b)
    deno = -s + cos(a) * cos(b) + sin(a) * sin(b) * cos(o)

    sgn = np.sign((s * cos(b) - cos(a)) * o)

    return sgn * np.arccos(nom / deno)


def calc_gamma1(sigma: int, omega: float, alpha: float, beta: float) -> float:
    """
    Calc the angle gamma1
    See eq. 11
    """
    return calc_gamma2(-sigma, omega, alpha, PI - beta)
