import logging

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import origami.quadranglearray
from origami import miuraori
from origami.utils import linalgutils

logger = logging.getLogger('origami')
logger.setLevel(logging.DEBUG)


class ZigzagMiuraOri(object):
    def __init__(self, crease_dots: np.ndarray, d_rows: int, d_cols: int):
        assert crease_dots.shape[0] == 2
        assert crease_dots.shape[1] == d_rows * d_cols

        self.d_rows = d_rows
        self.d_cols = d_cols
        self.indexes = np.arange(d_rows * d_cols).reshape((d_rows, d_cols))
        self.initial_dots = np.zeros((3, d_rows * d_cols))
        self.initial_dots[:2, :] = crease_dots

        self.dots = self.initial_dots.copy()

        self._omega = 0

    def set_omega(self, omega, should_center=True):
        self._omega = omega
        self.dots = self.initial_dots.copy()
        dots = self.dots
        indexes = self.indexes

        omega = omega

        R_vertical = linalgutils.create_XZ_rotation_matrix(omega)

        # We need to choose the start sign such that we get to the most left vertical
        # crease, the sign would be positive, so in agreement with the sign of omega
        start_sign_x = 1 if self.d_rows % 2 == 1 else -1
        omega_sign_factor = 1 if self.d_cols % 2 == 0 else -1

        # The first row is done externally, because we don't change signs there
        Rv_alternating = R_vertical.copy()
        if start_sign_x == -1:
            Rv_alternating = Rv_alternating.transpose()

        p_rows = self.d_rows - 1
        p_cols = self.d_cols - 1

        for c in range(p_cols, 1, -1):
            dots_indices = indexes[-1:, c:].flat
            logger.debug('Changing last row. cols indices {}'.format(np.array(dots_indices)))

            base_x = dots[0, indexes[-1, c - 1]]
            dots[0, dots_indices] -= base_x
            dots[:, dots_indices] = Rv_alternating @ dots[:, dots_indices]
            dots[0, dots_indices] += base_x
            Rv_alternating = Rv_alternating.transpose()

        for r in range(p_rows - 1, -1, -1):
            Rv_alternating = R_vertical.copy()
            if start_sign_x == -1:
                Rv_alternating = Rv_alternating.transpose()
            start_sign_x *= -1

            for c in range(p_cols, 1, -1):
                dots_indices = indexes[r, c:]
                logger.debug('Changing cols indices {}'.format(np.array(dots_indices)))

                base_x = dots[0, indexes[r, c - 1]]
                dots[0, dots_indices] -= base_x
                dots[:, dots_indices] = Rv_alternating @ dots[:, dots_indices]
                dots[0, dots_indices] += base_x
                Rv_alternating = Rv_alternating.transpose()

            if r == 0:
                break

            h_axis = dots[:, indexes[r, 1]] - dots[:, indexes[r, 0]]
            y_axis = np.array([0, 1, 0])
            o = omega_sign_factor * start_sign_x * omega
            h_angle = miuraori.calc_gamma(linalgutils.calc_angle(h_axis, y_axis), 1, o)

            Rh = linalgutils.create_rotation_around_axis(h_axis, h_angle)
            dots_indices = self.indexes[r:, :].flat
            logger.debug('Changing row indices {}'.format(np.array(dots_indices)))
            base_y = dots[1, self.indexes[r, 0]]
            dots[1, dots_indices] -= base_y
            dots[:, dots_indices] = Rh @ dots[:, dots_indices]
            dots[1, dots_indices] += base_y

        if should_center:
            self.dots = origami.quadranglearray.center_dots(self.dots, self.indexes)

    def get_omega(self):
        return self._omega

    def plot(self, ax: Axes3D, alpha=1):
        return origami.dotsarray.plot(self.dots, self.indexes, ax, alpha)

    def is_valid(self):
        return origami.quadranglearray.is_valid(self.initial_dots, self.dots, self.indexes)
