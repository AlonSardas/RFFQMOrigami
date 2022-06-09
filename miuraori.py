import matplotlib.axes
import numpy as np
import logging

logger = logging.getLogger('origami')
logger.setLevel(logging.DEBUG)


class SimpleMiuraOri(object):
    def __init__(self, ls_x, ls_y, angle=np.pi / 4):
        self.ls_x = ls_x
        self.ls_y = ls_y
        self.angle = angle
        self.beta = np.pi / 2 - angle
        self.alpha = np.pi - self.beta

        self.num_of_panels = len(ls_x) * len(ls_y)
        self.rows = len(ls_y)
        self.columns = len(ls_x)
        self._d_rows = self.rows + 1
        self._d_cols = self.columns + 1

        # This array contains 1 dot per panel (the bottom left for example)
        # with 3 coordinates
        self.initial_dots = np.zeros((3, self._d_rows * self._d_cols))
        self.indexes = np.arange(self._d_rows * self._d_cols).reshape((self._d_rows, self._d_cols))
        self._build_dots()
        self.dots = self.initial_dots.copy()
        self._omega = 0
        self._omega = 0

    def _build_dots(self):
        dxs = np.append(0, self.ls_x)
        xs = np.cumsum(dxs)
        xs = xs.astype('float64')
        xs *= np.cos(self.angle)
        for r in range(self._d_rows):
            self.initial_dots[0, self.indexes[r, :]] = xs

        ys = np.append(0, np.cumsum(self.ls_y))
        ys = ys.astype('float64')
        sign = +1
        for c in range(self._d_cols):
            ys += sign * dxs[c] * np.sin(self.angle)
            self.initial_dots[1, self.indexes[:, c]] = ys
            sign *= -1

    def plot(self, ax: matplotlib.axes.Axes, should_rotate=True, should_center=True, alpha=1):
        # print('zs: {}'.format(dots[2, :].reshape(self._d_rows, self._d_cols)))

        # ax.scatter3D(self.dots[0, :], self.dots[1, :], self.dots[2, :])
        # ax.scatter3D(self.dots[0, 4:], self.dots[1, 4:], self.dots[2, 4:])

        plot_dots = self.dots.copy()
        if should_rotate:
            angle = self.angle / 2
            R_xy = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])
            plot_dots = R_xy.transpose() @ plot_dots

        if should_center:
            plot_dots -= plot_dots.mean(axis=1)[:, None]

        ax.plot_surface(plot_dots[0, :].reshape((self._d_rows, self._d_cols)),
                        plot_dots[1, :].reshape((self._d_rows, self._d_cols)),
                        plot_dots[2, :].reshape((self._d_rows, self._d_cols)), alpha=alpha)

    def set_omega(self, omega):
        self._omega = omega
        self.dots = self.initial_dots.copy()
        dots = self.dots

        sigma = 1
        omega1 = omega
        omega2 = self._calc_omega2(sigma, omega1)
        self._gamma = omega2

        logger.debug('omega1: {}, omega2: {}'.format(omega1, omega2))

        c1 = np.cos(omega1)
        s1 = np.sin(omega1)

        R1 = np.array([
            [c1, 0, s1],
            [0, 1, 0],
            [-s1, 0, c1]])
        R2 = self._create_R2(omega2)

        # Not sure why, but this condition works
        if self._d_cols % 2 == 1:
            start_sign_x = 1
        else:
            start_sign_x = -1
        start_sign_y = -1

        R2_alternating = R2.copy()
        if start_sign_y == -1:
            R2_alternating = R2_alternating.transpose()

        # The first row is done externally, beause we don't change signs there
        R1_alternating = R1.copy()
        if start_sign_x == -1:
            R1_alternating = R1_alternating.transpose()
        # """
        for c in range(self.columns, 1, -1):
            dots_indices = self.indexes[-1:, c:].flat
            logger.debug('Changing last row. cols indices {}'.format(np.array(dots_indices)))

            base_x = dots[0, self.indexes[-1, c - 1]]
            dots[0, dots_indices] -= base_x
            dots[:, dots_indices] = R1_alternating @ dots[:, dots_indices]
            dots[0, dots_indices] += base_x
            R1_alternating = R1_alternating.transpose()

        for r in range(self.rows - 1, -1, -1):
            R1_alternating = R1.copy()
            if start_sign_x == -1:
                R1_alternating = R1_alternating.transpose()
            start_sign_x *= -1

            for c in range(self.columns, 1, -1):
                dots_indices = self.indexes[r, c:]
                logger.debug('Changing cols indices {}'.format(np.array(dots_indices)))

                base_x = dots[0, self.indexes[r, c - 1]]
                dots[0, dots_indices] -= base_x
                dots[:, dots_indices] = R1_alternating @ dots[:, dots_indices]
                dots[0, dots_indices] += base_x
                R1_alternating = R1_alternating.transpose()
            # """
            # """
            if r == 0:
                # It is "ok" but not necessary to fold the last row, since
                # it translates to a rigid rotation
                R2_alternating = self._create_R2(+omega2 / 2)
                # R2_alternating = R2_alternating.transpose()

            dots_indices = self.indexes[r:, :].flat
            logger.debug('Changing row indices {}'.format(np.array(dots_indices)))
            base_y = dots[1, self.indexes[r, 0]]
            dots[1, dots_indices] -= base_y
            dots[:, dots_indices] = R2_alternating @ dots[:, dots_indices]
            dots[1, dots_indices] += base_y

            R2_alternating = R2_alternating.transpose()

    def get_omega(self):
        return self._omega

    def _create_R2(self, omega2):
        c2 = np.cos(omega2)
        s2 = np.sin(omega2)

        R2_t = np.array([
            [c2, 0, s2],
            [0, 1, 0],
            [-s2, 0, c2]])
        angle = np.pi / 2 - self.angle
        R_xy = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]])
        R2 = R_xy @ R2_t @ R_xy.transpose()
        return R2

    def _calc_omega2(self, s, o):
        a = np.pi / 2 - self.angle
        b = np.pi - a
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        cos_b = np.cos(b)
        sin_b = np.sin(b)
        nom = (-s + cos_a * cos_b) * np.cos(o) + sin_a * sin_b
        deno = -s + cos_a * cos_b + sin_a * sin_b * np.cos(o)
        return np.arccos(nom / deno)

    def get_gamma(self):
        return self._gamma
