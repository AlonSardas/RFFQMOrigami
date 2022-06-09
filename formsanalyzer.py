import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from miuraori import SimpleMiuraOri


class SimpleMiuraOriFormAnalyzer(object):
    def __init__(self, ls_x, ls_y, angle=np.pi / 4):
        assert len(ls_x) == 2
        assert len(ls_y) == 4
        self.origami = SimpleMiuraOri(ls_x, ls_y, angle)
        self.ps = np.zeros((3, 4))
        self.ns = np.zeros((3, 4))
        self.set_omega(0.001)

    def set_omega(self, omega):
        self.origami.set_omega(omega)
        origami = self.origami
        indexes = origami.indexes
        dots = origami.dots

        ps_in_middle = False
        if ps_in_middle:
            # This is only one definition to the points, that a point
            # is located in the middle of the y-zigzag
            p1 = 1 / 2 * (dots[:, indexes[0, 0]] + dots[:, indexes[2, 0]])
            p2 = 1 / 2 * (dots[:, indexes[2, 0]] + dots[:, indexes[4, 0]])
            p3 = 1 / 2 * (dots[:, indexes[0, 2]] + dots[:, indexes[2, 2]])
            p4 = 1 / 2 * (dots[:, indexes[2, 2]] + dots[:, indexes[4, 2]])
        else:
            p1 = dots[:, indexes[2, 0]]
            p2 = dots[:, indexes[4, 0]]
            p3 = dots[:, indexes[2, 2]]
            p4 = dots[:, indexes[4, 2]]

        self.ps = np.stack([p1, p2, p3, p4], axis=1)
        n1 = self._calc_normal(1, 0)
        n2 = self._calc_normal(3, 0)
        n3 = self._calc_normal(1, 2)
        n4 = self._calc_normal(3, 2)
        self.ns = np.stack([n1, n2, n3, n4], axis=1)

    def _calc_normal(self, i, j):
        origami = self.origami
        indexes = origami.indexes
        dots = origami.dots
        l1t1 = dots[:, indexes[i - 1, j]] - dots[:, indexes[i, j]]
        l2t2 = dots[:, indexes[i + 1, j]] - dots[:, indexes[i, j]]
        n_perp = l2t2 - l1t1
        plane_norm = np.cross(l1t1, l2t2)
        n = np.cross(n_perp, plane_norm)
        assert np.abs(np.sum(n * n_perp)) < 1e-6, \
            'The vectors are not perpendicular. n={}, n_perp={}'.format(n, n_perp)
        n = n / np.linalg.norm(n)
        return n

    def plot_pattern(self, ax):
        self.origami.plot(ax, should_center=False, should_rotate=False, alpha=0.3)

    def plot_forms_patches(self, ax: Axes3D):
        ps = self.ps
        ns = self.ns
        ax.scatter(ps[0, :], ps[1, :], ps[2, :])

        for i in range(4):
            arrow = np.stack([ps[:, i], ps[:, i] + ns[:, i]], axis=1)
            ax.plot(arrow[0, :], arrow[1, :], arrow[2, :], '-')

    def plot(self, ax):
        self.plot_pattern(ax)
        self.plot_forms_patches(ax)

    def compare_to_theory(self):
        origami = self.origami
        alpha = origami.alpha
        beta = origami.beta
        ps = self.ps
        omega = origami.get_omega()
        gamma = origami.get_gamma()

        d_x = ps[:, 2] - ps[:, 0]
        d_y = ps[:, 1] - ps[:, 0]

        a_11 = np.linalg.norm(d_x) ** 2
        a_12 = np.inner(d_x, d_y)
        a_22 = np.linalg.norm(d_y) ** 2

        phi = 1 / 2 * np.arccos(-np.cos(omega) * np.sin(beta) ** 2 + np.cos(beta) ** 2)

        t_d_x = origami.ls_x[0] * np.array([np.sin(phi), -np.cos(phi), 0]) + \
                origami.ls_x[1] * np.array([np.sin(phi), +np.cos(phi), 0])

        theta = 1 / 2 * np.arccos(-np.sin(alpha) ** 2 * np.cos(gamma) - np.cos(alpha) ** 2)
        t1 = np.array([0, -np.sin(theta), np.cos(theta)])
        t2 = np.array([0, np.sin(theta), np.cos(theta)])
        t_d_y = origami.ls_y[3] * t2 - origami.ls_y[2] * t1

        t_a_11 = np.linalg.norm(t_d_x) ** 2
        t_a_12 = np.inner(t_d_x, t_d_y)
        t_a_22 = np.linalg.norm(t_d_y) ** 2

        assert np.abs(t_a_11 - a_11) < 1e-9, \
            'a_11 do not agree. theoretical: {}, computed: {}'.format(t_a_11, a_11)
        assert np.abs(t_a_12 - a_12) < 1e-9, \
            'a_12 do not agree. theoretical: {}, computed: {}'.format(t_a_12, a_12)
        assert np.abs(t_a_22 - a_22) < 1e-9, \
            'a_22 do not agree. theoretical: {}, computed: {}'.format(t_a_22, a_22)

        t_d_x_d_y = np.cos(phi) * np.sin(theta) * (origami.ls_y[2] + origami.ls_y[3]) * (
                    origami.ls_x[1] - origami.ls_x[0])
        assert np.isclose(t_d_x_d_y, t_a_12), \
            'The expression for inner product a_12 is incorrect'

        def calc_n(l1, l2):
            norm = 1 / np.sqrt((l2 - l1) ** 2 + 4 * l1 * l2 * np.sin(theta) ** 2)
            v = np.array([0, (l1 - l2) * np.cos(theta), (l1 + l2) * np.sin(theta)])
            return v * norm

        t_n_1 = calc_n(origami.ls_y[0], origami.ls_y[1])
        t_n_2 = calc_n(origami.ls_y[2], origami.ls_y[3])

        n1 = self.ns[:, 0]
        n2 = self.ns[:, 1]

        assert np.isclose(np.linalg.norm(t_n_1), 1), \
            'The calculated normal vector n1 is not normalized. n={}'.format(t_n_1)
        assert np.isclose(np.linalg.norm(t_n_2), 1), \
            'The calculated normal vector n2 is not normalized. n={}'.format(t_n_2)

        # These assertion do not work, since in the computed folding there is also
        # a rigid rotation. To avoid this issue, we compare the inner product of the normals
        # instead
        # ------
        # assert np.abs(np.inner(t_n_1, n1) - 1) < 1e-6, \
        #     'n1 do not agree. theoretical: {}, computed: {}'.format(t_n_1, n1)
        # assert np.abs(np.inner(t_n_2, n2) - 1) < 1e-6, \
        #     'n2 do not agree. theoretical: {}, computed: {}'.format(t_n_1, n2)
        # ------
        assert np.isclose(np.inner(t_n_1, t_n_2), np.inner(n1, n2)), \
            'The dot product of the normals do not agree.'

        print('Theoretical and computed values agree!')
