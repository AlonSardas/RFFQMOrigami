import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami import miuraori
from origami.continuousorigami import follow_curve, plot_zigzag
from origami.utils import plotutils, sympyutils

FIGURES_PATH = '../../RFFQM/Figures'


class DiscreteOrigami(object):
    def __init__(self, du, dv, beta, omega):
        self.du = du
        self.dv = dv

        us = np.arange(0.5, np.pi - 0.5, du)
        vs = np.arange(0, 2 * np.pi, dv)

        gamma = miuraori.calc_gamma(np.pi - beta, 1, omega)
        self.gamma = gamma
        self.alpha = np.pi - beta
        self.beta = beta
        alpha = self.alpha

        YZ_zigzag_angle = 1 / 2 * np.arccos(-np.sin(alpha) ** 2 * np.cos(gamma) - np.cos(alpha) ** 2)

        self.YZ_ys = -np.cos(us)
        self.YZ_zs = np.sin(us)

        self.ly, self.YZ_mid_y, self.YZ_mid_z = follow_curve(self.YZ_ys, self.YZ_zs, YZ_zigzag_angle)
        self.YZ_zigzag_angle = YZ_zigzag_angle

        XY_zigzag_angle = 1 / 2 * np.arccos(-np.sin(beta) ** 2 * np.cos(omega) + np.cos(beta) ** 2)

        self.XY_xs = vs
        self.XY_ys = np.sin(vs)
        self.lx, self.XY_mid_x, self.XY_mid_y = follow_curve(self.XY_xs, self.XY_ys, XY_zigzag_angle)
        self.XY_zigzag_angle = XY_zigzag_angle

        self.origami = miuraori.SimpleMiuraOri(self.lx, self.ly, np.pi / 2 - beta)
        self.origami.set_omega(omega)

    def get_gamma(self):
        return self.gamma

    def plot_YZ_zigzag(self, ax: Axes):
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

        plot_zigzag(ax, self.YZ_ys, self.YZ_zs, self.YZ_mid_y, self.YZ_mid_z)

        smooth_us = np.arange(0.5, np.pi - 0.5, 0.01)
        smooth_ys = -np.cos(smooth_us)
        smooth_zs = np.sin(smooth_us)
        ax.plot(smooth_ys, smooth_zs, '--')

    def plot_XY_zigzag(self, ax: Axes):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plot_zigzag(ax, self.XY_xs, self.XY_ys, self.XY_mid_x, self.XY_mid_y)

        smooth_vs = np.arange(0, 2 * np.pi, 0.01)
        smooth_xs = smooth_vs
        smooth_ys = np.sin(smooth_vs)
        ax.plot(smooth_xs, smooth_ys, '--')

    def calc_FFF(self, i, j):
        # The ith point that touches the surface is actually the 2i point in the array
        # This is because of the zigzagging, every 2nd point reflects a point on the surface
        i = i * 2
        j = j * 2

        origami = self.origami
        dots = origami.dots
        dx = dots[:, origami.indexes[i, j + 2]] - dots[:, origami.indexes[i, j]]
        dy = dots[:, origami.indexes[i + 2, j]] - dots[:, origami.indexes[i, j]]

        print(f'dx={dx}, dy={dy}')

        FFF = np.array([[dx.dot(dx), dx.dot(dy)],
                        [dx.dot(dy), dy.dot(dy)]])

        dx /= self.dv
        dy /= self.du

        normalized_FFF = np.array([[dx.dot(dx), dx.dot(dy)],
                                   [dx.dot(dy), dy.dot(dy)]])

        return FFF, normalized_FFF


def plot_smooth_surface():
    us = np.linspace(0.5, np.pi - 0.5, 100)
    vs = np.linspace(0, 2 * np.pi, 100)

    Us, Vs = np.meshgrid(us, vs)

    xs = Vs
    ys = -np.cos(Us) + np.sin(Vs)
    zs = np.sin(Us)
    # xs = np.cos(Us)+Vs
    # ys = 0
    # zs = np.sin(Us)

    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xs, ys, zs, alpha=0.6)

    plotutils.set_3D_labels(ax)

    plt.savefig(os.path.join(FIGURES_PATH, 'smooth_surface.png'))

    plt.show()


def plot_discrete():
    origami = DiscreteOrigami(0.15, 0.4, beta=0.8 * np.pi / 2, omega=0.6 * np.pi)
    print(f'gamma={origami.get_gamma():.2f}')
    print(f'Theta={origami.YZ_zigzag_angle:.2f}')
    print(f'phi={origami.XY_zigzag_angle:.2f}')
    print(f'ls={origami.ly}')
    print(f'cs={origami.lx}')

    fig, axes = plt.subplots(1, 2)
    fig: Figure = fig
    ax: Axes = axes[0]
    origami.plot_YZ_zigzag(ax)
    ax.set_title('YZ zigzag')

    ax: Axes = axes[1]
    origami.plot_XY_zigzag(ax)
    ax.set_title('XY zigzag')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'zigzag_approximation.png'))

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    origami.origami.plot(ax, alpha=0.5)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)
    plotutils.set_3D_labels(ax)

    fig.savefig(os.path.join(FIGURES_PATH, 'folded-origami.png'))

    plt.show()


def compare_forms():
    origami = DiscreteOrigami(0.15, 0.4, beta=0.8 * np.pi / 2, omega=0.6 * np.pi)

    # It has no meaning to see what point we consider,
    # because we center and rotate the folded
    # point = origami.origami.dots[:, origami.origami.indexes[1, 1]]
    # print(f'Point considered: {point}')
    _print_comparison_for_point(origami, 1, 1)
    _print_comparison_for_point(origami, 7, 4)


def _print_comparison_for_point(origami: DiscreteOrigami, i, j):
    print(f'---------Comparing forms for i={i}, j={j}')
    u = 0.5 + i * origami.du
    v = 0 + j * origami.dv
    print(f'u={u:.3f}, v={v:.3f}')

    FFF, normalized_FFF = origami.calc_FFF(i, j)
    np.set_printoptions(precision=3)
    print('--------- FFF - not normalized')
    print(sympyutils.np_matrix_to_latex(FFF))
    print()
    print('--------- FFF')
    print(sympyutils.np_matrix_to_latex(normalized_FFF))
    print()

    exact_FFF = np.array([[1 + np.cos(v) ** 2, np.sin(u) * np.cos(v)],
                          [np.sin(u) * np.cos(v), 1]])

    # print(str(exact_FFF))
    print('exact FFF')
    print(sympyutils.np_matrix_to_latex(exact_FFF))
    print('-----------end of comparison')
    print()
    print()


def main():
    # plot_smooth_surface()
    # plot_discrete()
    compare_forms()


if __name__ == '__main__':
    main()
