import os.path
from fractions import Fraction
from typing import Union, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami import miuraori, simplemiuraplots
from origami.utils import plotutils
from origami.zigzagmiuraori import ZigzagMiuraOri

FIGURES_PATH = '../../RFFQM/Figures/zigzag-origami'


def create_zigzag(n, dxs, y, angle) -> np.ndarray:
    dots = np.zeros((2, n))
    dots[0, 1:] = np.cumsum(dxs)

    dys = np.zeros(n - 1)
    dys[0::2] = dxs[0::2] / np.tan(angle)
    dys[1::2] = -dxs[1::2] / np.tan(angle)
    dots[1, 1:] = np.cumsum(dys)
    dots[1, :] += y
    return dots


def create_zigzag_dots(angles: np.ndarray, n, ls: Union[float, Sequence[float]], dxs) -> np.ndarray:
    if hasattr(ls, '__len__'):
        assert len(ls) == len(angles) - 1
    else:
        ls = np.ones(len(angles) - 1) * ls
    ls = np.append(ls, 0)

    if hasattr(dxs, '__len__'):
        assert len(dxs) == n - 1, \
            f'Got {len(dxs)} dxs while there should be {n}-1'
    else:
        dxs = np.ones(n - 1) * dxs

    dots = np.zeros((2, len(angles) * n))

    y = 0
    for i, angle in enumerate(angles):
        dots[:, i * n:(i + 1) * n] = create_zigzag(n, dxs, y, angle)

        y += ls[i]

    return dots


def create_basic_crease1():
    n = 3
    dx = 1
    dy = 2

    rows = 3
    angles = 0.2 * np.ones(rows) * np.pi

    dots = create_zigzag_dots(angles, n, dy, dx)

    return dots, len(angles), n


def create_basic_crease2():
    n = 7
    dx = 1
    dy = 2

    angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi

    dots = create_zigzag_dots(angles, n, dy, dx)

    return dots, len(angles), n


def create_full_cylinder():
    n = 7
    rows = 21
    dx = 1
    dy = 4

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    dots = create_zigzag_dots(angles, n, dy, dx)
    return dots, len(angles), n


def create_cylinder_changing_dxs():
    n = 10
    rows = 21
    dxs = np.array([1, 2, 1, 3, 1, 4, 1, 5, 1])
    dy = 20

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    dots = create_zigzag_dots(angles, n, dy, dxs)
    return dots, len(angles), n


def create_spiral():
    cols = 5
    rows = 40
    dx = 1
    ls = 2 + np.arange(rows - 1) * 0.15

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    dots = create_zigzag_dots(angles, cols, ls, dx)
    return dots, rows, cols


def create_changing_cs_example():
    rows = 10
    # dxs = np.array([4, 7, 4, 8, 4, 5, 4, 2, 7, 5, 4, 2, 4, 3, 4, 4, 4, 5, 4]) * 0.6
    dxs = np.array([1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7]) * 0.6
    dxs = np.append(dxs, dxs[::-1])
    cols = len(dxs) + 1
    ls = 20

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.3 * np.pi
    # angles[::2] += 0.1 * np.pi
    angles[::2] += 0.4 * np.pi
    dots = create_zigzag_dots(angles, cols, ls, dxs)
    return dots, rows, cols


def create_spiral_changing_cs():
    rows = 40
    dxs = np.array([1, 0.5] * 30) * 0.1
    cols = len(dxs) + 1
    ls = 4 + np.arange(rows - 1) * 0.15

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    dots = create_zigzag_dots(angles, cols, ls, dxs)
    return dots, rows, cols


def plot():
    # logutils.enable_logger()

    # dots, rows, cols = create_full_cylinder()
    # dots, rows, cols = create_spiral()
    # dots, rows, cols = create_changing_cs_example()
    dots, rows, cols = create_spiral_changing_cs()
    # dots, rows, cols = create_cylinder_changing_dxs()
    # dots, rows, cols = create_basic_crease2()
    origami = ZigzagMiuraOri(dots, rows, cols)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    origami.plot(ax)

    origami.set_omega(1)
    valid, reason = miuraori.is_valid(origami.initial_dots, origami.dots, origami.indexes)
    if not valid:
        # raise RuntimeError(f'Not a valid folded configuration. Reason: {reason}')
        print(f'Not a valid folded configuration. Reason: {reason}')

    plot_interactive(origami)


def plot_interactive(origami):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    origami.plot(ax)

    # We need to assign the return value to variable slider for the slider object
    # stay alive and keep functioning
    # noinspection PyUnusedLocal
    slider = simplemiuraplots.add_slider(ax, origami, should_plot_normals=False)

    plt.show()


def plot_simple_example():
    dots, rows, cols = create_basic_crease2()
    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, _ = plot_flat_configuration(origami)
    fig.savefig(os.path.join(FIGURES_PATH, 'simple-example-flat.png'))

    origami.set_omega(0.4)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=157, elev=32)
    origami.plot(ax)
    ax.set_zlim(-2, 2)

    fig.savefig(os.path.join(FIGURES_PATH, 'simple-example-folded.png'))

    plt.show()


def plot_full_cylinder():
    dots, rows, cols = create_full_cylinder()
    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, _ = plot_flat_configuration(origami)
    fig.savefig(os.path.join(FIGURES_PATH, 'cylinder-flat.png'))

    origami.set_omega(1.85)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-30, elev=21)
    origami.plot(ax)
    plotutils.set_3D_labels(ax)

    fig.savefig(os.path.join(FIGURES_PATH, 'cylinder-folded.png'))

    plt.show()


def plot_spiral():
    dots, rows, cols = create_spiral()
    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, _ = plot_flat_configuration(origami)
    fig.savefig(os.path.join(FIGURES_PATH, 'spiral-flat.png'))

    origami.set_omega(2.2)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-20, elev=15)
    origami.plot(ax)
    ax.set_xlim(-3, 3)
    fig.savefig(os.path.join(FIGURES_PATH, 'spiral-folded.png'))

    plt.show()


def plot_flat_configuration(origami):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=90, elev=-100)
    origami.set_omega(0)
    origami.plot(ax)
    ax.set_zlim(-1, 1)
    return fig, ax


def plot_theta_vs_alpha():
    fig, ax = plt.subplots()
    xs = np.linspace(0, np.pi, 200)
    s_a = np.sin(xs)
    s_a_2 = s_a ** 2
    omega = 2
    c_o = np.cos(omega)
    ys = 1 / 2 * np.arccos((2 - 3 * s_a_2 + s_a_2 * c_o) / (-2 + s_a_2 + s_a_2 * c_o))

    ax.plot(xs, ys)

    plotutils.set_pi_ticks(ax, 'x')
    plotutils.set_pi_ticks(ax, 'y', (0, Fraction(1, 2)))

    ax.set_xlabel(r'$ \alpha $')
    ax.set_ylabel(r'$ \theta\left(\omega=2\right) $ vs  $ \alpha $')

    fig.savefig(os.path.join(FIGURES_PATH, 'theta_vs_alpha.png'))

    plt.show()


def plot_unit_cell():
    dxs = np.array([1, 1.5])
    cols = 3
    rows = 2
    ls = 1.1
    angles = [0.6 * np.pi, 0.7 * np.pi]
    dots = create_zigzag_dots(angles, cols, ls, dxs)

    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, ax = plot_flat_configuration(origami)
    edge_points = origami.dots[:, [origami.indexes[0, 0],
                                   origami.indexes[0, -1],
                                   origami.indexes[-1, 0],
                                   origami.indexes[-1, -1]]]
    ax.scatter3D(edge_points[0, :], edge_points[1, :], edge_points[2, :], color='r', s=220)
    fig.savefig(os.path.join(FIGURES_PATH, 'zigzag-FFF-unit.png'))
    plt.show()


def main():
    # plot()
    # plot_spiral()
    # plot_full_cylinder()
    # plot_simple_example()
    # plot_theta_vs_alpha()
    plot_unit_cell()


if __name__ == '__main__':
    main()
