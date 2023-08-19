"""
We found an expression that describes the curvature in the linear regime of the perturbations
Here we want to verify numerically these results.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from origami import origamimetric
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import set_perturbations_by_func_v1
from origami.origamiplots import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import plotutils

sin, cos, tan = np.sin, np.cos, np.tan


def test_F():
    """
    We expect to have constant curvature when F grows cubically
    here we examine that
    """
    # F = lambda x: x ** 3 / 1000000
    F = lambda x: 0.04
    G = lambda y: 0

    # angle = np.pi-0.2
    # angle = 1.5
    angle = 0.5

    rows = 30
    cols = 30
    ls = np.ones(rows)
    cs = np.ones(cols) * 0.5

    # cs[::2] = 3+np.arange(len(cs)//2)*0.1
    # ls[::2]=2-np.arange(len(ls)//2)**2/500
    # ls[::2]=2-np.arange(len(ls)//2)**2/500
    ls[::2] += np.arange(len(ls) // 2) * 1.3
    ls /= 100
    cs /= 10
    # ls += np.arange(len(ls)//2)*0.3

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    ori.set_gamma(1, should_center=False)
    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    print(f"max curvature: {np.max(np.abs(Ks))}")
    print(Ks)

    plot_interactive(ori)


def test_metric():
    """
    We compare the expression for the metric in the *flat* paper to the results we found
    """
    F = lambda x: x ** 2 / 1000000 - x / 50000
    H = lambda y: 0.00000001 * y - 0.000001 * y ** 2
    # H = lambda y:0

    G = lambda y: H(y + 1) - H(y)
    Ftilde = lambda x: F(x / 2)
    Gtilde = lambda y: G(y / 2)

    dF = lambda x: F(x + 1) - F(x)
    ddF = lambda x: dF(x + 1) - dF(x)
    dH = lambda y: H(y + 1) - H(y)
    ddH = lambda y: dH(y + 1) - dH(y)

    # print(dH(0))

    angle = 1
    C0 = 5
    L0 = 1

    rows, cols = 150, 150
    # rows, cols = 9, 9
    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0
    # cs[81]=1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(Ftilde, Gtilde, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

    # print([ddF(x) for x in range(100)])

    # cs_func = lambda u, v: C0 + 2 / sin(angle) * (2 * F(u) + 1 / 2 * dF(u)) * L0 * ys - \
    #                        4 / tan(angle) * C0 * (H(v) + 1 / 2 * dH(v))
    cs_func = lambda u, v: C0 + 1 / (2 * sin(angle)) * (L0 + L0) * (4 * F(u) + dF(u)) * v + \
                           + 2 * C0 / tan(angle) * (2 * (H(v) - H(0)) + (dH(v) - dH(0)))
    # ds_func = lambda u, v: C0 + 2 / sin(angle) * (2 * F(u) + dF(u) + 1 / 2 * dF(u) + 1 / 4 * ddF(u)) * L0 * ys + \
    #                        4 / tan(angle) * C0 * (H(v) + 1 / 2 * dH(v))
    # ds_func = lambda u, v: C0 + 2 / sin(angle) * (F(u) + 1 / 2 * dF(u) + F(u) + dF(u)) * L0 * ys + \
    #                        4 / tan(angle) * C0 * (H(v) + 1 / 2 * dH(v))
    ds_func = lambda u, v: C0 + 1 / (2 * sin(angle)) * (L0 + L0) * (4 * F(u) + 3 * dF(u)) * v + \
                           - 2 * C0 / tan(angle) * (2 * (H(v) - H(0)) + (dH(v) - dH(0)))

    g11_func = lambda u, v: 4 * sin(angle) * C0 * (sin(angle) * C0 - cos(angle) * C0 * (2 * F(u) + dF(u)) +
                                                   +4 * v * L0 * (2 * F(u) + dF(u)))
    """g11_func = lambda x, y: \
        - 2 * C0 * sin(angle) * (2 * C0 * cos(angle) * (dF(x) + 2 * F(x)) -
                                 2 * C0 * sin(angle) +
                                 L0 * y * (ddF(x) - ddF(0) - 16 * F(x)))
    """

    g22_func = lambda u, v: 4 * L0 * ((1 - 2 / tan(angle) * F(u)) * L0 + 4 * u / sin(angle) * C0 * (2 * dH(v) + ddH(v)))
    # g12_func = lambda x, y: -2 * C0 * L0 * sin(angle) * (
    #         3 * dF(x) + 4 * F(x) + ddH(y) + 4 * dH(y) - 2 * H(y) / (tan(angle) ** 2))
    g12_func = lambda u, v: 1 / 2 / sin(angle) * L0 * (-8 * v * cos(angle) * L0 * dF(u) +
                                                       -C0 * (8 * F(u) - 16 * H(v) + 3 * dF(u) + ddH(v)) +
                                                       +cos(2 * angle) * C0 * (
                                                               8 * F(u) + 16 * H(v) + 3 * dF(u) + 16 * dH(v) + ddH(v)))
    # print(g11_func(0, 0))

    cells_shape = g11.shape
    xs, ys = np.meshgrid(np.arange(cells_shape[1]), np.arange(cells_shape[0]))
    # print(g11)
    # print(g11_func(xs, ys))
    expected_g11 = g11_func(xs, ys)
    expected_g22 = g22_func(xs, ys)
    expected_g12 = g12_func(xs, ys)

    cs_xy = cs_func(xs, ys)
    ds_xy = ds_func(xs, ys)
    # print(expected_g11[1, 1])
    expected_g11 = cs_xy ** 2 + ds_xy ** 2 - 2 * cs_xy * ds_xy * cos(2 * angle)
    # print(expected_g11[1, 1])

    # print(np.abs())

    fig, axes = plt.subplots(2, 2)
    _imshow_with_colorbar(fig, axes[0, 0], g11 - expected_g11, "g11 comparison")
    _imshow_with_colorbar(fig, axes[0, 1], g22 - expected_g22, "g22 comparison")
    _imshow_with_colorbar(fig, axes[1, 0], g12 - expected_g12, "g12 comparison")
    _imshow_with_colorbar(fig, axes[1, 1], Ks, "K")

    # Compare the lengths:
    y = 2
    ls_vs_xs = np.zeros(quads.indexes.shape[1] // 2)
    ms_vs_xs = np.zeros(quads.indexes.shape[1] // 2)
    for j in range(len(ls_vs_xs)):
        ls_vs_xs[j] = np.linalg.norm(
            quads.dots[:, quads.indexes[2 * y + 1, 2 * j]] - quads.dots[:, quads.indexes[2 * y, 2 * j]])
        ms_vs_xs[j] = np.linalg.norm(
            quads.dots[:, quads.indexes[2 * y + 2, 2 * j]] - quads.dots[:, quads.indexes[2 * y + 1, 2 * j]])

    xs = np.arange(len(ls_vs_xs))
    # angle = np.pi-angle
    expected_ls = L0 - L0 / (tan(angle)) * (F(xs) - F(0)) + \
                  C0 / sin(angle) * (4 * dH(y) + ddH(y)) * xs
    # expected_ls = L0 - L0 / (tan(angle)) * (F(xs) - F(0) + 1 / 4 * (dF(xs) - dF(0))) + \
    #               C0 / sin(angle) * (4 * dH(y) + ddH(y)) * xs
    expected_ms = L0 - L0 / (tan(angle)) * (F(xs) - F(0)) + \
                  C0 / sin(angle) * (4 * dH(y) + 3 * ddH(y)) * xs

    print(f"max error in ls: {np.max(np.abs(expected_ls - ls_vs_xs))}")

    # expected_ls = L0 - L0 / (2 * tan(angle)) * (F(xs) + F(xs+1/2)) + \
    #               2 * C0 * (2 * dH(y) + 1 / 2 * ddH(y)) / sin(angle) * xs
    # expected_ls = L0 - L0 / (2 * tan(angle)) * (2 * F(xs) + 1 / 2 * dF(xs)) + \
    #               4 * C0 * (dH(y+2) + dH(y+3)) / sin(angle) * xs

    fig, ax = plt.subplots()
    ax.plot(ls_vs_xs, '+', label="l")
    ax.plot(ms_vs_xs, 'x', label="m")
    ax.plot(expected_ls, label="expected l")
    ax.plot(expected_ms, label="expected m")
    ax.legend()

    cs_vs_ys = np.zeros(quads.indexes.shape[0] // 2)
    ds_vs_ys = np.zeros(quads.indexes.shape[0] // 2)
    x = 40
    for i in range(len(cs_vs_ys)):
        cs_vs_ys[i] = np.linalg.norm(
            quads.dots[:, quads.indexes[2 * i, 2 * x + 1]] - quads.dots[:, quads.indexes[2 * i, 2 * x]])
        ds_vs_ys[i] = np.linalg.norm(
            quads.dots[:, quads.indexes[2 * i, 2 * x + 2]] - quads.dots[:, quads.indexes[2 * i, 2 * x + 1]])

    fig: Figure = plt.figure()
    ax: Axes = fig.subplots()
    ys = np.arange(len(cs_vs_ys))

    expected_cs = cs_func(x, ys)

    expected_ds = ds_func(x, ys)
    print("max error in c", np.max(np.abs(cs_vs_ys - expected_cs)))
    ax.set_title(f'c,d vs y, x={x}')
    ax.plot(cs_vs_ys, '+', label="c")
    ax.plot(ds_vs_ys, 'x', label="d")
    ax.plot(expected_cs, '--', label="expected c")
    ax.plot(expected_ds, '--', label="expected d")
    ax.legend()

    plt.show()
    print()


def _imshow_with_colorbar(fig: Figure, ax: Axes, data: np.ndarray, ax_title):
    plotutils.imshow_with_colorbar(fig, ax, data, ax_title)


def main():
    test_F()
    # test_metric()


if __name__ == '__main__':
    main()
