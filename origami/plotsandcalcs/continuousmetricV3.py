"""
We found an expression that describes the curvature in the linear regime of the perturbations
Here we want to verify numerically these results.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami import origamimetric
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import set_perturbations_by_func_v1
from origami.interactiveplot import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import plotutils, linalgutils

sin, cos, tan = np.sin, np.cos, np.tan


def test_unit_cell():
    FF = lambda x: 0
    # G = lambda y: 0.3 * (y == 2)
    G = lambda y: 0

    # angle = np.pi-0.2
    # angle = 1.5
    angle = 1

    rows = 6
    cols = 6
    ls = np.ones(rows)
    cs = np.ones(cols) * 0.5

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(FF, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)
    # ori.set_omega(2)
    ori.set_gamma(ori.calc_gamma_by_omega(2))

    j = 0
    jj = 2 * (j + 1)
    quads = ori.dots
    base_dot = quads.dots[:, quads.indexes[0, jj]]
    v0 = quads.dots[:, quads.indexes[1, jj]] - base_dot
    v1 = quads.dots[:, quads.indexes[0, jj - 1]] - base_dot
    v2 = quads.dots[:, quads.indexes[0, jj + 1]] - base_dot
    n1 = np.cross(v1, v0)
    n2 = np.cross(v0, v2)
    omega = linalgutils.calc_angle(n1, n2)
    print("omega", omega)

    base_dot = quads.dots[:, quads.indexes[1, jj]]
    v0 = quads.dots[:, quads.indexes[1, jj]] - base_dot
    v1 = quads.dots[:, quads.indexes[0, jj - 1]] - base_dot
    v2 = quads.dots[:, quads.indexes[0, jj + 1]] - base_dot
    n1 = np.cross(v1, v0)
    n2 = np.cross(v0, v2)
    omega = linalgutils.calc_angle(n1, n2)
    print("omega", omega)

    fig, _ = plot_flat_quadrangles(quads)
    plot_interactive(ori)


def _imshow_with_colorbar(fig: Figure, ax: Axes, data: np.ndarray, ax_title):
    im = ax.imshow(data)
    plotutils.create_colorbar(fig, ax, im)
    ax.set_title(ax_title)
    ax.invert_yaxis()


def reproduce_sphere():
    """
    We try to build the sphere again, using the results we got for the lengths
    """
    Ms = np.array([1.01, 1.14075, 1.20849, 1.26581, 1.31854, 1.36899, 1.4184, 1.46759, \
                   1.51714, 1.56753, 1.61921, 1.67259, 1.72808, 1.78615, 1.84729, \
                   1.91206, 1.98112, 2.05524, 2.13539, 2.22272, 2.31871, 2.42526, \
                   2.54488, 2.68092, 2.83809, 3.02316, 3.24638, 3.52411, 3.88458, \
                   4.38191, 5.13774])
    # Ms = Ms[:10]
    W0 = 1.0
    angle = 1.3
    C0 = 0.001
    L0 = 1
    expectedK = 0.5
    F0 = 0.02

    F = lambda x: F0
    G = lambda y: 0

    rows = len(Ms) * 2
    cols = 90
    ls = np.ones(rows)
    cs = np.ones(cols) * C0

    ls[::2] = L0
    ls[1::2] = Ms

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    # W0=0.2
    print("gamma", ori.calc_gamma_by_omega(W0))
    # gamma = 2.75
    ori.set_gamma(ori.calc_gamma_by_omega(W0), should_center=True)
    # ori.set_omega(0.1W0 should_center=False)
    print("calculated omega", ori.calc_omegas_vs_x()[0])
    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    print(f"max curvature: {np.max(np.abs(Ks))}")
    print(f"mean curvature: {np.mean(np.abs(Ks))}")
    # print(Ks)

    print('g11=', g11[:, 3])
    # print('cumsum', np.cumsum(Ms))
    ys = np.arange(len(g11[:, 6]))
    MsInt = np.cumsum(Ms) - Ms[0]
    expected_noL_g11 = 3.0464 * (0.000963558 + 0.04 * (ys + MsInt[:-1])) ** 2
    expected_Lg11 = 2.82841e-6 + 0.000237467 * ys + 0.000237467 * MsInt[:-1]
    fig, ax = plt.subplots()
    ax.set_title("Comparing g11")
    ax.plot(g11[:, 3], '.', label="numerical")
    ax.plot(expected_Lg11, '.', label="linearized g11")
    ax.plot(expected_noL_g11, '.', label="not-linearized g11")
    ax.legend()

    print("g22=", g22[:, 5])
    expected_g22 = 1 - 1.02883 * Ms[:-1] + (Ms[:-1]) ** 2
    print("expected g22", expected_g22)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ori.dots.plot(ax)
    plotutils.set_axis_scaled(ax)

    fig, ax = plt.subplots()
    ax.plot(Ks[5:-5, 1].flat, '.')

    plot_interactive(ori)

    plt.show()


# plot_interactive(ori)


def main():
    # test_unit_cell()
    # test_metric()
    reproduce_sphere()


if __name__ == '__main__':
    main()
