import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import origami
from origami import continuousperturbations, RFFQMOrigami
from origami.continuousperturbations import set_perturbations_by_func_v1
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import linalgutils

FIGURES_PATH = os.path.join(origami.BASE_PATH, 'RFFQM/Compatibility/Figures/continuous-lengths')


def test_special_case_F_const():
    # Test the case where F is constant and G is 0
    F = lambda x: 0.01
    G = lambda y: 0
    # F = lambda x: 0
    # G = lambda y: 0.01
    angle = 1

    rows = 10
    cols = 30
    ls = np.ones(rows - 1) * 1
    cs = np.ones(cols - 1) * 0.1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)


def test_special_case_G_const():
    G_0 = 0.03
    F = lambda x: 0
    G = lambda y: G_0
    angle = 1

    rows = 50
    cols = 14
    ls = np.ones(rows - 1) * 1
    cs = np.ones(cols - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    valid, reason = quads.is_valid()
    if not valid:
        print(f"Got a not-valid pattern. Reason: {reason}")

    fig, _ = plot_flat_quadrangles(quads)
    fig.savefig(os.path.join(FIGURES_PATH, 'perturbed_G_const.png'))

    c_vs_ys = np.zeros(quads.indexes.shape[0])
    for i in range(len(c_vs_ys)):
        c_vs_ys[i] = np.linalg.norm(
            quads.dots[:, quads.indexes[i, 2]] - quads.dots[:, quads.indexes[i, 1]])

    fig: Figure = plt.figure()
    ax: Axes = fig.subplots()

    ys = np.arange(len(c_vs_ys))
    expected_cs = np.exp(-ys * 2 * G_0 / np.tan(angle))
    ax.plot(expected_cs, '-')
    ax.plot(c_vs_ys, '.')
    ax.set_ylabel('$ c_{i1} $')
    ax.set_xlabel('i = y axis')

    fig.savefig(os.path.join(FIGURES_PATH, 'perturbed_G_const-c_i1_vs_y.png'))

    for i in range(len(c_vs_ys)):
        c_vs_ys[i] = np.linalg.norm(
            quads.dots[:, quads.indexes[i, 1]] - quads.dots[:, quads.indexes[i, 0]])

    fig: Figure = plt.figure()
    ax: Axes = fig.subplots()

    ys = np.arange(len(c_vs_ys))
    expected_cs = np.exp(ys * 2 * G_0 / np.tan(angle))
    ax.plot(expected_cs, '-')
    ax.plot(c_vs_ys, '.')
    ax.set_ylabel('$ c_{i0} $')
    ax.set_xlabel('i = y axis')

    # fig.savefig(os.path.join(FIGURES_PATH, 'perturbed_G_const-c_i0_vs_y.png'))

    print(c_vs_ys)


def test_omega():
    F = lambda x: 0.03 * np.cos(x / 40) - 0.01 * np.sin(x / 50 + 10) + 0.0001 * (x - 30)
    G = lambda y: 0  # 0.03 * np.cos(y / 30) + 0.02 * np.sin(y / 50 + 10)
    angle = 1

    rows = 4
    cols = 200
    ls = np.ones(rows - 1) * 1
    cs = np.ones(cols - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_func = continuousperturbations.create_angles_func(F, G)
    continuousperturbations.set_perturbations_by_func(angles_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    valid, reason = quads.is_valid()
    if not valid:
        print(f"Got a not-valid pattern. Reason: {reason}")

    fig, _ = plot_flat_quadrangles(quads)

    rffqm = RFFQMOrigami.RFFQM(quads)
    quads = rffqm.set_omega(-2, should_center=False)
    fig, _ = plot_flat_quadrangles(quads)

    omegas = np.zeros(cols // 2 - 1)
    for j in range(len(omegas)):
        jj = 2 * (j + 1)
        base_dot = quads.dots[:, quads.indexes[0, jj]]
        v0 = quads.dots[:, quads.indexes[1, jj]] - base_dot
        v1 = quads.dots[:, quads.indexes[0, jj - 1]] - base_dot
        v2 = quads.dots[:, quads.indexes[0, jj + 1]] - base_dot
        n1 = np.cross(v1, v0)
        n2 = np.cross(v0, v2)
        omegas[j] = linalgutils.calc_angle(n1, n2)
        print(omegas[j])

    fig, ax = plt.subplots()
    ax: Axes = ax
    ax.plot(omegas, '.')

    xs = np.linspace(0, len(omegas), 200)
    # C = 1.31      # To fix initial condition
    C = np.tan(np.pi / 2 - 1.3472 / 2) / (np.exp(F(0) * np.tan(angle)))
    expected = np.pi - 2 * np.arctan(C * np.exp(F(xs * 2) * np.tan(angle)))
    ax.plot(xs, expected)
    ax.set_xlabel('j = x axis')
    ax.set_ylabel(r'$ \omega $')

    path = os.path.join(origami.BASE_PATH, 'RFFQM/ContinuousMetric/Figures', 'omega_comparison.png')
    fig.savefig(path)

    # fig, ax = plt.subplots()
    # ax: Axes = ax
    # ax.plot(np.tan((np.pi-omegas)/2), '.')
    #
    # xs = np.linspace(0, len(omegas), 200)
    # C = 1.34  # To fix initial condition
    # expected = C * np.exp(F(xs * 2) * np.tan(angle))
    # ax.plot(xs, expected)


def _calc_normal(v1, v2) -> np.ndarray:
    perp = np.cross(v1, v2)
    return perp / np.linalg.norm(perp)


def main():
    # test_special_case_G_const()
    test_omega()
    plt.show()


if __name__ == '__main__':
    main()
