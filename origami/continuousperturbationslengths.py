import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import origami
from origami.continuousperturbations import set_perturbations_by_func_v1
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles

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


def main():
    test_special_case_G_const()
    plt.show()


if __name__ == '__main__':
    main()
