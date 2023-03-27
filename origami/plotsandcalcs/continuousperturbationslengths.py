import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import origami
import origami.angleperturbation
from origami import RFFQMOrigami
from origami.angleperturbation import set_perturbations_by_func_v1
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles, QuadrangleArray
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


def test_cs_approximation():
    F_0 = 0.0001
    F = lambda x: F_0
    G = lambda y: 1 / 100 * (0.03 * np.cos(y / 40) - 0.001 * np.sin(y / 50 + 10) + 0.0001 * (y - 30))
    # G = lambda y: 0 * y
    angle = 1

    rows = 70
    cols = 4
    ls = np.ones(rows - 1) * 1
    cs = np.ones(cols - 1) * 1
    l0 = c0 = d0 = 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    valid, reason = quads.is_valid()
    if not valid:
        print(f"Got a not-valid pattern. Reason: {reason}")

    fig, _ = plot_flat_quadrangles(quads)
    # fig.savefig(os.path.join(FIGURES_PATH, 'perturbed_G_const.png'))

    print(quads.indexes.shape[0])
    cs_vs_ys = np.zeros(quads.indexes.shape[0] // 2)
    for i in range(len(cs_vs_ys)):
        cs_vs_ys[i] = np.linalg.norm(
            quads.dots[:, quads.indexes[2 * i, 2]] - quads.dots[:, quads.indexes[2 * i, 1]])

    fig: Figure = plt.figure()
    ax: Axes = fig.subplots()

    tF = lambda x: F(x) + F(x + 1)
    tG = lambda y: G(y) + G(y + 1)

    ys = np.arange(len(cs_vs_ys))
    print(len(ys))

    tGInt = np.append(0, np.cumsum(tG(np.arange(len(ys) * 2))))
    print(len(tGInt))

    expected_ls = c0 - c0 / (np.tan(angle)) * (tGInt[2 * ys]) + 2 * l0 * tF(1) / np.sin(angle) * ys
    ax.plot(expected_ls, '-')
    ax.plot(cs_vs_ys, '.')
    ax.set_ylabel(r'$ c_{1,y} $')
    ax.set_xlabel('i = y axis')


def test_ls_approximation():
    G_0 = 0.00001
    F = lambda x: 0.03 * np.cos(x / 40) - 0.01 * np.sin(x / 50 + 10) + 0.0001 * (x - 30)
    G = lambda y: G_0
    angle = 1

    rows = 4
    cols = 1200
    ls = np.ones(rows - 1) * 1
    cs = np.ones(cols - 1) * 1
    l0 = c0 = d0 = 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, np.pi - angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    valid, reason = quads.is_valid()
    if not valid:
        print(f"Got a not-valid pattern. Reason: {reason}")

    fig, _ = plot_flat_quadrangles(quads)
    # fig.savefig(os.path.join(FIGURES_PATH, 'perturbed_G_const.png'))

    ls_vs_xs = np.zeros(quads.indexes.shape[1] // 2)
    for j in range(len(ls_vs_xs)):
        ls_vs_xs[j] = np.linalg.norm(
            quads.dots[:, quads.indexes[1, 2 * j]] - quads.dots[:, quads.indexes[0, 2 * j]])

    fig: Figure = plt.figure()
    ax: Axes = fig.subplots()

    tF = lambda x: F(x) + F(x + 1)
    tG = lambda y: G(y) + G(y + 1)

    xs = np.arange(len(ls_vs_xs))
    expected_ls = l0 + l0 / (2 * np.tan(angle)) * (tF(2 * xs) - tF(0)) + (c0 + d0) * tG(1) / np.sin(angle) * xs
    ax.plot(expected_ls, '-')
    ax.plot(ls_vs_xs, '.')
    ax.set_ylabel(r'$ \ell_{x,1} $')
    ax.set_xlabel('j = x axis')

    # fig.savefig(os.path.join(FIGURES_PATH, 'perturbed_G_const-c_i1_vs_y.png'))


def test_omega():
    Ftilde = lambda x: 0.03 * np.cos(x / 40) - 0.01 * np.sin(x / 50 + 10) + 0.0001 * (x - 30)
    G = lambda y: 0  # 0.03 * np.cos(y / 30) + 0.02 * np.sin(y / 50 + 10)
    angle = 1

    F = lambda x: Ftilde(2 * x + 1)

    rows = 20
    cols = 200
    ls = np.ones(rows - 1) * 1
    cs = np.ones(cols - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_func = origami.angleperturbation.create_angles_func(Ftilde, G)
    origami.angleperturbation.set_perturbations_by_func(angles_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    valid, reason = quads.is_valid()
    if not valid:
        print(f"Got a not-valid pattern. Reason: {reason}")

    fig, _ = plot_flat_quadrangles(quads)

    rffqm = RFFQMOrigami.RFFQM(quads)
    # quads = rffqm.set_omega(2, should_center=False)
    quads = rffqm.set_gamma(rffqm.calc_gamma_by_omega(2), should_center=False)
    fig, _ = plot_flat_quadrangles(quads)

    omegas = rffqm.calc_omegas_vs_x()
    print(omegas)

    fig, ax = plt.subplots()
    ax: Axes = ax
    ax.plot(omegas, '.')

    xs = np.linspace(0, len(omegas), 200)
    # C = 1.31      # To fix initial condition
    C = np.tan(np.pi / 2 - 1.3472 / 2) / (np.exp(Ftilde(0) * np.tan(angle)))
    # C = np.tan(np.pi / 2 - 1.447 / 2) / (np.exp(Ftilde(0) * np.tan(angle)))
    # expected = np.pi-2 * np.arctan((C * np.exp(Ftilde(xs * 2) * np.tan(angle))))
    W0 = 2

    expected = 2 * np.arctan(np.tan(W0/2)*(np.exp((F(xs)-F(0)) * np.tan(angle))))
    Lexpected = W0+(-F(0)+F(xs))*np.sin(W0)*np.tan(angle)
    # expected = 2 * np.arctan(np.tan(1.447 / 2) * np.exp((Ftilde(xs * 2) - Ftilde(0)) * np.tan(angle)))
    ax.plot(xs, expected)
    ax.plot(xs, Lexpected)
    ax.set_xlabel('j = x axis')
    ax.set_ylabel(r'$ \omega $')

    compared = np.zeros((len(omegas), 2))
    compared[:, 0] = np.tan(omegas / 2)
    compared[:, 1] = np.pi / 2 - C * np.exp(Ftilde(np.arange(len(omegas)) * 2) * np.tan(angle))
    # print(compared)

    path = os.path.join(origami.BASE_PATH, 'RFFQM/ContinuousMetric/Figures', 'omega_comparison.png')
    # fig.savefig(path)

    # fig, ax = plt.subplots()
    # ax: Axes = ax
    # ax.plot(np.tan((np.pi-omegas)/2), '.')
    #
    # xs = np.linspace(0, len(omegas), 200)
    # C = 1.34  # To fix initial condition
    # expected = C * np.exp(Ftilde(xs * 2) * np.tan(angle))
    # ax.plot(xs, expected)


def test_gamma():
    Ftilde = lambda x: 0#0.0003 * np.cos(x / 40) - 0.001 * np.sin(x / 50 + 10)
    G = lambda y: 0.003 * np.cos(y / 40) + 0.002 * np.sin(y / 50 + 10)
    angle = 1

    F = lambda x: Ftilde(2 * x + 1)

    rows = 80
    cols = 10
    ls = np.ones(rows - 1) * 0.2
    cs = np.ones(cols - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_func = origami.angleperturbation.create_angles_func(Ftilde, G)
    origami.angleperturbation.set_perturbations_by_func(angles_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    valid, reason = quads.is_valid()
    if not valid:
        print(f"Got a not-valid pattern. Reason: {reason}")

    fig, _ = plot_flat_quadrangles(quads)

    rffqm = RFFQMOrigami.RFFQM(quads)
    quads = rffqm.set_gamma(2, should_center=False)
    fig, _ = plot_flat_quadrangles(quads)

    gammas = rffqm.calc_gammas_vs_y()

    fig, ax = plt.subplots()
    ax.plot(gammas, '.')
    ax.set_title("gamma vs y")


def _calc_normal(v1, v2) -> np.ndarray:
    perp = np.cross(v1, v2)
    return perp / np.linalg.norm(perp)


def main():
    # test_special_case_G_const()
    # test_ls_approximation()
    # test_cs_approximation()
    # test_omega()
    test_gamma()
    plt.show()


if __name__ == '__main__':
    main()
