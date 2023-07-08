import os

import numpy as np
from matplotlib import pyplot as plt

import origami.plotsandcalcs
from origami import origamimetric
from origami.plotsandcalcs.continuousmetricalternating import create_perturbed_origami
from origami.quadranglearray import plot_flat_quadrangles
from origami.utils import fitter

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM/ContinuousMetric/AlternatingFigures')

sin, cos, tan = np.sin, np.cos, np.tan

cot = lambda x: 1 / tan(x)
csc = lambda x: 1 / sin(x)
sec = lambda x: 1 / cos(x)


def test_omega():
    rows = 10
    cols = 34
    angle = 1.5
    W0 = 2

    F0 = 0.1
    F = lambda x: F0 / 2 * x

    # MM = lambda y: 0.04 * y ** 2
    MM = lambda y: 0.0 * y

    FF = lambda x: F(x * 2)

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    fig, _ = plot_flat_quadrangles(ori.dots)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    omegas = ori.calc_omegas_vs_x()

    omega_func = lambda x: np.arccos(
        ((cos(W0) + 1) * cos(F0 * (2 * x + 1)) + cos(F0) * (cos(W0) - 1) + 2 * cos(W0)) / (
                (cos(W0) + 1) * cos(F0 * (2 * x + 1)) - cos(F0) * (cos(W0) - 1) + 2))

    fig, ax = plt.subplots()
    ax.plot(omegas, '.')
    xs = np.linspace(0, len(omegas), 100)
    ax.plot(xs, omega_func(xs))

    gammas = ori.calc_gammas_vs_y()
    fig, ax = plt.subplots()
    ax.plot(gammas, '.')

    plt.show()

    # plot_interactive(ori)


def test_plot_g():
    rows = 90
    cols = 90
    angle = 1.5
    W0 = 2.5

    F0 = 0.01
    M0 = 0.01
    F = lambda x: F0 / 2 * x

    MM = lambda y: M0 * y ** 2

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    fig, _ = plot_flat_quadrangles(ori.dots)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    fig, ax = plt.subplots()
    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    # plotutils.imshow_with_colorbar(fig, ax, np.diff(g12, axis=1), 'g12')
    ys = g11[4, :]
    ax.plot(ys, '.')
    params = fitter.FitParams(lambda x, a, b, c: a * x ** 2 + b * x + c, np.arange(len(ys)), ys)
    fit = fitter.FuncFit(params)
    fit.plot_fit(ax)
    fit.print_results()

    plt.show()


def main():
    # test_omega()
    test_plot_g()


if __name__ == '__main__':
    main()
