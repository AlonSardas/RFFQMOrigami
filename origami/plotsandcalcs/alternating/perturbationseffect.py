"""
We found that if both F,M exist and allowed to change we have non-vanishing Gaussian curvature.
Here we test other cases where they do not exist or remain constant and see its effect on the curvature.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from origami import quadranglearray, origamimetric, origamiplots
from origami.alternatingpert.utils import create_perturbed_origami
from origami.utils import plotutils


def test_F_x():
    # F = lambda x: 0.1 * np.sin(x / 14) - 0.2 * np.tanh((x - 25) / 10)
    F = lambda x: 0.2 * (x > 15) - 0.3 * (x > 28)

    MM = lambda y: 0.0 * y

    L0 = 1
    C0 = 2.2

    rows = 30
    cols = 50
    angle = 1
    omega = 1.3
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    quadranglearray.plot_flat_quadrangles(ori.dots)
    ori.set_gamma(ori.calc_gamma_by_omega(omega))

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
    fig: Figure = plt.figure()
    ax: Axes = fig.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    origamiplots.plot_interactive(ori)


def test_F_M():
    F = lambda x: 0.3

    MM = lambda y: 0.6 * y

    L0 = 1
    C0 = 2.2

    rows = 30
    cols = 50
    angle = 1
    omega = 1.3
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    # quadranglearray.plot_flat_quadrangles(ori.dots)
    ori.set_gamma(ori.calc_gamma_by_omega(omega))

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
    plt.show()

    origamiplots.plot_interactive(ori)


def test_F_M_x():
    # F = lambda x: 0.4 * np.sin(x / 15) - 0.15 * x / 15 + 0.2 * np.tanh((x - 35) / 6)
    F = lambda x: -0.2 * (x > 16) - 0.2 * (x > 28)
    MM = lambda y: 0.6 * y

    L0 = 1
    C0 = 2.2

    rows = 30
    cols = 50
    angle = 1
    omega = -2.9
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    xs = np.linspace(0, cols, 100)
    fig, ax = plt.subplots()
    ax.plot(xs, F(xs))

    # quadranglearray.plot_flat_quadrangles(ori.dots)
    ori.set_gamma(ori.calc_gamma_by_omega(omega))

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    origamiplots.plot_interactive(ori)


def test_F_M_y():
    F = lambda x: 0.3
    MM = lambda y: 1.2 * np.sin(y / 3) + 0.6 * (y/6)**2 + 0.9 * np.tanh((y - 10) / 6)

    L0 = 1
    C0 = 2.2

    rows = 30
    cols = 50
    angle = 1
    omega = -2.9
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    ys = np.linspace(0, rows, 100)
    fig, ax = plt.subplots()
    ax.plot(ys, (MM(ys)-MM(ys-1)))

    # quadranglearray.plot_flat_quadrangles(ori.dots)
    ori.set_gamma(ori.calc_gamma_by_omega(omega))

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    origamiplots.plot_interactive(ori)


def main():
    # test_F_x()
    # test_F_M()
    test_F_M_x()
    # test_F_M_y()


if __name__ == '__main__':
    main()
