"""
We already examined a case where the angle perturbations are given by:
delta   = F(x)+G(y)
eta     = F(x)-G(y)
where F,G are general functions

Here we assume that the slowly varying fields of the angle is applied to alternating angles
when moving upwards. Here we check alternative versions for the perturbation
"""
import matplotlib.pyplot as plt
import numpy as np

from origami import angleperturbation, origamimetric
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import set_perturbations_by_func
from origami.interactiveplot import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import plotutils


def test_continuous_perturbations():
    # F = lambda x: 0.2+np.sin(x / 50) / 20
    # G = lambda xy: (np.sin(xy / 33) + 0.6 * np.cos(xy / 21)) / 190
    F = lambda x: -0.001 * x**2
    F1 = lambda x: F(x / 2)
    # F1 = lambda x: 0.1
    F2 = lambda x: -F1(x)

    angle = 1

    rows = 40
    cols = 30
    ls = np.ones(rows)
    cs = np.ones(cols) * 1
    # ls[0::2] *= 2

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func(create_angles_func(F1, F2), angles_left, angles_bottom, 'delta+eta')

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    calced_cs, calced_ls = marching.get_cs(), marching.get_ls()
    c_vs_ys = calced_cs[::2, 2]
    ys = np.linspace(0, len(c_vs_ys), 100)
    expected_cs = 1 - 1 / 2 / np.sin(angle) * (-ls[0] + ls[1]) * 0.04 * ys / 2

    fig, ax = plt.subplots()
    ax.plot(c_vs_ys, '.')
    ax.plot(ys, expected_cs)

    y = 4
    ls_vs_xs = calced_ls[2*y, ::2]
    fig, ax = plt.subplots()
    ax.plot(ls_vs_xs, '.')

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    ori.set_gamma(2.9)

    fig, ax = plt.subplots()
    ax.plot(ori.calc_omegas_vs_x(), '.')
    ax.set_title("omega vs x")

    plot_interactive(ori)


def plot_sphere_like():
    # F = lambda x: 0.2+np.sin(x / 50) / 20
    # G = lambda xy: (np.sin(xy / 33) + 0.6 * np.cos(xy / 21)) / 190

    F = lambda x: -0.4 * np.sin(x / 20)
    # F = lambda x: 0.2
    G = lambda xy: 0
    # G = lambda xy: 0.1*(xy%2-0.5)
    # G = lambda xy: 0.003 * np.cos(xy/10)

    angle = 1

    rows = 30
    cols = 90
    ls = np.ones(rows)
    cs = np.ones(cols) * 1
    ls[::2] += 2 * np.sin(np.arange(len(ls) // 2) / 30)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func(create_angles_func_bad2(F, G), angles_left, angles_bottom, 'delta+eta', 'bottom')

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    calced_cs, calced_ls = marching.get_cs(), marching.get_ls()
    c_vs_ys = calced_cs[::2, 4]
    fig, ax = plt.subplots()
    ax.plot(c_vs_ys, '.')
    ax.plot(calced_cs[::2, 34], '.')
    ax.set_title("cs")

    fig, ax = plt.subplots()
    ax.plot(calced_cs[20, ::2], '.')
    ax.set_title("cs vs x")

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    ori.set_gamma(2.9)

    fig, ax = plt.subplots()
    ax.plot(ori.calc_omegas_vs_x())
    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks')

    plot_interactive(ori)


def create_angles_func(F1, F2) -> angleperturbation.AnglesFuncType:
    def func(xs, ys):
        deltas = F1(xs) * (1 - ys % 2) + F2(xs) * (ys % 2)
        etas = F1(xs) * (ys % 2) + F2(xs) * (1 - ys % 2)

        return deltas, etas

    return func


def create_angles_func_bad2(F, G) -> angleperturbation.AnglesFuncType:
    def func(xs, ys):
        deltas = F(xs)
        etas = -F(xs)
        return deltas, etas

    return func


def create_angles_func_bad(F, G) -> angleperturbation.AnglesFuncType:
    def func(xs, ys):
        deltas = -G(xs + ys)
        etas = -G(xs + ys) + F(xs)
        return deltas, etas

    return func


def main():
    # test_continuous_perturbations()
    plot_sphere_like()


if __name__ == '__main__':
    main()
