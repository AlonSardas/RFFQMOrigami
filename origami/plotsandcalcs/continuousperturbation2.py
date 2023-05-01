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
from origami.angleperturbation import set_perturbations_by_func, create_angles_func_vertical_alternation
from origami.interactiveplot import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import plotutils


def test_continuous_perturbations():
    # F = lambda x: 0.2+np.sin(x / 50) / 20
    # G = lambda xy: (np.sin(xy / 33) + 0.6 * np.cos(xy / 21)) / 190
    F = lambda x: -0.01 * x ** 2
    # F = lambda x: 0.1 * (x%2-0.5)
    # F = lambda x: 0.3
    F1 = lambda x: F(x / 2)
    F2 = lambda x: -F1(x)

    angle = 1

    rows = 10
    cols = 10
    ls = np.ones(rows)
    cs = np.ones(cols) * 1
    # ls[0::2] *= 2

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func(create_angles_func_vertical_alternation(F1, F2), angles_left, angles_bottom, 'delta+eta')

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    W0 = 2.9
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
    print(Ks)

    expected_K = 0.02 * np.tan(W0) ** 2 / (np.cos(angle) * np.sin(angle) * 4)
    print(expected_K)

    plot_interactive(ori)


def plot_sphere_like():
    F = lambda x: -0.4 * np.sin(x / 20)
    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

    FF = lambda x: F(x * 2)
    dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)
    ddFF = lambda x: dFF(x + 0.5) - dFF(x - 0.5)
    dddFF = lambda x: ddFF(x + 0.5) - ddFF(x - 0.5)

    angle = 1

    rows = 30
    cols = 90
    L0 = 1
    ls = np.ones(rows) * L0
    cs = np.ones(cols) * 1
    ls[::2] += 2 * np.sin(np.arange(len(ls) // 2) / 30)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    ori.set_gamma(2.9)

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks')

    plot_interactive(ori)


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
