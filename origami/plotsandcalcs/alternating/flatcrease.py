import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import origami
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import create_angles_func_vertical_alternation, set_perturbations_by_func
from origami.origamiplots import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.plotsandcalcs.alternating.utils import create_perturbed_origami, csc, sin
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import plotutils

FIGURES_PATH = os.path.join(
    origami.plotsandcalcs.BASE_PATH,
    'RFFQM', 'ContinuousMetric', 'AlternatingFigures', 'flat')


def plot_simple_flat_examples():
    # F = lambda x: -0.2 + 0.2 * (x >= 6) + 0.2 * (x >= 12)
    rows = 6
    cols = 6
    angle = 1

    L0 = 0.3
    C0 = 0.5
    MM = lambda y: 0.0 * y

    F = lambda x: -0.2 * (x >= 1)
    ori1 = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    F = lambda x: 0.2 * (x >= 1)
    ori2 = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    fig: Figure = plt.figure(figsize=(10, 6))
    ax: Axes3D = fig.add_subplot(121, projection='3d', azim=-90, elev=90)
    ori1.dots.plot(ax)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    ax.dist = 7
    ax.set_title('F<0')

    ax: Axes3D = fig.add_subplot(122, projection='3d', azim=-90, elev=90)
    ori2.dots.plot(ax)
    ax.set_title('F>0')
    # ax.set_zlim(0, 0.01)
    ax.set_axis_off()
    ax.dist = 7
    # ax.autoscale(tight=True)
    plotutils.set_axis_scaled(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'simple-flat-example.png'))

    plt.show()


def debug_lengths_perturbations():
    # F = lambda x: -0.02 * x
    # F = lambda x: -0.2 * np.sin(x / 40)
    F = lambda x: -0.2 * np.sin(x ** 2 / 900)
    # F = lambda x: +0.00001*(x+1)**2

    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

    FF = lambda x: F(x * 2)
    dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)

    ddFF = lambda x: dFF(x + 0.5) - dFF(x - 0.5)
    dddFF = lambda x: ddFF(x + 0.5) - ddFF(x - 0.5)

    angle = 1

    rows = 70
    cols = 100
    L0 = 0.2
    ls = np.ones(rows) * L0
    cs = np.ones(cols) * 1
    ls[1::2] += 0.2 * np.sin((np.arange(len(ls) // 2) - 13) / 5)
    # cs += -0.5+0.1*np.sin(np.arange(len(cs))/20)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom, 'delta+eta')

    print("pert test", pert_func(0, 0))

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    calced_cs, calced_ls = marching.get_cs(), marching.get_ls()

    xs = np.linspace(0, cols / 2, 200)
    cot_angle = 1 / np.tan(angle)
    csc_angle = 1 / np.sin(angle)
    fig, ax = plt.subplots()
    ax.plot(calced_ls[20, ::2], 'x', label='ls')
    ax.plot(xs,
            L0 * (1 + 1 / 4 * cot_angle * (dFF(xs) - dFF(0)) + 1 / (2 * np.sin(angle) ** 2) * (FF(xs) - FF(0)) ** 2),
            '--', label='Expansion 2nd order')

    # fig.savefig(os.path.join(FIGURES_PATH, 'ls-vs-xs.png'))

    fig, ax = plt.subplots()
    x = 20
    ax.plot(calced_cs[::2, 2 * x], 'x', label=f'cs[i,{x}] vs i')
    ls_diff_integral = np.append(0, -np.cumsum(ls[0::2]) + np.cumsum(ls[1::2]))
    # ls_diff_integral = -np.cumsum(calced_ls[0::2, 2*x]) + np.cumsum(calced_ls[1::2,2*x])
    # ls_diff_integral -= ls_diff_integral[0]
    # ls_integral = np.cumsum(calced_ls[0::2, 2*x]) + np.cumsum(calced_ls[1::2,2*x])
    ls_integral = np.append(0, np.cumsum(ls[0::2]) + np.cumsum(ls[1::2]))

    ls_sum1 = np.append(0, np.cumsum(ls[0::2]))
    ms_sum1 = np.append(0, np.cumsum(ls[1::2]))

    ls_sum2 = np.append(0, np.cumsum(calced_ls[0::2, 2 * x]))
    ms_sum2 = np.append(0, np.cumsum(calced_ls[1::2, 2 * x]))

    # ls_integral -= ls_integral[0]
    print(ls_diff_integral)
    print(ls_integral)
    ax.plot(cs[x * 2] + 1 / 2 * csc_angle * (ls_diff_integral * dFF(x))
            , '--', label='approx1')
    ax.plot(cs[x * 2] + 1 / 2 * csc_angle * ls_diff_integral * dFF(x)
            + cot_angle * csc_angle * (1 / 2 * (FF(x) * dFF(x))) * ls_integral
            , '--', label='approx2')
    ax.plot(cs[x * 2] + csc_angle * ls_diff_integral * (1 / 2 * dFF(x) + 1 / 8 * ddFF(x))
            + cot_angle * csc_angle * (1 / 2 * FF(x) * dFF(x)) * ls_integral
            , '--', label='approx3')

    ax.plot(cs[x * 2] + 1 / 2 * csc_angle * ls_diff_integral * dFF(x)
            + cot_angle * csc_angle * (
                    1 / 2 * FF(x) * dFF(x) + 1 / 4 * dFF(x) ** 2 + 1 / 8 * FF(x) * ddFF(x)) * ls_integral
            , '--', label='approx4')
    ax.plot(cs[x * 2] + csc_angle * ls_diff_integral * (
            1 / 2 * dFF(x) + 1 / 8 * ddFF(x) +
            1 / 4 * FF(x) ** 2 * dFF(x) + 1 / 2 * cot_angle ** 2 * FF(x) ** 2 * dFF(x) +
            1 / 48 * dddFF(x))
            + cot_angle * csc_angle * (
                    1 / 2 * FF(x) * dFF(x) + 1 / 4 * dFF(x) ** 2 + 1 / 8 * FF(x) * ddFF(x)) * ls_integral
            , '--', label='approx5')
    exact1 = cs[x * 2] + csc(angle + FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ls_sum1 - \
             csc(angle - FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ms_sum1
    exact2 = cs[x * 2] + csc(angle + FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ls_sum2 - \
             csc(angle - FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ms_sum2
    ax.plot(exact1, label="exact1")
    ax.plot(exact2, '-+', label="exact2")
    ax.legend()

    # fig, ax = plt.subplots()
    # ax.plot(marching.alphas[:, 20])

    plt.show()

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    ori.set_gamma(2.9)

    plot_interactive(ori)


def test_jump_in_F():
    rows = 10
    cols = 40
    angle = np.pi / 2 - 0.1
    W0 = 2.0

    F = lambda x: 0.1 * (x >= 10) + 0.1 * (x >= 20)
    MM = lambda y: 0 * y

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    fig, _ = plot_flat_quadrangles(ori.dots)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    plot_interactive(ori)


def create_MARS_Barreto_using_alternating_angles():
    """
    We just show that MARS-Barreto is a special case of our alternating angles
    """
    angle_base = 0.65 * np.pi
    angle = angle_base

    rows, cols = 8, 8
    L0, C0 = 1, 1

    F = lambda x: np.pi / 2 - angle_base
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, None)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
    ori.dots.rotate_and_center()
    ori.dots.plot(ax)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    ax.dist = 7
    # ax.set_title(r'F=$ \frac{\pi}{2}-\vartheta $')
    fig.savefig(os.path.join(FIGURES_PATH, 'MARS-barreto-by-alternating.svg'))

    plot_interactive(ori)


def main():
    create_MARS_Barreto_using_alternating_angles()


if __name__ == '__main__':
    main()


