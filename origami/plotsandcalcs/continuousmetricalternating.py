import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import origami.plotsandcalcs
from origami import origamimetric
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import create_angles_func_vertical_alternation, set_perturbations_by_func
from origami.interactiveplot import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils.plotutils import imshow_with_colorbar

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM/ContinuousMetric/AlternatingFigures')

sin, cos, tan = np.sin, np.cos, np.tan

cot = lambda x: 1 / tan(x)
csc = lambda x: 1 / sin(x)
sec = lambda x: 1 / cos(x)


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


def test_angles():
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
    L0 = 1
    C0 = 1
    # ls[1::2] += 0.2 * np.sin((np.arange(len(ls) // 2) - 13) / 5)
    # cs += -0.5+0.1*np.sin(np.arange(len(cs))/20)
    MM = lambda y: 0
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    fig, _ = plot_flat_quadrangles(ori.dots)

    G0 = 2
    ori.set_gamma(G0)
    gammas = ori.calc_gammas_vs_y()
    fig, ax = plt.subplots()
    ax.plot(gammas, '.')

    W0 = 2
    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    omegas = ori.calc_omegas_vs_x()

    fig, ax = plt.subplots()
    ax.plot(omegas, '.')
    xs = np.linspace(0, len(omegas), 200)
    ax.plot(xs, W0 + 1 / 2 * (FF(xs) ** 2 - FF(0) ** 2) * sin(W0), '--')

    plot_interactive(ori)


def test_metric():
    # F = lambda x: -0.02 * x
    # F = lambda x: -0.2 * np.sin(x / 40)
    F = lambda x: -0.2 * np.sin(x ** 2 / 900)
    # F = lambda x: +0.00001*(x+1)**2

    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

    FF = lambda x: F(x * 2)
    # dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)
    dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)

    ddFF = lambda x: dFF(x + 0.5) - dFF(x - 0.5)
    dddFF = lambda x: ddFF(x + 0.5) - ddFF(x - 0.5)

    angle = 1

    rows = 60
    cols = 100
    L0 = 0.2
    C0 = 1
    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0

    MM = lambda y: -np.cos((y - 13) / 5)
    dMM = lambda y: MM(y + +0.5) - MM(y - 0.5)
    ddMM = lambda y: dMM(y + +0.5) - dMM(y - 0.5)
    dddMM = lambda y: ddMM(y + +0.5) - ddMM(y - 0.5)
    ms_func2 = lambda y: L0 + 0.2 * np.sin((y - 13) / 5)
    dms_func2 = lambda y: 0.2 / 5 * np.cos((y - 13) / 5)
    ms_func = lambda y: L0 + dMM(y)
    dms_func = ddMM

    ys = np.arange(len(ls) // 2)
    ls[1::2] = L0 + dMM(ys) + 1 / 2 * ddMM(ys)
    # print(dMM(ys))
    # print(0.2 * np.sin((ys - 13) / 5))
    # print(L0 + dMM(ys) + 1 / 2 * ddMM(ys))
    # print(ms_func2(ys))

    # cs += -0.5+0.1*np.sin(np.arange(len(cs))/20)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    W0 = 2
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    ls_diff_integral = np.append(0, -np.cumsum(ls[0::2]) + np.cumsum(ls[1::2]))
    ls_integral = np.append(0, np.cumsum(ls[0::2]) + np.cumsum(ls[1::2]))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

    fig, axes = plt.subplots(2, 2)
    imshow_with_colorbar(fig, axes[0, 0], g11, "g11")
    imshow_with_colorbar(fig, axes[0, 1], g22, "g22")
    imshow_with_colorbar(fig, axes[1, 0], g12, "g12")
    imshow_with_colorbar(fig, axes[1, 1], Ks, "K")

    expected_g11_func_O1 = lambda x, y: 4 * cos(W0 / 2) ** 2 * sin(angle) ** 2 * C0 ** 2
    expected_g11_func_O2 = lambda x, y: expected_g11_func_O1(x, y) + sin(angle) ** 2 * C0 * (
            (4 * csc(angle) ** 2 * FF(x) ** 2 * sin(W0 / 2) ** 2 + (-FF(x) ** 2) * sin(W0) ** 2) * C0 +
            2 * (1 + cos(W0)) * csc(angle) * (ls_diff_integral[y]) * dFF(x))
    # print(expected_g11_func_O1(10, 10))

    expected_g12_func_O1 = lambda x, y: -2 * FF(x) * sin(angle) * C0 * (L0 - ms_func(y))
    expected_g12_func_O2 = lambda x, y: expected_g12_func_O1(x, y) + sin(angle) * sin(W0 / 2) ** 2 * (
            C0 * dFF(x) * (-L0 + ms_func(y)))

    y0 = 14
    gs = g11[y0, :]
    xs = np.linspace(0, len(gs), 200)
    expected_gsO1 = np.ones(len(xs)) * expected_g11_func_O1(xs + 0.5, y0)
    expected_gsO2 = expected_g11_func_O2(xs + 0.5, y0)
    fig, ax = plt.subplots()
    ax.plot(gs, '.')
    ax.plot(xs, expected_gsO1, '--', label="O(1)")
    ax.plot(xs, expected_gsO2, '--', label="O(2)")
    ax.set_title("g11 comparison")
    ax.legend()

    y0 = 20
    gs = g12[y0, :]
    xs = np.linspace(0, len(gs), 200)
    expected_gsO1 = expected_g12_func_O1(xs, y0)
    expected_gsO2 = expected_g12_func_O2(xs, y0)
    fig, ax = plt.subplots()
    ax.plot(gs, '.')
    ax.plot(xs, expected_gsO1, '--', label="O(1)")
    ax.plot(xs, expected_gsO2, '--', label="O(2)")
    ax.legend()

    W0expr = 4 * (-2 * cos(angle) ** 2 * cos(W0 / 2) ** 2 + cos(2 * angle) * (-1 + cos(W0))) / (
            3 + cos(2 * angle) - 2 * cos(W0) * sin(angle) ** 2)
    expected_K_func = lambda x, y: 1 / (4 * C0) * csc(angle) * tan(W0 / 2) ** 2 * dFF(x) * (
            2 * (L0 - ms_func(y)) / (L0 ** 2 - W0expr * L0 * ms_func(y) + ms_func(y) ** 2) +
            -(L0 - ms_func(y)) * dms_func(y) * (-W0expr * L0 + 2 * ms_func(y) + dms_func(y)) /
            (L0 ** 2 - W0expr * L0 * ms_func(y) + ms_func(y) ** 2) ** 2 +
            -(2 * (L0 - ms_func(y) - dms_func(y))) / (
                    L0 ** 2 - W0expr * (L0 * (ms_func(y) + dms_func(y))) + (ms_func(y) + dms_func(y)) ** 2)
    )

    expected_K_func2 = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                    ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
    expected_K_func3 = lambda x, y: -1 / (32 * C0 * L0 ** 3) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                    (2 * L0 * (ddMM(y) + dddMM(y) - 3 * dMM(y) * ddMM(y))) * (
                                            cos(W0) - 2 * csc(angle) ** 2 + 1)

    y0 = 10
    calc_Ks = Ks[y0, :]
    xs = np.linspace(0, len(calc_Ks), 200)
    expected_KsO2 = expected_K_func(xs, y0)
    expected_KsV2 = expected_K_func2(xs, y0)
    expected_KsV3 = expected_K_func3(xs, y0)
    fig, ax = plt.subplots()
    ax.plot(calc_Ks, '.')
    ax.plot(xs, expected_KsO2, '--', label="O(2)")
    ax.plot(xs, expected_KsV2, '--', label="V2")
    ax.plot(xs, expected_KsV3, '--', label="V2-O(3)")
    ax.set_title(f"Ks comparison, y={y0}")
    ax.legend()

    fig, ax = _compare_curvatures(Ks, expected_K_func)
    # fig.savefig(os.path.join(FIGURES_PATH, 'Ks-comparison.png'))
    plt.show()

    plot_interactive(ori)


def _compare_curvatures(Ks, expected_K_func) -> Tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(2)

    len_ys, len_xs = Ks.shape
    xs, ys = np.arange(len_xs), np.arange(len_ys)
    Xs, Ys = np.meshgrid(xs, ys)

    im = imshow_with_colorbar(fig, axes[0], Ks, "K")
    vmin, vmax = im.get_clim()
    im2 = imshow_with_colorbar(fig, axes[1], expected_K_func(Xs, Ys), "expected K")
    im2.set_clim(vmin, vmax)

    return fig, axes


def test_constant_curvature():
    rows = 40
    cols = 40
    # rows = 8
    # cols = 24
    angle = np.pi / 2 - 0.2
    W0 = 2.6

    F = lambda x: 0.01 * (x - cols / 2)

    MM = lambda y: 0.01 * ((y - rows / 4)) ** 2

    FF = lambda x: F(x * 2)

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    fig, _ = plot_flat_quadrangles(ori.dots)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    # KsOther, _, _, _ = origamimetric.calc_curvature_by_triangles(ori.dots)

    fig, axes = plt.subplots(2, 2)
    imshow_with_colorbar(fig, axes[0, 0], g11, "g11")
    imshow_with_colorbar(fig, axes[0, 1], g22, "g22")
    imshow_with_colorbar(fig, axes[1, 0], g12, "g12")
    imshow_with_colorbar(fig, axes[1, 1], Ks, "K")

    # fig, ax = plt.subplots()
    # imshow_with_colorbar(fig, ax, KsOther, "Ks by triangles")

    dF = FF(1) - FF(0)
    ddMM = MM(2) + MM(0) - 2 * MM(1)
    expectedK = -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dF * \
                ddMM * (cos(W0) - 2 * csc(angle) ** 2 + 1)
    print(expectedK)
    print("angles only", tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * \
          (cos(W0) - 2 * csc(angle) ** 2 + 1))

    plot_interactive(ori)


def plot_constant_curvature():
    F = lambda x: 0.01 * (x - cols / 2)

    MM = lambda y: 0.01 * ((y - rows / 4)) ** 2
    dMM = lambda y: MM(y + +0.5) - MM(y - 0.5)
    ddMM = lambda y: dMM(y + +0.5) - dMM(y - 0.5)

    FF = lambda x: F(x * 2)
    dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)

    L0 = 1
    C0 = 0.5

    angles = [0.9, 1.2, np.pi / 2 - 0.2, np.pi / 2 - 0.1, np.pi / 2 - 0.05, np.pi / 2 - 0.02]
    rows, cols = 34, 34
    W0 = 2.7
    fig, axes = plt.subplots(3, 2)
    fig: Figure = fig
    axes = axes.flat
    print(rf"Params: activation angle $ \omega={W0:.3f} $, rows,cols={rows}X{cols}")
    for i, angle in enumerate(angles):
        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
        expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
        K0 = expected_K_func(1, 1)
        imshow_with_colorbar(fig, axes[i], Ks, fr'$ \vartheta={angle:.2f} $, K0={K0:.3f}')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'constant-curvature-vs-angle'))

    rows = 50
    cols = 50
    angle = np.pi / 2 - 0.2
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    print(rf"Params: angle $ \vartheta={angle:.3f} $, rows,cols={rows}X{cols}")

    omegas = [0.01, 1, 2, 2.5, 2.8, 3]
    fig, axes = plt.subplots(3, 2)
    fig: Figure = fig
    axes = axes.flat
    for i, omega in enumerate(omegas):
        # print("angles only", tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * \
        #       (cos(W0) - 2 * csc(angle) ** 2 + 1))

        expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        W0 = omega
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
        K0 = expected_K_func(1, 1)
        imshow_with_colorbar(fig, axes[i], Ks, f'$ \omega={W0:.2f} $, K0={K0:.3f}')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'constant-curvature-vs-omega'))

    plt.show()

    # plot_interactive(ori)


def plot_hills():
    rows, cols = 80, 80
    angle = np.pi / 2 - 0.2
    W0 = 3

    L0 = 1
    C0 = 0.5

    config = [(0.02, 0.1, 'hills-small.png'), (0.1, 0.7, 'hills-big.png')]
    for F0, MM0, name in config:
        F = lambda x: F0 * np.sin(2 * np.pi * x / 39)

        MM = lambda y: MM0 * np.cos(2 * np.pi * y / 16)
        dMM = lambda y: MM(y + 1) - MM(y)
        ddMM = lambda y: dMM(y + 1) - dMM(y)

        FF = lambda x: F(x * 2)
        dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)

        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

        expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        fig, ax = _compare_curvatures(Ks, expected_K_func)
        print(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
        fig.suptitle(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_PATH, name))

    plt.show()


def create_perturbed_origami(angle, rows, cols, L0, C0, F, MM) -> RFFQM:
    dMM = lambda y: MM(y + 1) - MM(y)
    ddMM = lambda y: dMM(y + 1) - dMM(y)

    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0

    ys = np.arange(len(ls) // 2)
    ls[1::2] = L0 + dMM(ys) + 1 / 2 * ddMM(ys)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)
    return ori


def main():
    # debug_lengths_perturbations()
    # test_angles()
    # test_metric()
    # test_constant_curvature()
    plot_hills()
    # plot_constant_curvature()


if __name__ == '__main__':
    main()
