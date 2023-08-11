import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

import origami.plotsandcalcs
from origami import origamimetric, quadranglearray
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import create_angles_func_vertical_alternation, set_perturbations_by_func
from origami.interactiveplot import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.plotsandcalcs.alternating import betterapproxcurvatures
from origami.plotsandcalcs.alternating.utils import sin, cos, tan, csc, sec, compare_curvatures, get_FF_dFF_dMM_ddMM, \
    create_perturbed_origami, create_F_from_list
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import plotutils, zigzagutils
from origami.utils.fitter import FitParams, FuncFit
from origami.utils.plotutils import imshow_with_colorbar

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM/ContinuousMetric/AlternatingFigures')


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

    fig, axes = plt.subplots(2)
    compare_curvatures(fig, axes, Ks, expected_K_func)
    # fig.savefig(os.path.join(FIGURES_PATH, 'Ks-comparison.png'))
    plt.show()

    plot_interactive(ori)


def test_high_resolution():
    resolution_factors = [1, 2, 3]
    angle = np.pi / 2 - 0.2
    W0 = 3
    fig, axes = plt.subplots(3)
    axes = axes.flat

    for i, factor in enumerate(resolution_factors):
        rows = 24 * factor
        cols = 24 * factor

        F = lambda x: 0.004 / factor * (x - cols / 2)
        MM = lambda y: 0.02 * ((y - rows / 4) / factor) ** 2
        FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

        L0 = 1 / factor
        C0 = 0.5 / factor
        print(F(cols))

        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

        im = imshow_with_colorbar(fig, axes[i], Ks, f"factor: {factor}")
        im.set_extent([0, cols, 0, rows])

        dF = FF(1) - FF(0)
        ddMM = MM(2) + MM(0) - 2 * MM(1)
        expectedK = -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dF * \
                    ddMM * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        # if i == 2:
        #     plot_interactive(ori)
        print(expectedK)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, "constant-curvature-changing-resolution.png"))
    plt.show()


def test_keeping_angles_factor_constant():
    angles_factor = 1000

    L0 = 1
    C0 = 0.5
    rows, cols = 30, 40
    F = lambda x: 0.005 * (x - cols / 2)

    MM = lambda y: 0.02 * ((y - rows / 4)) ** 2
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

    def find_omega_for_angle(angle) -> float:
        eq_func = lambda w: -tan(w / 2) ** 2 * tan(angle) * sec(angle) * \
                            (cos(w) - 2 * csc(angle) ** 2 + 1) - angles_factor
        return fsolve(eq_func, 1, factor=0.1)[0]

    print(F(0))

    angles = [0.7, 1, 1.3, 1.4, 1.45, 1.5]
    fig, axes = plt.subplots(2, 3)
    fig: Figure = fig
    axesflat = axes.flat
    for i, angle in enumerate(angles):
        W0 = find_omega_for_angle(angle)
        print('W0', W0)

        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
        expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
        K0 = expected_K_func(1, 1)
        print(K0)
        ax = axesflat[i]
        im = ax.imshow(Ks, vmin=0.005, vmax=0.06, origin='lower')
        # ax.invert_yaxis()
        ax.set_title(fr'$ \vartheta={angle:.2f} $' + '\n' + f'W0={W0:.2f}')

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()
    fig.savefig(os.path.join(FIGURES_PATH, 'constant-curvature-constant-angles-factor.png'))


def test_constant_curvature():
    rows = 40
    cols = 40
    # rows = 8
    # cols = 24
    angle = np.pi / 2 - 0.2
    W0 = 3

    F = lambda x: -0.01 * (x - cols / 2)

    MM = lambda y: 0.04 * ((y - rows / 4)) ** 2

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


def plot_cylinders():
    rows = 30
    cols = 80
    angle = np.pi / 2 - 0.1
    W0 = 2.9

    F = lambda x: -0.02 * (x - cols / 2)
    MM = lambda y: 0 * y

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    # fig, _ = plot_flat_quadrangles(ori.dots)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    _plot_Ks_metric(Ks, g11, g12, g22)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ori.dots.plot(ax, alpha=0.3)
    fig.savefig(os.path.join(FIGURES_PATH, 'cylinder-in-x.png'))

    # plot_interactive(ori)

    rows = 60
    cols = 20
    angle = np.pi / 2 - 0.1
    W0 = 0.7

    F = lambda x: 0 * x
    MM = lambda y: 0.005 * y ** 2

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    # fig, _ = plot_flat_quadrangles(ori.dots)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    _plot_Ks_metric(Ks, g11, g12, g22)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-33, elev=6)
    ori.dots.plot(ax, alpha=0.5)
    fig.savefig(os.path.join(FIGURES_PATH, 'cylinder-in-y.png'))

    plot_interactive(ori)


def _plot_Ks_metric(Ks, g11, g12, g22):
    fig, axes = plt.subplots(2, 2)
    imshow_with_colorbar(fig, axes[0, 0], g11, "g11")
    imshow_with_colorbar(fig, axes[0, 1], g22, "g22")
    imshow_with_colorbar(fig, axes[1, 0], g12, "g12")
    imshow_with_colorbar(fig, axes[1, 1], Ks, "K")
    return fig, axes


def test_2nd_order_MM():
    F = lambda x: 0.005 * (x - cols / 2)

    MM = lambda y: 0.005 * ((y - rows / 4)) ** 2
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)
    dddMM = lambda y: ddMM(y + 1) - ddMM(y)

    L0 = 1
    C0 = 0.5

    angle = np.pi / 2 - 0.3
    rows, cols = 34, 34
    W0 = 2.7
    fig, axes = plt.subplots(3, 1)
    fig: Figure = fig
    axes = axes.flat

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                   ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
    expected_K_func_MM2 = lambda x, y: -1 / (32 * C0 * L0 ** 3) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) \
                                       * (cos(W0) - 2 * csc(angle) ** 2 + 1) * \
                                       (2 * L0 * (dddMM(y) + ddMM(y)) - 3 * dMM(y) * ddMM(y))

    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)

    compare_curvatures(fig, (axes[0], axes[1]), Ks, expected_K_func)
    compare_curvatures(fig, (axes[0], axes[2]), Ks, expected_K_func_MM2)

    fig.tight_layout()
    # fig.savefig(os.path.join(FIGURES_PATH, 'MM-higher-order-correction.png'))

    F = lambda x: 0.005 * (x - cols / 2)

    K0 = 4 * 0.005
    L0 = 1
    C0 = 0.5

    MM = lambda y: K0 / (4 * L0) * (y - rows / 4) ** 2 + 3 / (2 * L0) * K0 ** 2 / (24 * L0 ** 2) * (y - rows / 4) ** 3
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)
    dddMM = lambda y: ddMM(y + 1) - ddMM(y)

    angle = np.pi / 2 - 0.3
    rows, cols = 34, 34
    W0 = 2.7
    fig, axes = plt.subplots(3, 1)
    fig: Figure = fig
    axes = axes.flat

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                   ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
    expected_K_func_MM2 = lambda x, y: -1 / (32 * C0 * L0 ** 3) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) \
                                       * (cos(W0) - 2 * csc(angle) ** 2 + 1) * \
                                       (2 * L0 * (dddMM(y) + ddMM(y)) - 3 * dMM(y) * ddMM(y))

    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)

    compare_curvatures(fig, (axes[0], axes[1]), Ks, expected_K_func)
    compare_curvatures(fig, (axes[0], axes[2]), Ks, expected_K_func_MM2)
    fig.suptitle("with correction")

    plt.show()


def test_M_vs_zigzag():
    F = lambda x: 0.005 * (x - cols / 2)

    MM = lambda y: 0.005 * ((y - rows / 4)) ** 2

    L0 = 1
    C0 = 0.5

    angle = np.pi / 2 - 0.3
    rows, cols = 50, 20
    W0 = 2.7

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    Ks_linear_M, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)

    # Now with zigzag
    xs = np.arange(-0.6, 0.8, 0.05)
    ys = 1 - np.sqrt(1 - xs ** 2)
    zigzag_angle = 2.5  # Some approximation

    ls_sphere, mid_x, mid_y = zigzagutils.follow_curve(xs, ys, (np.pi - zigzag_angle) / 2)
    ls = ls_sphere * 13.0
    fig, ax = plt.subplots()
    print(ls_sphere)
    zigzagutils.plot_zigzag(ax, xs, ys, mid_x, mid_y)
    more_xs = np.linspace(-0.6, 0.8, 100)
    ax.plot(more_xs, 1 - np.sqrt(1 - more_xs ** 2), '--')

    cols = 60
    # F = lambda x: 0.03 * sin(x/10)
    F = lambda x: 0.01 * (x - cols / 2)

    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

    cs = np.ones(cols) * C0

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori_zigzag = RFFQM(quads)
    ori_zigzag.set_gamma(ori_zigzag.calc_gamma_by_omega(W0))
    Ks_zigzag, _, _, _ = origamimetric.calc_curvature_and_metric(ori_zigzag.dots)

    fig, axes = plt.subplots(1, 2)
    imshow_with_colorbar(fig, axes[0], Ks_linear_M, "Ks MM parabolic")
    imshow_with_colorbar(fig, axes[1], Ks_zigzag, "Ks by zigzag ls")
    # print(Ks_zigzag-Ks_linear_M)

    should_plot_Ks = False
    if should_plot_Ks:
        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d')
        ori.dots.plot(ax, alpha=0.3)
        ori_zigzag.dots.plot(ax, alpha=0.3)
        plotutils.set_axis_scaled(ax)
        plotutils.set_3D_labels(ax)

    fig, ax = plt.subplots()
    quads = ori_zigzag.dots
    column1_indexes = quads.indexes[:, 0]
    column2_indexes = quads.indexes[:, cols // 2]
    column3_indexes = quads.indexes[:, -1]
    ax.plot(quads.dots[1, column1_indexes], quads.dots[2, column1_indexes] - quads.dots[2, column1_indexes[0]], '.')
    ax.plot(quads.dots[1, column2_indexes], quads.dots[2, column2_indexes] - quads.dots[2, column2_indexes[0]], '.')
    ax.plot(quads.dots[1, column3_indexes], quads.dots[2, column3_indexes] - quads.dots[2, column3_indexes[0]], '.')

    plt.show()


def test_horizontal_decay():
    F = lambda x: 0.005 * (x - cols / 2)

    MM = lambda y: 0.01 * ((y - rows / 4)) ** 2
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)
    dddMM = lambda y: ddMM(y + 1) - ddMM(y)

    L0 = 1
    C0 = 0.5

    rows = 20
    cols = 450
    angle = np.pi / 2 - 0.35
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    print(f"max F: {F(0)}")

    omega = 1.1
    print(rf"Params: angle $ \vartheta={angle:.3f} $ "
          rf"activation angle $ \omega={omega:.3f} $, rows,cols={rows}X{cols}")

    fig, ax = plt.subplots()
    fig: Figure = fig
    ax: Axes = ax
    quadranglearray.plot_flat_quadrangles(ori.dots)
    ori.set_gamma(ori.calc_gamma_by_omega(omega))
    # This is a patch that is supposed to fix the fact that we assume in the
    # calculations that F(0)=0
    W0 = ori.calc_omegas_vs_x()[cols // 4]
    print(W0)

    expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                   ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
    K0 = expected_K_func(1, 1)

    ys = Ks[5, :]
    # print('compare ks', K0, expected_K_func_MM2(5, 5))
    Ks_line = ys / K0
    xs = np.arange(len(Ks_line)) - len(Ks_line) / 2
    ax.plot(xs, ys, label=rf'$ \omega={omega:.2f} $')

    fig, axes = plt.subplots(2)
    compare_curvatures(fig, axes, Ks, expected_K_func)

    plt.show()


def plot_horizontal_decay():
    F = lambda x: 0.005 * (x - cols / 2)

    MM = lambda y: 0.01 * ((y - rows / 4)) ** 2
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)
    dddMM = lambda y: ddMM(y + 1) - ddMM(y)

    L0 = 1
    C0 = 0.5

    rows = 20
    cols = 250
    angle = np.pi / 2 - 0.2
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    print(rf"Params: angle $ \vartheta={angle:.3f} $, rows,cols={rows}X{cols}")

    omegas = [0.1, 0.5, 1, 1.05, 1.1, 1.15, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 2.8, 3]
    sigmas = np.zeros(len(omegas))
    sigmas_errs = np.zeros(len(omegas))
    fig, ax = plt.subplots()
    fig: Figure = fig
    ax: Axes = ax
    fig_unnormalized, ax_unnormalized = plt.subplots()
    # angle = angle - FF(0)
    for i, omega in enumerate(omegas):
        ori.set_gamma(ori.calc_gamma_by_omega(omega))
        # This is a patch that is supposed to fix the fact that we assume in the
        # calculations that F(0)=0
        W0 = ori.calc_omegas_vs_x()[cols // 4]
        print(W0 - omega)

        expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        expected_K_func_MM2 = lambda x, y: -1 / (32 * C0 * L0 ** 3) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(
            x) \
                                           * (cos(W0) - 2 * csc(angle) ** 2 + 1) * \
                                           (2 * L0 * (dddMM(y) + ddMM(y)) - 3 * dMM(y) * ddMM(y))

        Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
        K0 = expected_K_func(1, 1)

        ys = Ks[5, :]
        # print('compare ks', K0, expected_K_func_MM2(5, 5))
        Ks_line = ys / K0
        xs = np.arange(len(Ks_line)) - len(Ks_line) / 2
        ax_unnormalized.plot(xs, ys, label=rf'$ \omega={omega:.2f} $')

        # print(f"sum of ys: {np.sum(ys)}")

        def Gaussian_func(xs, A, x0, sigma):
            return A * np.exp(-(xs - x0) ** 2 / (2 * sigma ** 2))

        fit_params = FitParams(Gaussian_func, xs, Ks_line)
        fit_params.bounds = ([0.7, -10, 0], [2, 10, 300])
        fitter = FuncFit(fit_params)
        line1 = fitter.plot_data(ax, rf'$ \omega={omega:.2f} $')
        sigmas[i], sigmas_errs[i] = fitter.fit_results[2].n, fitter.fit_results[2].s
        if sigmas[i] < 250:
            line2 = fitter.plot_fit(ax)
            line2.set_color(line1.get_color())
        ax.set_xlim(xs[0] - 2, xs[-1] + 2)
        # fitter.print_results()

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('K[x]/K0')
    ax_unnormalized.legend()
    ax_unnormalized.set_ylabel('K[x]')

    fig.savefig(os.path.join(FIGURES_PATH, 'horizontal-decay-K-vs-x.png'))

    fig, ax = plt.subplots()
    ax.set_xlabel('$ \omega $')
    ax.set_ylabel('bell width, $ \sigma $')
    plotutils.set_pi_ticks(ax, 'x')
    # ax.errorbar(omegas, sigmas, sigmas_errs, fmt='.')
    ax.plot(omegas, sigmas, '.')

    fig.savefig(os.path.join(FIGURES_PATH, 'horizontal-decay-sigma-vs-omega.png'))

    plt.show()


def plot_constant_curvature():
    F = lambda x: 0.01 * (x - cols / 2)

    MM = lambda y: 0.01 * ((y - rows / 4)) ** 2
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

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
        FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

        expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        fig, axes = plt.subplots(2)
        compare_curvatures(fig, axes, Ks, expected_K_func)
        print(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
        fig.suptitle(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_PATH, name))

    plt.show()


def test_periodic_with_constant():
    rows, cols = 60, 60
    angle = np.pi / 2 - 0.1

    L0 = 1
    C0 = 0.5

    F0, MM0 = 0.002, 0.001
    F = lambda x: F0 * (np.sin(2 * np.pi * x / 39) + 5 * 2 * np.pi / 39 * x)

    MM = lambda y: MM0 * ((y - rows / 4)) ** 2
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    print(rf'F0={F0:.3f}, MM0={MM0:.3f}')
    fig, axes = plt.subplots(2, 3)
    omegas = [2, 2.8, 3]
    for i, omega in enumerate(omegas):
        W0 = omega
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

        expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        ax2 = axes[:, i]
        compare_curvatures(fig, ax2, Ks, expected_K_func)
    fig.suptitle(fr'$ \omega={omegas} $ from left to right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, "periodic-and-constant-vs-omega.png"))

    plt.show()


def plot_cone():
    rows, cols = 60, 60
    angle = np.pi / 2 - 0.1

    L0 = 1
    C0 = 0.5

    F0, MM0 = 0.001, 0.001
    F = lambda x: F0 * np.tanh((x - cols / 2) / 10)
    MM = lambda y: (((y - rows / 4)) ** 2 - 10 < 0) * MM0 * (((y - rows / 4)) ** 2 - 10)

    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    W0 = 2.6
    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

    expected_K_func = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                   ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)

    fig, axes = plt.subplots(2)
    compare_curvatures(fig, axes, Ks, expected_K_func)

    plt.show()


def test_zigzag_in_angles():
    """
    We know that we can achieve non-vanishing curvature in y-axis if
    we include different zigzagging angles in the horizontal direction.
    We test here if we can have this zigzag as well as F perturbation
    """
    rows, cols = 20, 20

    L0 = 0.6
    C0 = 0.5
    W0 = 2.5
    theta = 0.9

    kx = -0.5
    F0 = 0.0

    xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx, F0, 0, cols // 4 + 1)

    # Make Fs symmetric
    Fs = np.append(-Fs[1::][::-1], Fs)

    fig, axes = plt.subplots()
    axes.plot(Fs, '.')
    plt.show()

    F = create_F_from_list(Fs)
    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0

    angles_left, angles_bottom = create_miura_angles(ls, cs, theta)
    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)
    angles_left[:, 1::2] += 0.1

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    plot_interactive(ori)


def main():
    # plot_simple_flat_examples()
    # debug_lengths_perturbations()
    # test_angles()
    # test_metric()
    # test_constant_curvature()
    # plot_hills()
    # plot_constant_curvature()
    # test_periodic_with_constant()
    # plot_cone()
    # test_high_resolution()
    # test_keeping_angles_factor_constant()
    # test_2nd_order_MM()
    # plot_horizontal_decay()
    # test_horizontal_decay()
    # plot_cylinders()
    # test_M_vs_zigzag()
    # test_jump_in_F()
    test_zigzag_in_angles()


if __name__ == '__main__':
    main()
