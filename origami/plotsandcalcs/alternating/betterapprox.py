"""
Some history:
At first, we used alternating angle perturbation to calculate the Gaussian curvature
after linearization of the perturbed angles and vertical lengths
We got results that seems to work well but only when the deflection of the folded
paper from a plane were small. In a sense, this is what we could expect.

Here we test the hypothesis that the Gaussian curvature is simply the multiplication
of the radii of curvature caused separately by the angle perturbation and by the lengths perturbation
Hopefully it will give us better predictions to the Gaussian curvature.
"""
import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import fsolve

import origami
from origami import origamimetric
from origami.origamiplots import plot_interactive
from origami.alternatingpert import curvatures
from origami.alternatingpert.curvatures import create_kx_ky_funcs, create_expected_K_func, \
    create_expected_curvatures_func
from origami.alternatingpert.utils import compare_curvatures as compare_G_curvatures, create_F_from_list, \
    create_MM_from_list
from origami.alternatingpert.utils import csc, sec, get_FF_dFF_dMM_ddMM, \
    create_perturbed_origami, tan, cos
from origami.utils.plotutils import imshow_with_colorbar

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM/ContinuousMetric/AlternatingFigures/BetterApprox/')


def test_constant():
    rows = 40
    cols = 40
    theta = np.pi / 2 - 0.2
    W0 = 2.9

    F = lambda x: 0.01 * (x - cols / 2)
    MM = lambda y: 0.02 * (y - rows / 4) ** 2

    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

    kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, theta)
    expected_K_func = lambda x, y: kx_func(FF(x), dFF(x)) * ky_func(dMM(y), ddMM(y))

    fig, axes = plt.subplots(1, 2)
    compare_G_curvatures(fig, axes, Ks, expected_K_func)
    fig.tight_layout()
    # fig.savefig(os.path.join(FIGURES_PATH, 'constant.png'))

    angle = theta
    dF0 = dFF(1)
    ddMM0 = ddMM(1)
    expectedK = -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dF0 * \
                ddMM0 * (cos(W0) - 2 * csc(angle) ** 2 + 1)
    print(expectedK)

    plot_interactive(ori)

    plt.show()


def test_hills():
    rows, cols = 80, 80
    angle = np.pi / 2 - 0.2
    W0 = 3

    L0 = 1
    C0 = 0.5

    F0, MM0 = 0.1, 0.7  # This is large perturbations

    F = lambda x: F0 * np.sin(2 * np.pi * x / 39)
    MM = lambda y: MM0 * np.cos(2 * np.pi * y / 16)
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

    expected_K_func_bad = lambda x, y: \
        -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * \
        tan(angle) * sec(angle) * dFF(x) * \
        ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)

    kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, angle)
    expected_K_func = lambda x, y: kx_func(FF(x), dFF(x)) * ky_func(dMM(y), ddMM(y))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    vmin, vmax = -4.1, 4.1
    len_ys, len_xs = Ks.shape
    xs, ys = np.arange(len_xs), np.arange(len_ys)
    Xs, Ys = np.meshgrid(xs, ys)

    im = axes[0].imshow(Ks, vmin=vmin, vmax=vmax, origin='lower')
    axes[1].imshow(expected_K_func(Xs, Ys), vmin=vmin, vmax=vmax, origin='lower')
    axes[2].imshow(expected_K_func_bad(Xs, Ys), vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title("Ks")
    axes[1].set_title("better\napproximation")
    axes[2].set_title("old\napproximation")
    print(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
    fig.suptitle(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.08, shrink=1.0)
    fig.savefig(os.path.join(FIGURES_PATH, 'hills.png'))


def test_constant_angle_factor():
    angles_factor = 1000

    L0 = 1
    C0 = 0.5
    rows, cols = 30, 40
    F = lambda x: 0.005 * (x - cols / 2)

    MM = lambda y: 0.02 * (y - rows / 4) ** 2
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

    def find_omega_for_angle(angle) -> float:
        eq_func = lambda w: -tan(w / 2) ** 2 * tan(angle) * sec(angle) * \
                            (cos(w) - 2 * csc(angle) ** 2 + 1) - angles_factor
        return fsolve(eq_func, 1, factor=0.1)[0]

    print(F(0))

    angles = [0.7, 1, 1.4, 1.5]
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.5))
    fig: Figure = fig
    im = None
    for i, angle in enumerate(angles):
        W0 = find_omega_for_angle(angle)
        print('W0', W0)

        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
        expected_K_func_bad = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(
            x) * \
                                           ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
        K0 = expected_K_func_bad(1, 1)
        print(K0)

        expected_K_func = create_expected_K_func(L0, C0, W0, angle, F, MM)

        len_ys, len_xs = Ks.shape
        xs, ys = np.arange(len_xs), np.arange(len_ys)
        Xs, Ys = np.meshgrid(xs, ys)

        im = axes[0, i].imshow(Ks, vmin=0.005, vmax=0.07, origin='lower')
        axes[0, i].set_title(fr'$ \vartheta={angle:.2f} $' + '\n' + f'$ W_0={W0:.2f} $')
        im2 = axes[1, i].imshow(expected_K_func(Xs, Ys), vmin=0.005, vmax=0.07, origin='lower')
        axes[1, i].set_title("predicted")

    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.08, shrink=1.0)
    fig.savefig(os.path.join(FIGURES_PATH, 'constant-angles-factor.png'))

    plt.show()


def compare_y_curvature_discrete_to_continuous():
    Mt = [0.5, 0.311076, 0.230632, 0.178721, 0.140719, 0.110946, 0.0865849,
          0.0660338, 0.0482977, 0.0327172, 0.0188342, 0.0063181, -0.00507702,
          -0.0155388, -0.0252134, -0.0342169, -0.0426426, -0.0505672,
          -0.0580537, -0.065155, -0.0719158, -0.0783741, -0.0845626,
          -0.0905096, -0.0962397, -0.101775, -0.107134, -0.112333, -0.117389,
          -0.122315, -0.127122]
    MMs = [0., 0.423659, 0.74038, 0.98815, 1.18635, 1.34649, 1.47606, 1.58026,
           1.66291, 1.72687, 1.77437, 1.80721, 1.82682, 1.83439, 1.83093,
           1.81727, 1.79415, 1.76217, 1.72189, 1.67377, 1.61822, 1.55563,
           1.48631, 1.41056, 1.32864, 1.24079, 1.14721, 1.0481, 0.943626,
           0.833954, 0.719222]
    Mt = np.array(Mt)
    MMs = np.array(MMs)
    Ms = np.append(0, np.cumsum(Mt))

    rows = 60
    cols = 20
    angle = 0.8
    W0 = 3.0
    L0 = 0.2
    C0 = 0.9

    def MM_discrete_func(y):
        return Ms[y]

    def MM_continuous_func(y):
        return MMs[y]

    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    def calc_ky_by_MM_pert(MM_func):
        ori = create_perturbed_origami(angle, rows, cols, L0, C0, None, MM_func)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        return 2 * Hs

    ky = calc_ky_by_MM_pert(MM_discrete_func)
    im = imshow_with_colorbar(fig, axes[0], ky, 'discrete')
    im.set_clim(0.1 - 0.01, 0.1 + 0.01)
    ky = calc_ky_by_MM_pert(MM_continuous_func)
    imshow_with_colorbar(fig, axes[1], ky, 'continuous')

    Mt = [0.5, 0.481108, 0.463602, 0.447316, 0.432111, 0.417869, 0.404489,
          0.391885, 0.379983, 0.368717, 0.358031, 0.347876, 0.338208, 0.328987,
          0.320178, 0.311752, 0.30368, 0.295936, 0.288499, 0.281349, 0.274465,
          0.267832, 0.261434, 0.255257, 0.249288, 0.243514, 0.237926, 0.232513,
          0.227265, 0.222174, 0.217232]
    MMs = [0., 0.490784, 0.964, 1.42084, 1.86235, 2.28948, 2.70309, 3.10393,
           3.49272, 3.87008, 4.23659, 4.59278, 4.93914, 5.27612, 5.60414,
           5.92358, 6.2348, 6.53814, 6.83389, 7.12235, 7.4038, 7.67849, 7.94665,
           8.20851, 8.46428, 8.71416, 8.95835, 9.19701, 9.43032, 9.65843, 9.8815]
    Mt = np.array(Mt)
    MMs = np.array(MMs)
    Ms = np.append(0, np.cumsum(Mt))

    ky = calc_ky_by_MM_pert(MM_discrete_func)
    im = imshow_with_colorbar(fig, axes[2], ky, 'discrete')
    im.set_clim(0.01 - 0.001, 0.01 + 0.001)
    ky = calc_ky_by_MM_pert(MM_continuous_func)
    imshow_with_colorbar(fig, axes[3], ky, 'continuous')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'ky-discrete-vs-PDE.svg'))

    plt.show()


def test_x_constant():
    Fs = [0.0988263, 0.0925858, 0.0871846, 0.0824412, 0.078225, 0.0744391,
          0.0710099, 0.0678804, 0.0650055, 0.0623492, 0.0598823, 0.0575805,
          0.055424, 0.053396, 0.0514823, 0.0496708, 0.0479512, 0.0463143,
          0.0447526, 0.043259, 0.0418277, 0.0404533, 0.0391311, 0.0378569,
          0.0366269, 0.0354379, 0.0342867, 0.0331706, 0.0320872, 0.0310341,
          0.0300093, 0.0290109, 0.0280372, 0.0270866, 0.0261576, 0.0252488,
          0.024359, 0.023487, 0.0226318, 0.0217923, 0.0209675, 0.0201567,
          0.0193589, 0.0185734, 0.0177995, 0.0170364, 0.0162835, 0.0155402,
          0.0148059, 0.01408, 0.0133621, 0.0126515, 0.0119478, 0.0112506,
          0.0105593, 0.00987364, 0.00919311, 0.00851732, 0.00784589,
          0.00717844, 0.00651461, 0.00585403, 0.00519637, 0.00454127,
          0.00388841, 0.00323745, 0.00258808, 0.00193996, 0.00129279,
          0.00064624, 0., -0.00064624, -0.00129279, -0.00193996, -0.00258808,
          -0.00323745, -0.00388841, -0.00454127, -0.00519637, -0.00585404,
          -0.00651461, -0.00717845, -0.00784589, -0.00851732, -0.00919311,
          -0.00987365, -0.0105593, -0.0112506, -0.0119478, -0.0126515,
          -0.0133621, -0.0140801, -0.0148059, -0.0155403, -0.0162836,
          -0.0170365, -0.0177996, -0.0185735, -0.019359, -0.0201568,
          -0.0209677, -0.0217924, -0.022632, -0.0234872, -0.0243592,
          -0.0252491, -0.0261579, -0.027087, -0.0280377, -0.0290114,
          -0.0300099, -0.0310347, -0.0320879, -0.0331715, -0.0342877,
          -0.035439, -0.0366281, -0.0378583, -0.0391326, -0.0404551,
          -0.0418297, -0.0432613, -0.0447552, -0.0463173, -0.0479546,
          -0.0496747, -0.0514868, -0.0534011, -0.0554299, -0.0575873,
          -0.0598901, -0.0623584, -0.0650162, -0.0678929, -0.0710247,
          -0.0744567, -0.0782461, -0.0824667, -0.0872158, -0.0926245,
          -0.0988752]
    Fs = np.array(Fs)
    rows = 20
    cols = 139
    angle = 0.8
    W0 = 3.0
    L0 = 0.2
    C0 = 0.9

    def F(x):
        # return 0 * x
        if isinstance(x, np.ndarray):
            print(x.dtype)
            if np.issubdtype(x.dtype, 'float64'):
                x = x.astype('int')
        return Fs[x + 1]

    fig, axes = plt.subplots(2, 1)

    def calc_kx(F_func):
        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F_func, None)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        return 2 * Hs

    kx = calc_kx(F)
    im = imshow_with_colorbar(fig, axes[0], kx, 'better approx')
    # im.set_clim(0.1 - 0.01, 0.1 + 0.01)
    F = lambda x: -0.00064624 * (x - cols // 2)
    kx = calc_kx(F)
    imshow_with_colorbar(fig, axes[1], kx, 'old linearization')
    fig.savefig(os.path.join(FIGURES_PATH, 'x-curvature-approx-comparison.svg'))


def test_IP_constant():
    Fs = [0.0988263, 0.0925858, 0.0871846, 0.0824412, 0.078225, 0.0744391,
          0.0710099, 0.0678804, 0.0650055, 0.0623492, 0.0598823, 0.0575805,
          0.055424, 0.053396, 0.0514823, 0.0496708, 0.0479512, 0.0463143,
          0.0447526, 0.043259, 0.0418277, 0.0404533, 0.0391311, 0.0378569,
          0.0366269, 0.0354379, 0.0342867, 0.0331706, 0.0320872, 0.0310341,
          0.0300093, 0.0290109, 0.0280372, 0.0270866, 0.0261576, 0.0252488,
          0.024359, 0.023487, 0.0226318, 0.0217923, 0.0209675, 0.0201567,
          0.0193589, 0.0185734, 0.0177995, 0.0170364, 0.0162835, 0.0155402,
          0.0148059, 0.01408, 0.0133621, 0.0126515, 0.0119478, 0.0112506,
          0.0105593, 0.00987364, 0.00919311, 0.00851732, 0.00784589,
          0.00717844, 0.00651461, 0.00585403, 0.00519637, 0.00454127,
          0.00388841, 0.00323745, 0.00258808, 0.00193996, 0.00129279,
          0.00064624, 0., -0.00064624, -0.00129279, -0.00193996, -0.00258808,
          -0.00323745, -0.00388841, -0.00454127, -0.00519637, -0.00585404,
          -0.00651461, -0.00717845, -0.00784589, -0.00851732, -0.00919311,
          -0.00987365, -0.0105593, -0.0112506, -0.0119478, -0.0126515,
          -0.0133621, -0.0140801, -0.0148059, -0.0155403, -0.0162836,
          -0.0170365, -0.0177996, -0.0185735, -0.019359, -0.0201568,
          -0.0209677, -0.0217924, -0.022632, -0.0234872, -0.0243592,
          -0.0252491, -0.0261579, -0.027087, -0.0280377, -0.0290114,
          -0.0300099, -0.0310347, -0.0320879, -0.0331715, -0.0342877,
          -0.035439, -0.0366281, -0.0378583, -0.0391326, -0.0404551,
          -0.0418297, -0.0432613, -0.0447552, -0.0463173, -0.0479546,
          -0.0496747, -0.0514868, -0.0534011, -0.0554299, -0.0575873,
          -0.0598901, -0.0623584, -0.0650162, -0.0678929, -0.0710247,
          -0.0744567, -0.0782461, -0.0824667, -0.0872158, -0.0926245,
          -0.0988752]

    Mt = [0.5, 0.311076, 0.230632, 0.178721, 0.140719, 0.110946, 0.0865849,
          0.0660338, 0.0482977, 0.0327172, 0.0188342, 0.0063181, -0.00507702,
          -0.0155388, -0.0252134, -0.0342169, -0.0426426, -0.0505672,
          -0.0580537, -0.065155, -0.0719158, -0.0783741, -0.0845626,
          -0.0905096, -0.0962397, -0.101775, -0.107134, -0.112333, -0.117389,
          -0.122315, -0.127122]
    Ms = np.append(0, np.cumsum(Mt))
    Fs = np.array(Fs)
    # Ms = np.array(Ms)

    rows = 50
    cols = 139
    angle = 0.8
    W0 = 3.0
    L0 = 0.2
    C0 = 0.5

    F = create_F_from_list(Fs)
    MM = create_MM_from_list(Ms)

    # F = lambda x: -0.00064624 * (x - cols // 2)

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    # plot_interactive(ori)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    # This is a trick to correct that actual W0 by the place F vanishes
    omegas = ori.calc_omegas_vs_x()
    W0 = omegas[len(omegas) // 2]
    fig, ax = plt.subplots()
    ax.plot(omegas, '.')

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, angle, F, MM)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    # fig, _ = compare_curvatures(Ks, Hs, None, None)
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    plt.show()
    plot_interactive(ori)


def compare_curvatures(Ks, Hs, expected_K_func, expected_H_func) -> Tuple[Figure, np.ndarray[Axes]]:
    fig, axes = plt.subplots(2, 2)

    len_ys, len_xs = Ks.shape
    Nx, Ny = len_xs + 1, len_ys + 1
    xs, ys = np.arange(Nx - 1) / Nx, np.arange(Ny - 1) / Ny
    Xs, Ys = np.meshgrid(xs, ys)

    im = imshow_with_colorbar(fig, axes[0, 0], Ks, "K")
    vmin, vmax = im.get_clim()

    print(ys)

    if expected_K_func is not None:
        im2 = imshow_with_colorbar(fig, axes[1, 0], expected_K_func(Xs, Ys), "expected K")
        # im2.set_clim(vmin, vmax)

    im = imshow_with_colorbar(fig, axes[0, 1], Hs, "H")
    vmin, vmax = im.get_clim()
    if expected_H_func is not None:
        im2 = imshow_with_colorbar(fig, axes[1, 1], expected_H_func(Xs, Ys), "expected H")
        # im2.set_clim(vmin, vmax)
    return fig, axes


def compare_to_old():
    angle = np.pi / 2 - 0.2
    W0 = 3

    rows = 48
    cols = 48

    F = lambda x: 0.004 / 2 * (x - cols / 2)
    MMt = lambda y: 0.02 * ((y - rows / 4) / 2) ** 2
    MM = lambda y: MMt(y) - MMt(0)
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)

    L0 = 1 / 2
    C0 = 0.5 / 2

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    ori_old = ori

    fig, ax = plt.subplots()
    imshow_with_colorbar(fig, ax, Ks, "K - old")

    dF = FF(1) - FF(0)
    ddMM = MM(2) + MM(0) - 2 * MM(1)
    expectedK = -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dF * \
                ddMM * (cos(W0) - 2 * csc(angle) ** 2 + 1)

    print(expectedK)

    # kx = -1.00
    ky = -0.1

    kx = -1.5
    ky = -0.26

    print(kx * ky)

    # F0 = -0.048
    F0 = -0.055
    M0 = dMM(0)
    # M0 = 0.125
    theta = angle

    xs, Fs = curvatures.get_deltas_for_kx(L0, C0, W0, theta, kx, F0, 0)
    ys, MMs = curvatures.get_Deltas_for_ky(L0, C0, W0, theta, ky, M0, 0)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(xs, FF(xs), '.', label='old')
    axes[0].plot(xs, Fs, '.', label='better')
    axes[0].legend()
    # axes[1].plot(ys, np.diff(MMs), '.')
    axes[1].plot(ys, MM(ys), '.', label='old')
    axes[1].plot(ys, MMs[:-1], '.', label='better')
    axes[1].legend()
    # plt.show()

    F = create_F_from_list(Fs)
    MM = create_MM_from_list(MMs)

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)

    plot_interactive(ori)
    # plot_interactive(ori_old)


def main():
    test_constant()
    # test_hills()
    # test_constant_angle_factor()
    # test_IP_constant()
    # compare_y_curvature_discrete_to_continuous()
    # test_x_constant()
    # compare_to_old()
    plt.show()


if __name__ == "__main__":
    main()
