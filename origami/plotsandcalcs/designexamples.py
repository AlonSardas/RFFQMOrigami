"""
Here we show some simple RFFQM origami that we can design by controlling
the principal curvatures.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import origami
from origami import origamimetric, origamiplots
from origami.origamiplots import plot_interactive
from origami.plotsandcalcs.alternating import curvatures
from origami.plotsandcalcs.alternating.betterapprox import compare_curvatures
from origami.plotsandcalcs.alternating.curvatures import create_expected_curvatures_func
from origami.plotsandcalcs.alternating.utils import create_F_from_list, create_MM_from_list, create_perturbed_origami, \
    plot_perturbations_by_list, create_perturbed_origami_by_list, get_pert_list_by_func, plot_perturbations
from origami.utils import plotutils, logutils
from origami.utils.plotutils import imshow_with_colorbar

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM', 'Figures', 'design-examples')


def plot_vase():
    """
    We reproduce the fig. 4.a.1 from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    rows, cols = 40, 30

    L0 = 0.5
    C0 = 1
    W0 = 2.5
    theta = 1.0

    delta0 = -0.3
    Delta0 = 0.3

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = L0 * Ny, C0 * Nx

    kx_func = lambda t: 7 * 1 / (80 / 2)
    # ky_func = lambda t: -0.2 * np.tanh((t - rows / 4) * 3)
    # kx_func = lambda t: -1 / C_tot
    ky_func = lambda t: +0.25 * np.tanh((t - 0.5) * 3)

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx_func, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky_func, Delta0)

    xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    plot_perturbations_by_list(xs, deltas, ys, Deltas)

    ori = create_perturbed_origami_by_list(theta, L0, C0, deltas, Deltas)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    fig.tight_layout()
    # fig.savefig(os.path.join(FIGURES_PATH, 'vase-curvatures.svg'))
    # fig.savefig(os.path.join(FIGURES_PATH, 'vase-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=50, elev=30)
    lim = 4.0
    panels = ori.dots.plot(ax, alpha=0.6, edge_alpha=0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # fig.tight_layout(rect=(0.3, 0, 0.7, 1))
    # fig.savefig(os.path.join(FIGURES_PATH, 'vase.svg'), pad_inches=0.4)
    # fig.savefig(os.path.join(FIGURES_PATH, 'vase.pdf'), pad_inches=0.4)
    plt.show()

    plot_interactive(ori)


def plot_spherical_cap():
    """
    We reproduce the fig. 4.b from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    rows, cols = 20, 30
    # kx_func = lambda t: 3.5 * 1 / (80 / 2)
    # ky_func = lambda t: 0.2 * (t - 10) / (rows / 2)
    # ky_func = lambda t: -0.2 * np.tanh((t - rows / 4) * 3)
    kx = 0.10
    ky = 0.10

    L0 = 1.0
    C0 = 1
    W0 = 2.4
    theta = 1.1

    F0 = -0.2
    M0 = -0.5
    Delta0 = M0 / L0
    delta0 = F0

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = L0 * Ny, C0 * Nx

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    plot_perturbations_by_list(xs, deltas, ys, Deltas)

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)
    ori.set_gamma(0)

    # fig, _ = plot_crease_pattern(ori)
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.svg'))
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.png'))

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    Hs *= -1
    expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical_cap-curvatures.svg'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-50, elev=20)
    lim = 5.0
    _, wire = ori.dots.plot_with_wireframe(ax, alpha=0.6)
    wire.set_alpha(0.0)

    plotutils.set_axis_scaled(ax)
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)

    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.svg'), pad_inches=0.4)
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.pdf'), pad_inches=0.4)
    plt.show()
    plot_interactive(ori)


def plot_saddle():
    """
    We reproduce the fig. 4.c from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    L_tot, C_tot = 15, 18

    rows, cols = 20, 30
    # rows, cols = 80, 40
    kx = -0.07
    ky = 0.07

    Nx, Ny = cols // 2, rows // 2
    # L0 = 1.5
    # C0 = 1.2
    L0 = L_tot / Ny
    C0 = C_tot / Nx
    print(L0, C0)
    W0 = 2.4
    theta = 1.0

    F0 = 0.18
    delta0 = F0
    Delta0 = -0.9

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    fig, axes = plt.subplots(2)
    plot_perturbations(axes, delta_func, Delta_func, Nx, Ny)

    # fig, ax = plt.subplots()
    # xs = np.linspace(0, 1, 50)
    # eps = 0.01
    # dDelta = 1/(2*eps)*(Delta_func(xs+eps)-Delta_func(xs-eps))
    # ax.plot(xs, dDelta, '.')
    # plt.show()

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)

    # xs, deltas = curvatures.get_delta_for_kx(L_tot, C_tot, W0, theta, kx, delta0, 1/Nx)
    # ys, Deltas = curvatures.get_Delta_for_ky_by_recurrence(L_tot, C_tot, W0, theta, ky, Delta0, Ny)
    # fig, axes = plt.subplots(2)
    # plot_perturbations_by_list(axes, xs, deltas, ys, Deltas)
    # ori = create_perturbed_origami_by_list(theta, L0, C0, deltas, Deltas)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    Hs *= -1
    expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    fig.savefig(os.path.join(FIGURES_PATH, 'saddle-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-60, elev=20)
    lim = 5.5
    ori.dots.plot(ax, alpha=0.6, edge_alpha=0.0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    fig.savefig(os.path.join(FIGURES_PATH, 'saddle.pdf'), pad_inches=0.4)
    plt.show()
    plot_interactive(ori)


def plot_2D_sinusoid():
    """
    We reproduce the fig. 4.d from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    rows, cols = 30, 30
    kx = lambda t: 0.2 * np.sin(np.pi * t * 2.2)
    ky = lambda t: 0.2 * np.sin(np.pi * t * 2.2)

    L0 = 1
    C0 = 1.3
    W0 = 2.3
    theta = 0.99

    F0 = -0.2
    M0 = -0.7
    delta0 = F0
    Delta0 = M0 / L0

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = L0 * Ny, C0 * Nx

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    fig, axes = plt.subplots(2)
    plot_perturbations_by_list(axes, xs, deltas, ys, Deltas)

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    Hs *= -1
    expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    # fig.savefig(os.path.join(FIGURES_PATH, '2D-sinusoid-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-66, elev=29)
    ori.dots.plot(ax, alpha=0.6, edge_alpha=0.1)

    plotutils.set_axis_scaled(ax)

    # fig.savefig(os.path.join(FIGURES_PATH, '2D-sinusoid.pdf'), pad_inches=0.4)
    plt.show()
    plot_interactive(ori)


def plot_cap_different_curvatures():
    rows, cols = 40, 30

    L0 = 0.5
    C0 = 1
    W0 = 2.4
    theta = 1.1

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = L0 * Ny, C0 * Nx
    chi, xi = 1 / Nx, 1 / Ny

    # K = 0.007
    K = 0.0040
    K_factor = 1.3
    # kxs = [-np.sqrt(K) * 1.7, -np.sqrt(K) / 1.7, -np.sqrt(K)]
    kxs = [np.sqrt(K) * K_factor, np.sqrt(K) / K_factor]

    fig_all: Figure = plt.figure()
    ax_all: Axes3D = fig_all.add_subplot(111, projection='3d', elev=25, azim=-130)

    for i, kx in enumerate(kxs):
        ky = K / kx
        print(f"Plot for kx={kx}, ky={ky}")

        delta0 = 0.0
        Delta0 = -0.4

        delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
        Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

        # Make Fs symmetric
        # deltas = np.append(-deltas[1::][::-1], deltas)
        # C0 /= 2
        old_delta_func = delta_func

        def new_delta_func(x):
            above_sign = x > 0.5
            return above_sign * old_delta_func(2 * x - 1) - (1 - above_sign) * (old_delta_func(1 - 2 * x))

        delta_func = new_delta_func
        fig, axes = plt.subplots(2)
        Nx *= 2
        C0 /= 2
        plot_perturbations(axes, delta_func, Delta_func, Nx, Ny)

        ori = create_perturbed_origami(theta, Nx, Ny, L_tot, C_tot, delta_func, Delta_func)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        dots, indexes = ori.dots.dots, ori.dots.indexes
        dots = dots.astype('float64')
        r, c = indexes.shape
        # ax_all.plot_surface(
        #     dots[0, indexes[::2, ::2]].reshape((rows // 2 + 1, cols // 2 + 1)),
        #     dots[1, indexes[::2, ::2]].reshape((rows // 2 + 1, cols // 2 + 1)),
        #     dots[2, indexes[::2, ::2]].reshape((rows // 2 + 1, cols // 2 + 1)), alpha=0.3, linewidth=100)
        z_shift = i * 2.1
        ax_all.plot_surface(
            dots[0, :].reshape((r, c)),
            dots[1, :].reshape((r, c)),
            z_shift + dots[2, :].reshape((r, c)),
            alpha=0.3, linewidth=100)
        # wire = ax_all.plot_wireframe(
        #     dots[0, :].reshape((r, c)),
        #     dots[1, :].reshape((r, c)),
        #     z_shift + dots[2, :].reshape((r, c)),
        #     alpha=0.1, color='g', linewidth=2)
        l1 = ax_all.plot(
            dots[0, indexes[::2, 0]],
            dots[1, indexes[::2, 0]],
            dots[2, indexes[::2, 0]] + z_shift, '--g', linewidth=3)[0]
        l2 = ax_all.plot(
            dots[0, indexes[0, ::2]],
            dots[1, indexes[0, ::2]],
            dots[2, indexes[0, ::2]] + z_shift, '--r', linewidth=3)[0]
        l1.set_zorder(30)
        l2.set_zorder(30)

        should_plot_geometry = True
        if should_plot_geometry:
            geometry = origamimetric.OrigamiGeometry(ori.dots)
            Ks, Hs = geometry.get_curvatures_by_shape_operator()
            expected_K, expected_H = create_expected_curvatures_func(
                L_tot, C_tot, W0, theta, delta_func, Delta_func)
            fig, axes = compare_curvatures(Ks, Hs, expected_K, expected_H)
            axes[1, 0].set_title(f'K={kx * ky:.3f}')
            axes[1, 1].set_title(f'H={-(kx + ky) / 2:.3f}')
            # fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}-curvatures.svg'))

        should_plot_separately = False
        if should_plot_separately:
            fig: Figure = plt.figure()
            ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-50, elev=20)
            lim = 1.8
            ori.dots.plot_with_wireframe(ax, alpha=0.6)

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            # ax.set_zlim(-lim, lim)

            # fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}.svg'), pad_inches=0.4)
        # plot_interactive(ori)
    ax_all.set_aspect('equal')
    plotutils.set_3D_labels(ax_all)
    ax_all.dist = 8.0
    fig_all.subplots_adjust(left=0.4)
    # fig_all.savefig(os.path.join(FIGURES_PATH, 'cap-different-principal-curvatures.svg'), pad_inches=0.3)
    # fig_all.savefig(os.path.join(FIGURES_PATH, 'cap-different-principal-curvatures.pdf'), pad_inches=0.3)
    plt.show()


def plot_cap_different_curvatures_ugly():
    """
    Here we try to plot the same cap with constant Gaussian curvature
    but with different principal curvatures
    """
    rows, cols = 20, 26
    # rows, cols = 4, 4
    K = 0.01
    kxs = [-np.sqrt(K) / 2, -np.sqrt(K), -np.sqrt(K) * 2]

    rows = 26
    cols = 40
    theta = 0.8
    W0 = 2.1
    L0 = 0.2
    C0 = 0.1

    for i, kx in enumerate(kxs):
        ky = K / kx
        print(f"Plot for kx={kx}, ky={ky}")

        F0 = 0.0
        M0 = -0.1

        xs, Fs = curvatures.get_deltas_for_kx(L0, C0, W0, theta, kx, F0, 0)
        ys, MMs = curvatures.get_Delta_for_ky_by_recurrence(L0, C0, W0, theta, ky, M0, 0, rows // 2)

        # Make Fs symmetric
        Fs = np.append(-Fs[1::][::-1], Fs)

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(Fs, '.')
        axes[1].plot(ys, np.diff(MMs), '.')
        plt.show()

        F = create_F_from_list(Fs)
        MM = create_MM_from_list(MMs)

        ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
        fig, axes = compare_curvatures(Ks, Hs, expected_K, expected_H)
        axes[1, 0].set_title(f'K={kx * ky:.3f}')
        axes[1, 1].set_title(f'H={-(kx + ky) / 2:.3f}')
        fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}-curvatures.svg'))

        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-50, elev=20)
        lim = 1.8
        ori.dots.plot_with_wireframe(ax, alpha=0.6)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # ax.set_zlim(-lim, lim)

        fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}.svg'), pad_inches=0.4)
        plt.show()
        plot_interactive(ori)


def plot_cap_different_curvatures_large_angle():
    """
    Here we try to plot the same cap with constant Gaussian curvature
    but with different principal curvatures
    """
    rows, cols = 20, 26
    # rows, cols = 4, 4
    K = 0.005
    kxs = [-np.sqrt(K) / 2, -np.sqrt(K), -np.sqrt(K) * 2]

    rows = 40
    cols = 90
    theta = 0.8
    W0 = 3.0
    L0 = 0.2
    C0 = 0.5

    """
    L0 = 0.6
    C0 = 0.5
    W0 = 2.9
    theta = 1.4
    """

    for i, kx in enumerate(kxs):
        ky = K / kx
        print(f"Plot for kx={kx}, ky={ky}")

        F0 = 0.0
        M0 = -0.1

        xs, Fs = curvatures.get_deltas_for_kx(L0, C0, W0, theta, kx, F0, 0)
        ys, MMs = curvatures.get_Delta_for_ky_by_recurrence(L0, C0, W0, theta, ky, M0, 0, rows // 2)
        print(ys, MMs)

        # Make Fs symmetric
        Fs = np.append(-Fs[1::][::-1], Fs)

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(Fs, '.')
        axes[1].plot(ys, np.diff(MMs), '.')
        plt.show()

        F = create_F_from_list(Fs)
        MM = create_MM_from_list(MMs)

        ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
        fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
        # fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}-curvatures.svg'))

        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-50, elev=20)
        lim = 5.0
        ori.dots.plot_with_wireframe(ax, alpha=0.6)

        # ax.set_xlim(-lim, lim)
        # ax.set_ylim(-lim, lim)
        # ax.set_zlim(-lim, lim)

        # fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}.svg'), pad_inches=0.4)
        plt.show()
        plot_interactive(ori)


def plot_cone_like():
    logutils.enable_logger()
    logging.getLogger('origami').setLevel(logging.WARNING)
    logging.getLogger('origami.alternating').setLevel(logging.DEBUG)

    L0 = 2
    C0 = 2
    chi = 1 / 30
    xi = 1 / 30
    theta = 1.40
    W0 = 2.4

    s = 1.0
    t = s

    # kx = lambda x: s * (0.4 < x < 0.6)
    # ky = lambda y: t * (0.4 < y < 0.6)

    a = 0.07
    kx = lambda x: s / (a * np.sqrt(np.pi)) * np.exp(-((x - 0.5) / a) ** 2)
    ky = lambda y: t / (a * np.sqrt(np.pi)) * np.exp(-((y - 0.5) / a) ** 2)

    delta0 = -0.5
    DeltaL0 = -0.2

    xs, deltas = curvatures.get_deltas_for_kx(L0, C0, W0, theta, kx, delta0, chi)
    ys, DeltaLs = curvatures.get_Deltas_for_ky(L0, C0, W0, theta, ky, DeltaL0, xi)

    plot_perturbations_by_list(xs, deltas, ys, DeltaLs)

    ori = create_perturbed_origami_by_list(theta, L0, C0, deltas, DeltaLs)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()

    fig, ax = plt.subplots()
    imshow_with_colorbar(fig, ax, Ks, 'Ks')

    origamiplots.plot_interactive(ori)


def main():
    # plot_vase()
    plot_spherical_cap()
    # plot_saddle()
    # plot_2D_sinusoid()
    plot_cap_different_curvatures()
    # plot_cap_different_curvatures_ugly()
    # plot_cone_like()


if __name__ == '__main__':
    main()
