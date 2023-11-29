"""
Here we show some simple RFFQM origami that we can design by controlling
the principal curvatures.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import origami
from origami import origamimetric
from origami.origamiplots import plot_interactive
from origami.plotsandcalcs.alternating import betterapproxcurvatures
from origami.plotsandcalcs.alternating.betterapprox import compare_curvatures
from origami.plotsandcalcs.alternating.betterapproxcurvatures import create_expected_curvatures_func
from origami.plotsandcalcs.alternating.utils import create_F_from_list, create_MM_from_list, create_perturbed_origami
from origami.utils import plotutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM', 'Figures', 'design-examples')


def plot_vase():
    """
    We reproduce the fig. 4.a.1 from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    rows, cols = 40, 30
    kx_func = lambda t: 7 * 1 / (80 / 2)
    # ky_func = lambda t: 0.2 * (t - 10) / (rows / 2)
    ky_func = lambda t: -0.2 * np.tanh((t - rows / 4) * 3)

    F0 = 0.3
    M0 = 0.2

    L0 = 0.5
    C0 = 1
    W0 = 2.5
    theta = 1.0

    xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx_func, F0, 0, cols // 2)
    ys, MMs = betterapproxcurvatures.get_MM_for_ky(L0, C0, W0, theta, ky_func, M0, 0, rows // 2)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(xs, Fs, '.')
    axes[1].plot(ys, np.diff(MMs), '.')

    F = create_F_from_list(Fs)
    MM = create_MM_from_list(MMs)

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'vase-curvatures.svg'))
    fig.savefig(os.path.join(FIGURES_PATH, 'vase-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=50, elev=30)
    lim = 4.0
    _, wire = ori.dots.plot(ax, alpha=0.6)
    wire.set_alpha(0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # fig.tight_layout(rect=(0.3, 0, 0.7, 1))
    fig.savefig(os.path.join(FIGURES_PATH, 'vase.svg'), pad_inches=0.4)
    fig.savefig(os.path.join(FIGURES_PATH, 'vase.pdf'), pad_inches=0.4)
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

    F0 = 0.2
    M0 = 0.5

    L0 = 1.0
    C0 = 1
    W0 = 2.4
    theta = 1.1

    xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx, F0, 0, cols // 2)
    ys, MMs = betterapproxcurvatures.get_MM_for_ky(L0, C0, W0, theta, ky, M0, 0, rows // 2)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(xs, Fs, '.')
    axes[1].plot(ys, np.diff(MMs), '.')
    # plt.show()

    F = create_F_from_list(Fs)
    MM = create_MM_from_list(MMs)

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
    ori.set_gamma(0)

    # fig, _ = plot_crease_pattern(ori)
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.svg'))
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.png'))

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical_cap-curvatures.svg'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-50, elev=20)
    lim = 5.0
    _, wire = ori.dots.plot(ax, alpha=0.6)
    wire.set_alpha(0.0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.svg'), pad_inches=0.4)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.png'), pad_inches=0.4)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.pdf'), pad_inches=0.4)
    plt.show()
    plot_interactive(ori)


def plot_saddle():
    """
    We reproduce the fig. 4.c from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    rows, cols = 20, 30
    kx = 0.08
    ky = -0.08

    F0 = 0.18
    M0 = -1.3

    L0 = 1.5
    C0 = 1.2
    W0 = 2.4
    theta = 1.0

    xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx, F0, 0, cols // 2)
    ys, MMs = betterapproxcurvatures.get_MM_for_ky(L0, C0, W0, theta, ky, M0, 0, rows // 2)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(xs, Fs, '.')
    axes[1].plot(ys, np.diff(MMs), '.')

    F = create_F_from_list(Fs)
    MM = create_MM_from_list(MMs)

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    fig.savefig(os.path.join(FIGURES_PATH, 'saddle-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-60, elev=20)
    lim = 5.5
    _, wire = ori.dots.plot(ax, alpha=0.6)
    wire.set_alpha(0)

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
    kx = lambda t: 0.15 * np.sin(np.pi * t / (cols / 4))
    ky = lambda t: 0.15 * np.sin(np.pi * t / (rows / 4))

    F0 = 0.3
    M0 = 0.5

    L0 = 1
    C0 = 1.3
    W0 = 2.3
    theta = 0.99

    xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx, F0, 0, cols // 2)
    ys, MMs = betterapproxcurvatures.get_MM_for_ky(L0, C0, W0, theta, ky, M0, 0, rows // 2)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(xs, Fs, '.')
    axes[1].plot(ys, np.diff(MMs), '.')
    # plt.show()

    F = create_F_from_list(Fs)
    MM = create_MM_from_list(MMs)

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    fig.savefig(os.path.join(FIGURES_PATH, '2D-sinusoid-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-66, elev=29)
    lim = 6.0
    _, wire = ori.dots.plot(ax, alpha=0.6)
    wire.set_alpha(0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    fig.savefig(os.path.join(FIGURES_PATH, '2D-sinusoid.pdf'), pad_inches=0.4)
    plt.show()
    plot_interactive(ori)


def plot_cap_different_curvatures():
    rows, cols = 20, 30
    kx = 0.10
    ky = 0.10

    L0 = 1.0
    C0 = 1
    W0 = 2.4
    theta = 1.1

    K = 0.007
    K_factor = 1.6
    # kxs = [-np.sqrt(K) * 1.7, -np.sqrt(K) / 1.7, -np.sqrt(K)]
    kxs = [-np.sqrt(K) * K_factor, -np.sqrt(K) / K_factor]

    fig_all: Figure = plt.figure()
    ax_all: Axes3D = fig_all.add_subplot(111, projection='3d', elev=-150, azim=130)

    for i, kx in enumerate(kxs):
        ky = K / kx
        print(f"Plot for kx={kx}, ky={ky}")

        F0 = 0.0
        M0 = -0.5

        xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx, F0, 0, cols // 4 + 1)
        ys, MMs = betterapproxcurvatures.get_MM_for_ky_by_recurrence(L0, C0, W0, theta, ky, M0, 0, rows // 2)

        # Make Fs symmetric
        Fs = np.append(-Fs[1::][::-1], Fs)

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(Fs, '.')
        axes[1].plot(ys, np.diff(MMs), '.')
        # plt.show()

        F = create_F_from_list(Fs)
        MM = create_MM_from_list(MMs)

        ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        dots, indexes = ori.dots.dots, ori.dots.indexes
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

        should_plot_geometry = False
        if should_plot_geometry:
            geometry = origamimetric.OrigamiGeometry(ori.dots)
            Ks, Hs = geometry.get_curvatures_by_shape_operator()
            expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, theta, F, MM)
            fig, axes = compare_curvatures(Ks, Hs, expected_K, expected_H)
            axes[1, 0].set_title(f'K={kx * ky:.3f}')
            axes[1, 1].set_title(f'H={-(kx + ky) / 2:.3f}')
            # fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}-curvatures.svg'))

        should_plot_separately = False
        if should_plot_separately:
            fig: Figure = plt.figure()
            ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-50, elev=20)
            lim = 1.8
            ori.dots.plot(ax, alpha=0.6)

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            # ax.set_zlim(-lim, lim)

            # fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}.svg'), pad_inches=0.4)
        # plot_interactive(ori)
    # ax_all.set_aspect('equal')
    plotutils.set_3D_labels(ax_all)
    ax_all.dist = 8.0
    fig_all.subplots_adjust(left=0.4)
    fig_all.savefig(os.path.join(FIGURES_PATH, 'cap-different-principal-curvatures.svg'), pad_inches=0.3)
    fig_all.savefig(os.path.join(FIGURES_PATH, 'cap-different-principal-curvatures.png'), pad_inches=0.3)
    fig_all.savefig(os.path.join(FIGURES_PATH, 'cap-different-principal-curvatures.pdf'), pad_inches=0.3)
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

        xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx, F0, 0, cols // 4 + 1)
        ys, MMs = betterapproxcurvatures.get_MM_for_ky_by_recurrence(L0, C0, W0, theta, ky, M0, 0, rows // 2)

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
        ori.dots.plot(ax, alpha=0.6)

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

        xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx, F0, 0, cols // 4 + 1)
        ys, MMs = betterapproxcurvatures.get_MM_for_ky_by_recurrence(L0, C0, W0, theta, ky, M0, 0, rows // 2)
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
        ori.dots.plot(ax, alpha=0.6)

        # ax.set_xlim(-lim, lim)
        # ax.set_ylim(-lim, lim)
        # ax.set_zlim(-lim, lim)

        # fig.savefig(os.path.join(FIGURES_PATH, f'cap-{i}.svg'), pad_inches=0.4)
        plt.show()
        plot_interactive(ori)


def main():
    plot_vase()
    plot_spherical_cap()
    plot_saddle()
    plot_2D_sinusoid()
    # plot_cap_different_curvatures()
    # plot_cap_different_curvatures_ugly()


if __name__ == '__main__':
    main()
