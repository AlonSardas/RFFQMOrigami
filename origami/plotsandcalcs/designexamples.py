"""
Here we show some simple RFFQM origami that we can design by controlling
the principal curvatures.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.axes import Axes
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

import origami
from origami import origamimetric, origamiplots, quadranglearray
from origami.alternatingpert import curvatures
from origami.alternatingpert.curvatures import create_expected_curvatures_func
from origami.alternatingpert.utils import create_perturbed_origami, \
    plot_perturbations_by_list, create_perturbed_origami_by_list, get_pert_list_by_func, plot_perturbations
from origami.origamiplots import plot_interactive
from origami.plotsandcalcs import articleillustrations
from origami.plotsandcalcs.alternating.betterapprox import compare_curvatures
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
    Delta0 = -0.6

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = L0 * Ny, C0 * Nx

    kx_func = lambda t: 7 * 1 / (80 / 2)
    # ky_func = lambda t: -0.2 * np.tanh((t - rows / 4) * 3)
    # kx_func = lambda t: -1 / C_tot
    # ky_func = lambda t: +0.25 * np.tanh((t - 0.5) * 3)
    f_y = lambda t: 0.02 * np.sin(np.pi * (t - 0.5))
    eps = 0.001
    df_dy = lambda t: (f_y(t + eps) - f_y(t - eps)) / (2 * eps)
    ddf_dy2 = lambda t: (df_dy(t + eps) - df_dy(t - eps)) / (2 * eps)
    k_extra_factor = 1
    ky_func = lambda t: k_extra_factor * ddf_dy2(t) / (1 + df_dy(t) ** 2) ** (3 / 2)

    should_plot_ky = False
    if should_plot_ky:
        fig, ax = plt.subplots()
        ys = np.linspace(0, 1)
        ax.plot(ys, ky_func(ys))
        ax.plot(ys, np.cos(ys * np.pi) * 0.2, '--')
        plt.show()
    # return

    print(ky_func(0.2), ky_func(0.5), ky_func(1))
    # ky_func = lambda t: +0.7 * (t - 0.5)

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx_func, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky_func, Delta0)

    xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    # fig, axes = plt.subplots(2)
    # plot_perturbations_by_list(axes, xs, deltas, ys, Deltas)

    ori = create_perturbed_origami_by_list(theta, L0, C0, deltas, Deltas)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    should_plot_geometry = False
    if should_plot_geometry:
        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
        fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
        fig.tight_layout()
        # fig.savefig(os.path.join(FIGURES_PATH, 'vase-curvatures.svg'))
        # fig.savefig(os.path.join(FIGURES_PATH, 'vase-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=39, azim=-140,
                                 computed_zorder=False)

    smooth_limits_x = 4.2
    smooth_limits_y = 6
    xs = np.linspace(-smooth_limits_x, smooth_limits_x, 50)
    ys = np.linspace(-smooth_limits_y, smooth_limits_y, 50)
    xs, ys = np.meshgrid(xs, ys)
    R0 = 1 / kx_func(0)
    Rs = R0 - 70 * f_y((ys + smooth_limits_y) / (2 * smooth_limits_y)) + 0.0
    # _, temp_ax = plt.subplots()
    # temp_ax.plot(Rs[:, 0], '.')
    # temp_ax.plot(f_y(np.linspace(0, 1)), '.')
    # print(Rs)
    sphere_Z_shift = 0.3
    zs = np.sqrt(Rs ** 2 - xs ** 2) - R0 + sphere_Z_shift
    zs += 0.3 * ys
    ax.plot_surface(xs, ys, zs, linewidth=0, rstride=1, cstride=1, zorder=-20)

    # ori.dots.plot(ax, alpha=0.6, edge_alpha=0.8, edge_color='k', panel_color='C1')
    # ori.dots.plot(ax, alpha=1, edge_alpha=1, edge_color='k', panel_color='C1')
    quadranglearray.plot_panels_manual_zorder(ori.dots, ax, alpha=1, edge_alpha=1, edge_color='k', panel_color='C1')

    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()

    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(0.95, 0.65)
    # new_bbox = Bbox.from_bounds(new_bbox.x0 + 0.05, new_bbox.y0, new_bbox.width, new_bbox.height - 0.5)
    fig.savefig(os.path.join(FIGURES_PATH, 'vase.svg'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'vase.pdf'), bbox_inches=new_bbox)
    plt.show()

    # plot_interactive(ori)


PANELS_ALPHA = 0.9
PANELS_COLOR = articleillustrations.PANELS_COLOR
EDGE_COLOR = 'k'
EDGE_ALPHA = 0.8
SMOOTH_SURF_ALPHA = 0.1
SMOOTH_SURF_COLOR = 'C0'
SMOOTH_EDGE_COLOR = 'C0'
SMOOTH_EDGE_ALPHA = 0.6
light_source = LightSource(azdeg=315 - 90 - 90, altdeg=45)


def plot_spherical_cap():
    """
    We reproduce the fig. 4.b from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    rows, cols = 20, 30
    # kx_func = lambda t: 3.5 * 1 / (80 / 2)
    # ky_func = lambda t: 0.2 * (t - 10) / (rows / 2)
    # ky_func = lambda t: -0.2 * np.tanh((t - rows / 4) * 3)
    kx = -0.10
    ky = -0.10

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=-155, azim=-61,
                                 computed_zorder=False)

    smooth_limits_x, smooth_limits_y = 6.3, 5.7
    xs = np.linspace(-smooth_limits_x, smooth_limits_x, 15)
    ys = np.linspace(-smooth_limits_y, smooth_limits_y, 15)
    xs, ys = np.meshgrid(xs, ys)
    R = 1 / kx
    sphere_Z_shift = +6.3
    # sphere_Z_shift = 1
    sphere_zs = -np.sqrt(R ** 2 - xs ** 2 - ys ** 2) + sphere_Z_shift
    _plot_smooth(ax, xs, ys, sphere_zs)

    L0 = 1.0
    C0 = 1
    W0 = 2.4
    theta = 1.1

    F0 = +0.2
    M0 = +0.5
    Delta0 = M0 / L0
    delta0 = F0

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = L0 * Ny, C0 * Nx

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    should_plot_pert = False
    if should_plot_pert:
        pert_fig, pert_axes = plt.subplots(2)
        plot_perturbations_by_list(pert_axes, xs, deltas, ys, Deltas)

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)
    ori.set_gamma(0)

    # fig, _ = plot_crease_pattern(ori)
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.svg'))
    # fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.png'))

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    should_plot_geometry = False
    if should_plot_geometry:
        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        Hs *= -1
        expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
        geo_fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
        fig.savefig(os.path.join(FIGURES_PATH, 'spherical_cap-curvatures.svg'))

    quads = ori.dots
    quads.dots[2, :] -= np.max(quads.dots[2, :])
    quads.plot(ax, panel_color=PANELS_COLOR,
               alpha=PANELS_ALPHA, edge_alpha=EDGE_ALPHA, edge_color='k', lightsource=light_source)

    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()

    # fig.tight_layout(h_pad=-5)
    # fig.subplots_adjust(0, 0, 1, 1)
    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(0.9, 0.65)
    new_bbox = Bbox.from_bounds(new_bbox.x0 + 0.05, new_bbox.y0, new_bbox.width, new_bbox.height - 0.5)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.svg'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.pdf'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.png'), bbox_inches=new_bbox, dpi=300)
    plt.show()
    plot_interactive(ori)


def plot_saddle():
    """
    We reproduce the fig. 4.c from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    L_tot, C_tot = 15, 15

    kappa = 0.1
    kx = kappa
    ky = -kappa

    # Nx, Ny = cols // 2, rows // 2
    Nx, Ny = 11, 13
    W0 = 2.2
    theta = 1.2

    delta0 = -0.4
    Delta0 = 0.5

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    should_plot_perts = False
    if should_plot_perts:
        pert_fig, pert_axes = plt.subplots(2)
        plot_perturbations(pert_axes, delta_func, Delta_func, Nx, Ny)

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    should_plot_geometry = False
    if should_plot_geometry:
        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        Hs *= -1
        expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
        fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
        fig.savefig(os.path.join(FIGURES_PATH, 'saddle-curvatures.pdf'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=-144, azim=-65,
                                 computed_zorder=False)

    smooth_limits_x, smooth_limits_y = 8.6, 7.2
    xs = np.linspace(-smooth_limits_x, smooth_limits_x, 10)
    ys = np.linspace(-smooth_limits_y, smooth_limits_y, 10)
    xs, ys = np.meshgrid(xs, ys)
    R = 1 / np.abs(kx)
    saddle_Z_shift = -0.74
    saddle_zs = -np.sqrt(R ** 2 + xs ** 2 - ys ** 2) + R + saddle_Z_shift
    _plot_smooth(ax, xs, ys, saddle_zs)

    ori.dots.plot(ax, panel_color=PANELS_COLOR, alpha=PANELS_ALPHA,
                  edge_alpha=EDGE_ALPHA,
                  edge_color=EDGE_COLOR, lightsource=light_source)

    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()

    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(0.95, 0.60)
    new_bbox = Bbox.from_bounds(new_bbox.x0 + 0.4, new_bbox.y0, new_bbox.width - 0.4, new_bbox.height)
    fig.savefig(os.path.join(FIGURES_PATH, 'saddle.pdf'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'saddle.svg'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'saddle.png'), bbox_inches=new_bbox, dpi=300)
    plt.show()
    # plot_interactive(ori)


def _plot_smooth(ax, xs, ys, zs):
    surf = ax.plot_surface(xs, ys, zs, color=SMOOTH_SURF_COLOR,
                           alpha=SMOOTH_EDGE_ALPHA, linewidth=1, rstride=1, cstride=1, zorder=20,
                           edgecolor=SMOOTH_EDGE_COLOR,
                           lightsource=light_source)
    surf.set_edgecolor(surf.get_edgecolor())
    surf.set_alpha(SMOOTH_SURF_ALPHA)
    return surf


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

    smooth_limits_x = 7.8
    smooth_limits_y = 9.5
    xs = np.linspace(-smooth_limits_x, smooth_limits_x, 50)
    ys = np.linspace(-smooth_limits_y, smooth_limits_y, 50)
    xs, ys = np.meshgrid(xs, ys)
    Z_shift = -0.2
    A = 1.06
    By = 1.15
    Bx = 1.10
    smooth_zs = Z_shift - A * (np.sin(np.pi * xs / smooth_limits_x * Bx) + np.sin(np.pi * ys / smooth_limits_y * By))

    should_plot_smooth_geometry = False
    if should_plot_smooth_geometry:
        quads = quadranglearray.from_mesh_gird(xs, ys, smooth_zs)
        geometry = origamimetric.OrigamiGeometry(quads)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        Hs *= -1
        fig, _ = compare_curvatures(Ks, Hs, None, None)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=36, azim=-113, computed_zorder=False)

    ax.plot_surface(xs, ys, smooth_zs, linewidth=0, rstride=1, cstride=1, zorder=-20)

    ori.dots.plot(ax, alpha=0.8, edge_alpha=0.8, edge_color='k', panel_color='C1')

    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()

    _save_fig(fig, '2D-sinusoid', 0.8, 0.6, 0.2)
    plt.show()
    # plot_interactive(ori)


def _save_fig(fig, name, expand_w=1.0, expand_h=1.0,
              extra_x0=0.0, extra_y0=0.0, extra_width=0.0, extra_height=0.0):
    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(expand_w, expand_h)
    new_bbox = Bbox.from_bounds(new_bbox.x0 + extra_x0, new_bbox.y0 + extra_y0,
                                new_bbox.width + extra_width, new_bbox.height + extra_height)
    fig.savefig(os.path.join(FIGURES_PATH, f'{name}.pdf'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, f'{name}.svg'), bbox_inches=new_bbox)


def plot_cap_different_curvatures():
    Nx, Ny = 10, 14

    W0 = 2.4
    theta = 1.2

    L_tot, C_tot = 6, 6

    K = 0.05
    K_factor = 1.4
    # kxs = [-np.sqrt(K) * 1.7, -np.sqrt(K) / 1.7, -np.sqrt(K)]
    kxs = [np.sqrt(K) * K_factor, np.sqrt(K) / K_factor]

    fig_all: Figure = plt.figure()
    ax_all: Axes3D = fig_all.add_subplot(111, projection='3d', elev=25, azim=-130, computed_zorder=True)

    pert_fig, pert_axes = plt.subplots(2)

    Delta0s = [-0.6, -0.6]

    for i, kx in enumerate(kxs):
        ky = K / kx
        print(f"Plot for kx={kx}, ky={ky}")

        # delta0 = 0.0
        Delta0 = Delta0s[i]

        # Make delta symmetric
        delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot / 2, W0, theta, kx, 0)
        new_delta0 = -delta_func(1)
        delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, new_delta0)

        Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

        # plot_perturbations(pert_axes, delta_func, Delta_func, Nx, Ny)

        ori = create_perturbed_origami(theta, Nx, Ny, L_tot, C_tot, delta_func, Delta_func)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        # np.savetxt(f'Pattern{i+1}.txt', ori.dots.dots)
        # np.savetxt(f'Pattern{i+1}-shape.txt', (ori.dots.rows, ori.dots.cols))

        # r, c = indexes.shape
        z_shift = i * 2.1
        quads = ori.dots
        quads.dots[2, :] += z_shift
        # quads.plot(ax_all, edge_color='k', edge_width=1, alpha=0.9)
        ax_all.computed_zorder = False
        quadranglearray.plot_panels_manual_zorder(quads, ax_all, panel_color=f'C{i}', edge_color='k', edge_width=1,
                                                  alpha=0.9)
        # ax_all.plot_surface(
        #     dots[0, :].reshape((r, c)),
        #     dots[1, :].reshape((r, c)),
        #     z_shift + dots[2, :].reshape((r, c)),
        #     alpha=1.0, linewidth=4)
        dots, indexes = quads.dots, quads.indexes
        l1 = ax_all.plot(
            dots[0, indexes[::2, 0]],
            dots[1, indexes[::2, 0]],
            dots[2, indexes[::2, 0]], '--', linewidth=5, color='#26c423')[0]
        l2 = ax_all.plot(
            dots[0, indexes[0, ::2]],
            dots[1, indexes[0, ::2]],
            dots[2, indexes[0, ::2]], '--r', linewidth=5)[0]
        l1.set_zorder(30)
        l2.set_zorder(30)

        should_plot_geometry = False
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
    plotutils.set_axis_scaled(ax_all)
    plotutils.set_3D_labels(ax_all)
    ax_all.set_zlabel('')
    plotutils.remove_tick_labels(ax_all)
    # ax_all.dist = 8.0
    # fig_all.subplots_adjust(left=0.4)

    # fig_all.tight_layout()
    # bbox = fig_all.get_tightbbox()
    plotutils.save_fig_cropped(fig_all, os.path.join(FIGURES_PATH, 'cap-different-principal-curvatures.pdf'),
                               0.95, 0.85, translate_x=0.1, translate_y=-0.1)
    plt.show()


def plot_wavy():
    rows, cols = 14, 18
    L0 = 8
    C0 = 8.5
    chi = 1 / cols * 2
    xi = 1 / rows * 2
    L_tot = L0 / xi
    C_tot = C0 / chi

    kx_func = lambda t: 1 / C_tot * 4.0 * np.tanh(-(t - 0.5) * 3)
    ky_func = lambda t: 1 / L_tot * 1.2

    delta0 = 0.40
    Delta0 = 0.7

    W0 = -2.1
    theta = 1.1

    xs, deltas = curvatures.get_deltas_for_kx(L_tot, C_tot, W0, theta, kx_func, delta0, chi)
    ys, Deltas = curvatures.get_Deltas_for_ky(L_tot, C_tot, W0, theta, ky_func, Delta0, xi)
    should_plot_perts = False
    if should_plot_perts:
        fig, axes = plt.subplots(2)
        plot_perturbations_by_list(axes, xs, deltas, ys, Deltas)

    ori = create_perturbed_origami_by_list(
        theta, L0, C0, deltas, Deltas)

    ori.set_gamma(ori.calc_gamma_by_omega(-W0))

    fig: Figure = plt.figure()
    light_source = LightSource(azdeg=315, altdeg=45)
    plot_option = 1
    if plot_option == 1:
        ax: Axes3D = fig.add_subplot(111, projection='3d', elev=-151, azim=119,
                                     computed_zorder=True)
        ori.dots.plot(ax, panel_color=PANELS_COLOR, alpha=PANELS_ALPHA,
                      edge_alpha=EDGE_ALPHA,
                      edge_color=EDGE_COLOR, lightsource=light_source)

    elif plot_option == 2:
        ax: Axes3D = fig.add_subplot(111, projection='3d', elev=-162, azim=130,
                                     computed_zorder=False)
        quadranglearray.plot_panels_manual_zorder(
            ori.dots, ax, panel_color=PANELS_COLOR, alpha=PANELS_ALPHA,
            edge_alpha=EDGE_ALPHA,
            edge_color=EDGE_COLOR, lightsource=light_source,
            z_shift=-0.01)

    quads = ori.dots
    dots, indexes = quads.dots, quads.indexes
    dots = dots.astype('float64')
    dots = dots[:, indexes[::2, ::2]]
    left, right = np.min(dots[0, :, :]), np.max(dots[0, :, :])
    bottom, top = np.min(dots[1, :, :]), np.max(dots[1, :, :])
    pad_x, pad_y = 1.5, 1.8
    left, right = left - pad_x, right + pad_x + 0.1
    bottom, top = bottom - pad_y, top + pad_y
    # ax.scatter3D(dots[0, :], dots[1, :], dots[2, :])

    interp = interpolate.SmoothBivariateSpline(
        dots[0, :, :].flatten(),
        dots[1, :, :].flatten(),
        dots[2, :, :].flatten())

    xs = np.linspace(left, right, 24)
    ys = np.linspace(bottom, top, 24)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = interp(Xs, Ys, grid=False)
    Zs -= 3.5
    _plot_smooth(ax, Xs, Ys, Zs)

    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()

    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(0.95, 0.50)
    new_bbox = Bbox.from_bounds(new_bbox.x0 + 0.3, new_bbox.y0, new_bbox.width - 0.4, new_bbox.height)
    fig.savefig(os.path.join(FIGURES_PATH, 'pos-neg.pdf'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'pos-neg.svg'), bbox_inches=new_bbox)

    # plot_interactive(ori)

    plt.show()


def plot_periodic():
    """
    We reproduce the fig. 4.c from:
    https://doi.org/10.1016/j.ijsolstr.2021.111224.
    """
    C_tot, L_tot = 14, 16

    Nx, Ny = 46, 56

    s, t = 13 / C_tot, 12 / L_tot
    # s, t = 8 / C_tot, 8 / L_tot

    W0 = 2.4
    theta = 1.3

    delta0 = -0.0
    Delta0 = -0.0

    ox, oy = 3.0, 3.0
    kx_func = lambda x: s * np.cos(2 * np.pi * x * ox)
    ky_func = lambda y: t * np.cos(2 * np.pi * y * oy)
    shifted_kx_func = lambda x: kx_func((x + 1 / (4 * ox)) % 1)
    shifted_ky_func = lambda y: ky_func((y + 1 / (4 * oy)) % 1)

    delta_func_orig = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx_func, delta0)
    Delta_func_orig = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky_func, Delta0)

    def shifted_delta_func(x):
        return delta_func_orig((x + 1 / (4 * ox)) % 1)

    def shifted_Delta_func(y):
        return Delta_func_orig((y + 1.18 / (4 * oy)) % 1)

    delta_func, Delta_func = shifted_delta_func, shifted_Delta_func
    # delta_func, Delta_func = delta_func_orig, Delta_func_orig

    # fig, axes = plt.subplots(2)
    # plot_perturbations(axes, delta_func, Delta_func, Nx, Ny)

    # Nx = Nx // 4
    # Ny = Ny // 4
    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)

    # plot_flat(ori, shifted_delta_func(0))
    # return

    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    Hs *= -1  # I'm still not sure where we were inconsistent about the orientation...
    # expected_K, expected_H = create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta_func, Delta_func)
    target_K = lambda x, y: kx_func(x) * ky_func(y)
    expected_H = lambda x, y: -1 / 2 * (kx_func(x) + ky_func(y))

    target_K = lambda x, y: shifted_kx_func(x) * shifted_ky_func(y)
    target_H = lambda x, y: 1 / 2 * (shifted_kx_func(x) + shifted_ky_func(y))

    # fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    # fig.savefig(os.path.join(FIGURES_PATH, 'periodic-curvatures.pdf'))

    def _compare_curvatures():
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 30
        fig, axes = plt.subplots(1, 2, sharey=True, layout='compressed')
        len_ys, len_xs = Ks.shape
        xs, ys = np.arange(Nx - 1) / Nx, np.arange(Ny - 1) / Ny
        Xs, Ys = np.meshgrid(xs, ys)
        target_Ks = target_K(Xs, Ys)

        vmin = min(np.min(target_Ks), np.min(Ks))
        vmax = max(np.max(target_Ks), np.max(Ks))

        ax: Axes = axes[0]
        im = ax.imshow(target_Ks, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title('Target K')
        ax.set_xlabel('j')
        ax.set_ylabel('i')

        ax: Axes = axes[1]
        im = ax.imshow(Ks, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title('Actual K')
        ax.set_xlabel('j')

        fig.colorbar(im, ax=axes, location='bottom')
        fig.savefig(os.path.join(FIGURES_PATH, 'periodic-K-comparison.svg'))
        fig.savefig(os.path.join(FIGURES_PATH, 'periodic-K-comparison.pdf'))

        fig, axes = plt.subplots(1, 2, sharey=True, layout='compressed')
        target_Hs = target_H(Xs, Ys)

        vmin = min(np.min(target_Hs), np.min(Hs))
        vmax = max(np.max(target_Hs), np.max(Hs))

        ax: Axes = axes[0]
        im = ax.imshow(target_Hs, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title('Target H')
        ax.set_xlabel('j')
        ax.set_ylabel('i')

        ax: Axes = axes[1]
        im = ax.imshow(Hs, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title('Actual H')
        ax.set_xlabel('j')

        fig.colorbar(im, ax=axes, location='bottom')

        fig.savefig(os.path.join(FIGURES_PATH, 'periodic-H-comparison.svg'))
        fig.savefig(os.path.join(FIGURES_PATH, 'periodic-H-comparison.pdf'))
        plt.show()

    # _compare_curvatures()

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=27, azim=-124)

    # ori.dots.plot(ax, alpha=1.0, edge_alpha=0.5, edge_width=0.2, edge_color='k')
    ax.computed_zorder = False
    quadranglearray.plot_panels_manual_zorder(ori.dots, ax, alpha=1.0, edge_alpha=0.5, edge_width=0.2, edge_color='k')

    plotutils.set_axis_scaled(ax)
    # plotutils.set_3D_labels(ax)
    ax.set_axis_off()

    fig.tight_layout()
    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(sw=0.95, sh=0.5)
    fig.savefig(os.path.join(FIGURES_PATH, 'periodic.png'), bbox_inches=new_bbox, transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'periodic.pdf'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'periodic.svg'), bbox_inches=new_bbox)
    return
    plt.show()


def plot_flat(ori, rotate_angle):
    fig, ax = origamiplots.plot_crease_pattern(ori, 0, rotate_angle)
    fig.set_layout_engine("constrained")
    zoomed_ax = inset_axes(ax, width='85%', height='85%', )
    make_axis_frame_circular(zoomed_ax)

    dots, indexes = ori.dots.dots, ori.dots.indexes
    zoomed_cells = dots[:, indexes[0:17, 0:17].flat]
    zoomed_quads = quadranglearray.QuadrangleArray(zoomed_cells, 17, 17)
    origamiplots.draw_creases(zoomed_quads, 0, zoomed_ax)

    # unit_cell_x, unit_cell_y = 1,1
    unit_cell_x, unit_cell_y = 4, 4
    dots, indexes = ori.dots.dots, ori.dots.indexes
    cell_dots = dots[:, indexes[unit_cell_y:unit_cell_y + 3, unit_cell_x:unit_cell_x + 3].flat]
    cell_quads = quadranglearray.QuadrangleArray(cell_dots, 3, 3)
    creases = origamiplots.draw_creases(cell_quads, 1, zoomed_ax)
    for line in creases:
        line.set_linewidth(4)

    x0, y0, width, height = 0.7, 0.7, 3, 3
    zoomed_ax.set_xlim(x0, x0 + width)
    zoomed_ax.set_ylim(y0, y0 + width)
    # pp, l1, l2 = mark_inset(ax, zoomed_ax, loc1=2, loc2=4, fc="none", ec="k", linewidth=2)
    # pp.remove()
    # l1.remove()
    # l2.remove()
    # bbox = pp.bbox
    # print(bbox.extents)
    # print(bbox)
    # print((bbox.x0, bbox.y0), bbox.width, bbox.height)
    # print(ax.transData, zoomed_ax.viewLim)
    # print(ax.transData.extents, zoomed_ax.viewLim.extents)
    # rect = TransformedBbox(zoomed_ax.viewLim, ax.transData)
    # print(ax.transData.inverted().transform(rect))

    # plt.draw()  # This is necessary for the zoomed axis to
    zoomed_ax.redraw_in_frame()  # This is necessary for the zoomed axis to

    # print(ax.transData.inverted().transform(ax.transAxes.transform(zoomed_ax.get_position())))
    print(zoomed_ax.bbox.extents)
    print(ax.transData.inverted().transform(zoomed_ax.bbox))
    (zoomed_x0, zoomed_y0), (zoomed_x1, zoomed_y1) = ax.transData.inverted().transform(
        zoomed_ax.patch.get_extents())
    zoomed_width = zoomed_x1 - zoomed_x0
    zoomed_height = zoomed_y1 - zoomed_y0

    linewidth = 5

    my_bbox = FancyBboxPatch((x0, y0), width, height,
                             boxstyle=f"round,pad=-0.0040,rounding_size={0.2 * width}",
                             ec="red", fc="dodgerblue", clip_on=False, lw=linewidth,
                             mutation_aspect=1,
                             )
    ax.add_patch(my_bbox)
    factor = 0.8
    start = x0 + width * factor, y0
    end = zoomed_x0 + factor * zoomed_width, zoomed_y0
    arrow = mpatches.FancyArrowPatch(start, end, color='r', lw=linewidth, zorder=5)
    ax.add_patch(arrow)
    start = x0, y0 + height * factor
    end = zoomed_x0, zoomed_y0 + zoomed_height * factor
    arrow = mpatches.FancyArrowPatch(start, end, color='r', lw=linewidth, zorder=5)
    ax.add_patch(arrow)

    # zoomed_ax.add_patch(arrow)
    # matplotlib.patches.Arrow
    # line = plt.Line2D((0.1, 0.2), (0.9, 0.4), color='r')
    # ax.add_patch(line)

    # fig.savefig(os.path.join(FIGURES_PATH, 'periodic-flat.png'), transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'periodic-flat.png'))
    fig.savefig(os.path.join(FIGURES_PATH, 'periodic-flat.pdf'))
    # plt.show()
    print(zoomed_ax.bbox.extents)


def make_axis_frame_circular(ax, lw=3):
    # Based on https://stackoverflow.com/questions/70041798/how-to-round-connection-for-matplotlib-axis-spines
    p_bbox = FancyBboxPatch((0, 0), 1, 1,
                            boxstyle="round,pad=-0.0040,rounding_size=0.2",
                            ec="black", fc="white", clip_on=False, lw=lw,
                            mutation_aspect=1,
                            transform=ax.transAxes)
    # zoomed_ax.add_patch(p_bbox)
    ax.patch = p_bbox
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines:
        ax.spines[s].set_visible(False)


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

    fig, axes = plt.subplots(2)
    plot_perturbations_by_list(axes, xs, deltas, ys, DeltaLs)

    ori = create_perturbed_origami_by_list(theta, L0, C0, deltas, DeltaLs)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()

    fig, ax = plt.subplots()
    imshow_with_colorbar(fig, ax, Ks, 'Ks')

    origamiplots.plot_interactive(ori)


def main():
    # plot_vase()
    # plot_spherical_cap()
    # plot_saddle()
    # plot_wavy()
    # plot_2D_sinusoid()
    # plot_cap_different_curvatures()
    # plot_cone_like()
    plot_periodic()


if __name__ == '__main__':
    main()
