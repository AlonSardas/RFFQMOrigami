import logging
import os.path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D

from origami import origamimetric
from origami.RFFQMOrigami import RFFQM
from origami.plotsandcalcs.alternating import curvatures
from origami.plotsandcalcs.alternating.curvatures import create_kx_ky_funcs
from origami.plotsandcalcs.alternating.utils import create_perturbed_origami, cos, tan, csc, \
    sec, create_perturbed_origami_by_list
from origami.plotsandcalcs.articleillustrations import FIGURES_PATH
from origami.utils import logutils, plotutils
from origami.utils.plotutils import imshow_with_colorbar


def plot_spherical_cap_compare_methods():
    chi = 1 / 16
    xi = 1 / 16
    theta = 1.40
    W0 = 2.4
    L0 = 2
    C0 = 1

    logutils.enable_logger()
    logging.getLogger('origami').setLevel(logging.WARNING)
    logging.getLogger('origami.alternating').setLevel(logging.DEBUG)

    delta0 = -0.15
    DeltaL0 = -0.2

    kx = ky = 1

    def calc_Ks_linear():
        s = 0.293
        t = 0.269
        sign = 1
        # delta = lambda x: sign * (x - 0.5) * s
        # DeltaL = lambda y: sign * L0 * (y - 0.5) * t
        delta = lambda x: sign * (s * x + delta0)
        DeltaL = lambda y: sign * (L0 * y * t + DeltaL0)
        ddelta = lambda x: (delta(x + 0.01) - delta(x - 0.01)) / 0.02
        dDeltaL = lambda y: (DeltaL(y + 0.01) - DeltaL(y - 0.01)) / 0.02

        xs = np.arange(0, 1 / chi, 0.5) * chi
        ys = np.arange(0, 1 / xi, 1, ) * xi
        print(xs, ys)
        # _plot_perturbations(xs, delta(xs), ys, DeltaL(ys) * xi)
        # plt.show()

        angle = theta
        ddelta0 = ddelta(1)
        dDeltaL0 = dDeltaL(1)
        K_factor = 1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * 1 * \
                   1 * (2 * csc(angle) ** 2 - cos(W0) - 1)

        expectedK = K_factor * ddelta0 * dDeltaL0
        print(f"Kfactor={K_factor}, expected K {expectedK}")

        kx_func, ky_func = curvatures.create_kx_ky_funcs_linearized(L0, C0, W0, theta)
        print(f"kx={kx_func(ddelta0)}, ky={ky_func(dDeltaL0)}, K={kx_func(ddelta0) * ky_func(dDeltaL0)}")
        print(f"needed s={1 / kx_func(1)}, needed t={1 / ky_func(1) / L0}")

        ori = create_perturbed_origami(theta, chi, xi, L0, C0, delta, DeltaL)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()

        kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, theta)
        expected_K_func = lambda j, i: kx_func(delta(j * chi), ddelta(j * chi)) * ky_func(DeltaL(i * xi),
                                                                                          dDeltaL(i * xi))
        # print(expected_K_func(2, 2), expected_K_func(5, 5))

        return delta, DeltaL, Ks

    def calc_Ks_better():
        xs, deltas = curvatures.get_delta_for_kx(L0, C0, W0, theta, kx, delta0, chi)
        ys, DeltaLs = curvatures.get_DeltaL_for_ky(L0, C0, W0, theta, ky, DeltaL0, xi)

        # _plot_perturbations(xs, deltas, ys, DeltaLs * xi)

        ori = create_perturbed_origami_by_list(theta, L0, C0, deltas, DeltaLs)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))

        geometry = origamimetric.OrigamiGeometry(ori.dots)
        Ks, Hs = geometry.get_curvatures_by_shape_operator()

        # origamiplots.plot_interactive(ori)

        return xs, deltas, ys, DeltaLs, ori, Ks

    delta_func, DeltaL_func, Ks_linear = calc_Ks_linear()
    xs, deltas, ys, DeltaLs, ori, Ks_better = calc_Ks_better()
    plot_perturbations_comparison(xs, deltas, DeltaLs, ys, delta_func, DeltaL_func)
    # plt.show()
    plot_results(ori, Ks_linear, Ks_better)
    plt.show()

    # fig, axes = plt.subplots(2)
    # imshow_with_colorbar(fig, axes[0], Ks_linear, 'Linear')
    # imshow_with_colorbar(fig, axes[1], Ks_better, 'Better')
    # origamiplots.plot_interactive(ori)


def _other_plots(ori: RFFQM):
    from origami.others import plotlyplots
    fig = plotlyplots.plot_with_plotly(ori.dots, alpha=0.5)
    fig.show()
    from mayavi import mlab
    from origami.others import mayaviplots
    mayaviplots.plot_with_mayavi(ori.dots)
    mlab.show()


def plot_perturbations_comparison(xs, deltas, DeltaLs, ys, delta_func, DeltaL_func):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(xs, deltas, '.', label='improved approx')
    axes[0].plot(xs, delta_func(xs), '.', label='linearized approx')
    axes[1].plot(ys, DeltaLs, '.', label='improved approx')
    axes[1].plot(ys, DeltaL_func(ys), '.', label='linearized approx')
    axes[0].set_xlabel('x coordinate')
    axes[0].set_ylabel(r'$\tilde{\delta}(x)$')
    axes[1].set_xlabel('y coordinate')
    axes[1].set_ylabel(r'$\widetilde{\Delta L}(y)$')
    axes[1].set_xlim(axes[0].get_xlim())
    axes[0].legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'inverse-design-perturbations-comparison.pdf'))


def _plot_perturbations(xs, deltas, ys, DeltaLs):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(xs, deltas, '.')
    axes[1].plot(ys, DeltaLs, '.')
    return fig, axes


def plot_spherical_cap_by_old_and_compare_to_new():
    chi = 1 / 16
    xi = 1 / 16
    theta = 1.40
    W0 = 2.4
    L0 = 2
    C0 = 1

    s = 0.293
    t = 0.269
    sign = 1
    delta = lambda x: sign * (x - 0.5) * s
    # delta = lambda x: np.sin(x / (2 * np.pi)) * s
    DeltaL = lambda y: sign * L0 * (y - 0.5) * t
    ddelta = lambda x: (delta(x + 0.01) - delta(x - 0.01)) / 0.02
    dDeltaL = lambda y: (DeltaL(y + 0.01) - DeltaL(y - 0.01)) / 0.02

    angle = theta
    ddelta0 = ddelta(1)
    dDeltaL0 = dDeltaL(1)
    K_factor = 1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * 1 * \
               1 * (2 * csc(angle) ** 2 - cos(W0) - 1)

    expectedK = K_factor * ddelta0 * dDeltaL0
    print(f"Kfactor={K_factor}, expected K {expectedK}")

    kx_func, ky_func = curvatures.create_kx_ky_funcs_linearized(L0, C0, W0, theta)
    print(f"kx={kx_func(ddelta0)}, ky={ky_func(dDeltaL0)}, K={kx_func(ddelta0) * ky_func(dDeltaL0)}")
    print(f"needed s={1 / kx_func(1)}, needed t={1 / ky_func(1) / L0}")

    logutils.enable_logger()
    logging.getLogger('origami').setLevel(logging.WARNING)
    logging.getLogger('origami.alternating').setLevel(logging.DEBUG)

    ori = create_perturbed_origami(theta, chi, xi, L0, C0, delta, DeltaL)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()

    kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, theta)
    expected_K_func = lambda j, i: kx_func(delta(j * chi), ddelta(j * chi)) * ky_func(DeltaL(i * xi), dDeltaL(i * xi))
    # print(expected_K_func(2, 2), expected_K_func(5, 5))

    expected_Ks = calc_expected_Ks(Ks, expected_K_func)

    # origamiplots.plot_interactive(ori)

    plot_results_1(ori, Ks, expected_Ks)
    plt.show()


def plot_results(ori: RFFQM, Ks_linearized: np.ndarray, Ks_better: np.ndarray):
    mpl.rcParams['font.size'] = 28

    vmin = min(Ks_linearized.min(), Ks_better.min())
    vmax = max(Ks_linearized.max(), Ks_better.max())

    fig, ax = plt.subplots()
    im = ax.imshow(Ks_linearized, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(FIGURES_PATH, 'inverse-design-comparison-linearized.pdf'))

    fig, ax = plt.subplots()
    im = ax.imshow(Ks_better, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    fig.savefig(os.path.join(FIGURES_PATH, 'inverse-design-comparison-better.pdf'))

    mpl.rcParams['font.size'] = 20

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=35, azim=-129, computed_zorder=False)
    panels = ori.dots.plot(ax, alpha=0.05, edge_alpha=0.35, edge_width=0.5)
    # panels.set_facecolor(np.random.random((3, 32, 32)))

    dots_color = '#FF0000'

    surface_dots = ori.dots.dots[:, ori.dots.indexes[::2, ::2]]
    surface_dots = surface_dots.astype('float64')
    ax.scatter3D(surface_dots[0, :], surface_dots[1, :], surface_dots[2, :], color=dots_color, zorder=5, s=6)

    plotutils.set_axis_scaled(ax)
    ax.set_xlabel('X', labelpad=30)
    ax.set_ylabel('Y', labelpad=30)
    ax.set_zlabel('')
    ax.set_zticklabels([])

    # ax.set_position(top=1.0, bottom=0.0, left=0.0, right=1.0)
    # ax.set_position([0, 0, 1, 1])
    # mpl.rcParams["savefig.bbox"] = "standard"
    # bbox = fig.get_tightbbox()
    # bbox.padded(-0.4, -0.4)
    fig.savefig(os.path.join(FIGURES_PATH, 'inverse-design-example.pdf'),
                bbox_inches=Bbox.from_extents(2.4, 1.2, 10, 7)
                )


def plot_results_1(ori: RFFQM, Ks: np.ndarray, expected_Ks: np.ndarray):
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots()
    im = imshow_with_colorbar(fig, ax, (expected_Ks - Ks) / Ks * 100, "percentage error")
    im.set_clim(0, 15)

    fig, ax = plt.subplots()
    im = imshow_with_colorbar(fig, ax, (1 - Ks) / Ks * 100, "bad percentage error")
    # im.set_clim(0, 15)

    vmin = min(Ks.min(), expected_Ks.min())
    vmax = max(Ks.max(), expected_Ks.max())

    fig, ax = plt.subplots()
    im = ax.imshow(Ks, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    fig.savefig(os.path.join(FIGURES_PATH, 'inverse-design-example-actual-K.pdf'))

    fig, ax = plt.subplots()
    im = ax.imshow(expected_Ks, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(FIGURES_PATH, 'inverse-design-example-predicted-K.pdf'))

    mpl.rcParams['font.size'] = 20

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=35, azim=-129, computed_zorder=False)
    _, wire = ori.dots.plot_with_wireframe(ax, alpha=0.15)
    # wire.set_color('k')
    wire.set_alpha(0.06)

    dots_color = '#FF0000'

    surface_dots = ori.dots.dots[:, ori.dots.indexes[::2, ::2]]
    surface_dots = surface_dots.astype('float64')
    ax.scatter3D(surface_dots[0, :], surface_dots[1, :], surface_dots[2, :], color=dots_color, zorder=5, s=6)

    plotutils.set_axis_scaled(ax)
    ax.set_xlabel('X', labelpad=30)
    ax.set_ylabel('Y', labelpad=30)
    ax.set_zlabel('')
    ax.set_zticklabels([])

    # ax.set_position(top=1.0, bottom=0.0, left=0.0, right=1.0)
    # ax.set_position([0, 0, 1, 1])
    # mpl.rcParams["savefig.bbox"] = "standard"
    # bbox = fig.get_tightbbox()
    # bbox.padded(-0.4, -0.4)
    fig.savefig(os.path.join(FIGURES_PATH, 'inverse-design-example.pdf'),
                bbox_inches=Bbox.from_extents(2.4, 1.2, 10, 7)
                )


def _plot_deltas(chi, xi, delta):
    N_x = int(1 / chi)
    N_y = int(1 / xi)
    fig, ax = plt.subplots()
    xs = np.arange(0, N_x + 0.5, 0.5)
    ax.plot(xs, delta(xs * chi), '.')
    ax.show()


def calc_expected_Ks(Ks, expected_K_func):
    len_ys, len_xs = Ks.shape
    xs, ys = np.arange(len_xs), np.arange(len_ys)
    Xs, Ys = np.meshgrid(xs, ys)

    expected_Ks = expected_K_func(Xs, Ys)
    return expected_Ks


def compare_curvatures(Ks, expected_Ks):
    fig = plt.figure(figsize=(10.3, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=(5, 5, 0.3))

    vmin = min(Ks.min(), expected_Ks.min())
    vmax = max(Ks.max(), expected_Ks.max())
    # vmin = min(0, vmin)

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(Ks, vmin=vmin, vmax=vmax)
    ax1.set_title("Ks")
    ax1.invert_yaxis()
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(expected_Ks, vmin=vmin, vmax=vmax)
    ax2.set_title("expected Ks")
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(gs[0, 2])

    fig.colorbar(im, cax=ax3)
    return fig


def main():
    plot_spherical_cap_compare_methods()
    # plot_spherical_cap_by_old_and_compare_to_new()


if __name__ == '__main__':
    main()
