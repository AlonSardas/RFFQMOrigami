"""
Plot an example pattern with an interpolating surface that the origami is supposed to approximate
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D

from origami import origamiplots
from origami.plotsandcalcs.alternating import curvatures
from origami.plotsandcalcs.alternating.utils import (create_perturbed_origami_by_list)
from origami.plotsandcalcs.articleillustrations import FIGURES_PATH
from origami.utils import plotutils


def plot_pattern_and_smooth_surface():
    rows, cols = 14, 16
    # kx_func = lambda t: -0.30 * np.tanh((t - 0.5) * 3)
    # ky_func=lambda t: -0.1
    kx_func = lambda t: 0.5 * np.tanh(-(t - 0.5) * 3)
    ky_func = lambda t: 0.15

    F0 = 0.40
    M0 = 4.0

    W0 = -2.1
    theta = 1.1

    L0 = 7
    C0 = 8.5

    print(L0, C0)

    chi = 1 / cols * 2
    xi = 1 / rows * 2

    xs, deltas = curvatures.get_delta_for_kx(L0, C0, W0, theta, kx_func, F0, chi)
    ys, DeltaLs = curvatures.get_DeltaL_for_ky(L0, C0, W0, theta, ky_func, M0, xi)

    should_plot_pert = False
    if should_plot_pert:
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(xs, deltas, '.')
        axes[1].plot(ys, DeltaLs, '.')
        plt.show()

    ori = create_perturbed_origami_by_list(
        theta, L0, C0, deltas, DeltaLs)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=17, azim=-110, computed_zorder=False)
    surf = ori.dots.plot(ax, alpha=0.5, edge_alpha=0.7)
    surf.set_linewidth(1.1)
    # surf.set_alpha(alpha=0.7)
    # surf.set_edgecolor(surf.get_edgecolor())
    # surf.set_alpha(0.5)
    # wire.set_antialiased(False)
    # surf.set_antialiased(False)

    dots_color = '#FF0000'
    interp_color = '#FF6530'

    ori.dots.dots = np.array(ori.dots.dots, np.float64)
    surface_dots = ori.dots.dots[:, ori.dots.indexes[::2, ::2]]
    ax.scatter(surface_dots[0, :], surface_dots[1, :], surface_dots[2, :], color=dots_color, zorder=25)

    ori.dots.dots[2, :] += 3.4
    interp = origamiplots.plot_smooth_interpolation(ori.dots, ax)
    interp.set_color(interp_color)
    interp.set_edgecolor(None)
    interp.set_linewidth(0)
    interp.set_alpha(0.6)

    # interp.set_edgecolor(None)
    # print(interp.get_linestyle())
    # interp.set_linestyle(':')
    # interp.set_linewidth(0.001)
    # print(interp.get_edgecolor())

    surface_dots = ori.dots.dots[:, ori.dots.indexes[::2, ::2]]
    ax.scatter(surface_dots[0, :], surface_dots[1, :], surface_dots[2, :], color=dots_color, zorder=5)

    # interp.set_zorder(10)
    # interp.set_edgecolor(('r', 0.1))
    # interp.set_linewidth(0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plotutils.set_labels_off(ax)

    plotutils.set_axis_scaled(ax)

    # plt.margins(0.8)

    # mpl.rcParams["savefig.bbox"] = "standard"
    # ax.set_box_aspect(ax.get_box_aspect(), zoom=1.3)
    zoom = 1.15

    plotutils.set_zoom_by_limits(ax, zoom)

    # fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'smooth-surface-approx.pdf'),
                bbox_inches=Bbox.from_extents(3.1, 1.7, 9.5, 7.2))
    plt.show()

    # origamiplots.plot_interactive(ori)


def main():
    plot_pattern_and_smooth_surface()


if __name__ == '__main__':
    main()
