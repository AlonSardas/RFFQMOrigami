"""
Plot an example pattern with an interpolating surface that the origami is supposed to approximate
"""

import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Bbox

import origami.plotsandcalcs
from origami import origamiplots
from origami.origamiplots import plot_interactive
from origami.plotsandcalcs.alternating import betterapproxcurvatures
from origami.plotsandcalcs.alternating.utils import (create_F_from_list,
                                                     create_MM_from_list,
                                                     create_perturbed_origami)
from origami.utils import plotutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM', 'Figures')


def plot_pattern_and_smooth_surface():
    rows, cols = 14, 16
    def kx_func(t): return -0.30 * np.tanh((t - cols / 4) * 3)
    def ky_func(t): return -0.1

    F0 = -0.5
    M0 = 0.5

    L0 = 1.0
    C0 = 1.0
    W0 = -2.0
    theta = 1.1

    xs, Fs = betterapproxcurvatures.get_F_for_kx(L0, C0, W0, theta, kx_func, F0, 0, cols // 2)
    ys, MMs = betterapproxcurvatures.get_MM_for_ky(L0, C0, W0, theta, ky_func, M0, 0, rows // 2)

    # fig, axes = plt.subplots(1, 2)
    # axes[0].plot(xs, Fs, '.')
    # axes[1].plot(ys, np.diff(MMs), '.')

    F = create_F_from_list(Fs)
    MM = create_MM_from_list(MMs)

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=19, azim=-110, computed_zorder=False)
    surf, wire = ori.dots.plot(ax, alpha=0.5)
    wire.set_alpha(alpha=0.6)
    wire.set_linewidth(0.8)
    wire.set_antialiased(False)

    dots_color = '#FF0000'
    interp_color = '#FF6530'

    ori.dots.dots = np.array(ori.dots.dots, np.float64)
    surface_dots = ori.dots.dots[:, ori.dots.indexes[::2, ::2]]
    ax.scatter(surface_dots[0, :], surface_dots[1, :], surface_dots[2, :], color=dots_color, zorder=15)
    
    ori.dots.dots[2, :] += 3.4
    interp = origamiplots.plot_smooth_interpolation(ori.dots, ax)
    interp.set_color(interp_color)
    interp.set_alpha(0.6)

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


def test_axes3d():
    angle = 1.1

    ori = create_perturbed_origami(angle, 7, 9, 1.5, 1, None, None)
    fig = plt.figure(figsize=(10, 5))
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-60, elev=32, computed_zorder=False)

    ori.set_gamma(1)
    # ori.dots.center()
    panels, surf = ori.dots.plot(ax, 0.6)
    
    # print(ax.get_box_aspect())
    # plotutils.set_axis_scaled(ax)
    # print(ax.get_box_aspect())
    # ax.set_box_aspect(ax.get_box_aspect(), zoom=1.5)
    ax.set_box_aspect((4, 4, 3), zoom=1.0)
    plt.show()



def main():
    plot_pattern_and_smooth_surface()
    # test_axes3d()


if __name__ == '__main__':
    main()
