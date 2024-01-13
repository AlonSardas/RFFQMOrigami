import os

import matplotlib as mpl
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami.plotsandcalcs import articleillustrations
from origami.plotsandcalcs.alternating.utils import create_perturbed_origami, create_F_from_list
from origami.quadranglearray import QuadrangleArray
from origami.utils import plotutils, linalgutils


def plot_YZ_zigzag_cross_section():
    mpl.rcParams['font.size'] = 30
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['lines.markersize'] = 25
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)

    phi = 0.6
    L0 = 1
    M = 0.4
    sign = 1

    p0 = np.array((0, 0))
    p1 = L0 * np.array((np.sin(phi), sign * np.cos(phi)))
    p2 = p1 + (L0 + M) * np.array((np.sin(phi), -sign * np.cos(phi)))

    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color=articleillustrations.MOUNTAIN_COLOR)
    ax.plot((p1[0], p2[0]), (p1[1], p2[1]), color=articleillustrations.VALLEY_COLOR)
    ax.plot(p0[0], p0[1], '.', zorder=50,
            color=articleillustrations.DOTS_ON_SURFACE_COLOR)
    ax.plot(p2[0], p2[1], '.', zorder=50,
            color=articleillustrations.DOTS_ON_SURFACE_COLOR)

    ax.text(p0[0] - 0.05, p0[1] - 0.00,
            r'A', ha='center', va='center')
    ax.text(p1[0] - 0.05, p1[1] + sign * 0.01,
            r'B', ha='center', va='center')
    ax.text(p2[0] + 0.07, p2[1],
            r'J', ha='center', va='center')

    _plot_arrow(ax, p0, (p2[0], 0), "Y'(y)", (0, sign * 0.1), length=0.96)
    _plot_arrow(ax, (p2[0], 0), p2, "Z'(y)", (0.13, 0))
    _plot_arrow(ax, p0, p2, r"$ \vec{AJ} $", (0, -sign * 0.09), length=0.96)

    ax.plot((p1[0], p1[0]), (p1[1], p1[1] - sign * 0.5), '--', color='0.5', zorder=-1)

    arc_size = 0.2
    angle_color = 'k'
    angle0 = -90 * sign
    arc = mpatches.Arc(
        p1, arc_size, arc_size, theta1=angle0, theta2=angle0 + np.rad2deg(phi),
        lw=5, zorder=-10, color=angle_color)
    ax.add_patch(arc)
    arc_size += 0.05
    arc = mpatches.Arc(
        p1, arc_size, arc_size,
        theta1=angle0 - np.rad2deg(phi), theta2=angle0,
        lw=5, zorder=-10, color=angle_color)
    ax.add_patch(arc)
    y_shift = 0.18
    x_shift = 0.02
    ax.text(p1[0] + x_shift, p1[1] - sign * y_shift,
            r'$ \phi $', ha='left', va='center', color=angle_color)
    ax.text(p1[0] - x_shift, p1[1] - sign * y_shift,
            r'$ \phi $', ha='right', va='center', color=angle_color)

    middle01 = (p0 + p1) / 2
    ax.text(middle01[0] - 0.07, middle01[1] + sign * 0.05,
            r'$ L_0 $', ha='center', va='center')
    middle12 = (p1 + p2) / 2
    ax.text(middle12[0] + 0.12, middle01[1] + sign * 0.05,
            r'$ L_0+\Delta M(y) $', ha='center', va='center')

    # ax.set_aspect('equal', 'datalim')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylim(-0.45, 0.96)
    plotutils.remove_tick_labels(ax)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.grid()
    # ax.set_axis_off()

    fig.savefig(os.path.join(articleillustrations.FIGURES_PATH, 'YZ-zigzag-cross-section.pdf'))
    fig.savefig(os.path.join(articleillustrations.FIGURES_PATH, 'YZ-zigzag-cross-section.svg'))

    plt.show()


def plot_XZ_cross_section():
    mpl.rcParams['font.size'] = 26
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['lines.markersize'] = 20

    L0 = 1
    C0 = 1.4

    rows, cols = 4, 4
    angle = 1.0
    F0 = -0.51
    Fs = np.array((0, 0, F0, F0, F0))
    Fs = create_F_from_list(Fs)
    ori = create_perturbed_origami(angle, rows, cols, L0, C0, Fs, None)

    fig: Figure = plt.figure(figsize=(9, 5))
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=7, azim=98)

    ori.set_gamma(-1.8, should_center=False)
    quads = ori.dots
    center_dots_by_first_unit_cell(quads)
    quads.dots = np.array(quads.dots, dtype='float64')
    quads.dots[0, :] *= -1
    dots, indexes = quads.dots, quads.indexes
    _, wire = quads.plot(ax, alpha=0.4)
    # wire.remove()
    wire.set_color("k")
    wire.set_alpha(0.2)

    A1 = dots[:, indexes[0, 2]]
    E1 = dots[:, indexes[0, 4]]

    _plot_arrow3D(ax, A1, E1, r'$ \vec{AE} $', (-0.07, 0, +0.21))

    AE = E1 - A1
    AEx = AE.copy()
    AEx[1] = 0
    AEx[2] = 0
    _plot_arrow3D(ax, A1, A1 + AEx, r"$ X'(x) $", (0, 0, -0.2))
    AEz = AE.copy()
    AEz[0] = 0
    AEz[1] = 0
    _plot_arrow3D(ax, A1 + AEx, A1 + AEx + AEz, r"$ Z'(x) $", (-0.3, 0, 0), length=0.8)

    ax.plot(A1[0], A1[1], A1[2], '.', zorder=50,
            color=articleillustrations.DOTS_ON_SURFACE_COLOR)
    ax.plot(E1[0], E1[1], E1[2], '.', zorder=50,
            color=articleillustrations.DOTS_ON_SURFACE_COLOR)

    ax.set_position([-0.2, -0.4, 1.4, 1.85])
    # fig.patch.set_facecolor('xkcd:mint green')
    # ax.patch.set_facecolor('#4545FF')

    # ax.set_box_aspect()
    ax.set_box_aspect((4, 4, 2))
    ax.set_aspect('equal', adjustable='datalim')
    # plotutils.set_axis_scaled(ax)
    # print(ax._box_aspect)
    # ax.set_box_aspect(ax._box_aspect, zoom=1.011)
    # plotutils.set_axis_scaled(ax)
    # ax.set_axis_off()
    plotutils.remove_tick_labels(ax)
    plotutils.set_zoom_by_limits(ax, 1.32)

    mpl.rcParams["savefig.bbox"] = "standard"
    fig.savefig(os.path.join(articleillustrations.FIGURES_PATH, 'XZ-cross-section.pdf'))
    fig.savefig(os.path.join(articleillustrations.FIGURES_PATH, 'XZ-cross-section.svg'))

    plt.show()


def _plot_arrow(ax: Axes, p0, p1, name, shift=(0.0, 0.0), color=None, length=0.93):
    start = p0 + (1 - length) / 2 * (p1 - p0)
    end = start + length * (p1 - p0)

    arrow = mpatches.FancyArrowPatch(
        start, end,
        arrowstyle='->,head_width=.25', mutation_scale=30, lw=6, zorder=30,
        color=color)
    ax.add_patch(arrow)
    middle = (p0 + p1) / 2
    ax.text(middle[0] + shift[0], middle[1] + shift[1],
            name, va='center', ha='center', zorder=30)


def _plot_arrow3D(ax: Axes3D, p0, p1, name, shift=(0.0, 0.0, 0.0), color=None, length=0.93):
    start = p0 + (1 - length) / 2 * (p1 - p0)
    end = start + length * (p1 - p0)

    arrow = plotutils.Arrow3D((start[0], end[0]), (start[1], end[1]), (start[2], end[2]),
                              arrowstyle='->,head_width=.25', mutation_scale=20, lw=3, zorder=30,
                              color=color)
    ax.add_patch(arrow)
    middle = (p0 + p1) / 2
    ax.text(middle[0] + shift[0], middle[1] + shift[1], middle[2] + shift[2],
            name, va='center', ha='center', zorder=35,
            # bbox={'facecolor': 'w', 'edgecolor': 'w'}
            )


def center_dots_by_first_unit_cell(quads: QuadrangleArray):
    dots, indexes = quads.dots, quads.indexes
    rows, cols = indexes.shape

    AJ: np.ndarray = dots[:, indexes[2, 0]] - dots[:, indexes[0, 0]]
    AE: np.ndarray = dots[:, indexes[0, 2]] - dots[:, indexes[0, 0]]
    n1: np.ndarray = np.cross(AJ, AE)

    n = np.array([0, 0, 1])

    # return

    angle = linalgutils.calc_angle(n, n1)
    # print(angle)
    rot_axis = np.cross(n, n1)
    R = linalgutils.create_rotation_around_axis(rot_axis, -angle)
    dots = R @ dots

    AJ: np.ndarray = dots[:, indexes[2, 0]] - dots[:, indexes[0, 0]]
    AE: np.ndarray = dots[:, indexes[0, 2]] - dots[:, indexes[0, 0]]

    angle_xy = np.pi - linalgutils.calc_angle(AJ, [0, 1, 0])
    R_xy = linalgutils.create_XY_rotation_matrix(angle_xy)
    dots = R_xy @ dots
    AJ: np.ndarray = dots[:, indexes[2, 0]] - dots[:, indexes[0, 0]]

    dots -= dots.mean(axis=1)[:, None]

    quads.dots = dots


def main():
    plot_YZ_zigzag_cross_section()
    # plot_XZ_cross_section()


if __name__ == '__main__':
    main()
