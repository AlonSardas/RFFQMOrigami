import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from origami.utils import plotutils
from origami.utils.zigzagutils import calc_zigzag_points_by_lengths, plot_zigzag

PI = np.pi

FIGURES_PATH = '../../../../RFFQM/Figures/continuous-origami'


def plot_ratio_field():
    phi = 0.6
    xs = np.linspace(0, 2 * PI, 200)
    Hs = (np.cos(xs) * np.tan(phi) + 1) / (1 - np.cos(xs) * np.tan(phi))

    fig, ax = plt.subplots()
    fig: Figure = fig
    ax: Axes = ax
    ax.plot(xs, Hs)
    plotutils.set_pi_ticks(ax, 'X', pi_range=(0, 2))
    ax.set_xlabel('x')
    ax.set_ylabel('H(x)')

    fig.savefig(os.path.join(FIGURES_PATH, 'H-vs-x.png'))


def plot_following_curve():
    phi = 0.6

    def H_func(x): return (np.cos(x) * np.tan(phi) + 1) / (1 - np.cos(x) * np.tan(phi))

    dx = 0.5
    xs = np.arange(0, 2 * PI, dx)
    Hs = H_func(xs[:-1] + dx / 2)

    cs = np.zeros((len(xs) - 1) * 2)
    cs[::2] = dx / ((1 + Hs) * np.sin(phi))
    cs[1::2] = cs[::2] * Hs
    # print(cs)

    xs, ys, mid_xs, mid_ys = calc_zigzag_points_by_lengths(cs, phi)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()

    points, middles, _ = plot_zigzag(ax, xs, ys, mid_xs, mid_ys)

    points.set_markersize(12)
    middles.set_markersize(12)

    smooth_xs = np.linspace(0, 2 * PI, 200)
    smooth_ys = np.sin(smooth_xs)

    ax.plot(smooth_xs, smooth_ys, '--')

    ax.set_xlabel('x')
    ax.set_ylabel(r'$ y=\sin(x) $')

    fig.savefig(os.path.join(FIGURES_PATH, 'following-sin-curve.png'))

    plt.show()


def plot_vector_by_zigzag():
    fig, ax = plt.subplots()

    mpl.rcParams['font.size'] = 30
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['lines.markersize'] = 25

    # Code based on
    # https://matplotlib.org/stable/gallery/spines/centered_spines_with_arrows.html
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["left", "bottom"]].set_linewidth(3)
    ax.spines[["top", "right"]].set_visible(False)

    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", markersize=20, transform=ax.get_xaxis_transform(), clip_on=False)

    ax.text(1.04, 0,
            r'$ x $', ha='left', va='center',
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            fontsize=24)
    ax.text(0.01, 1.025,
            r'$ y $', ha='center', va='bottom',
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            fontsize=24)

    phi = 0.7
    C1 = 0.4
    C2 = 0.7
    sign = -1

    p0 = np.array((0, 0))
    p1 = C1 * np.array((np.sin(phi), sign * np.cos(phi)))
    p2 = p1 + C2 * np.array((np.sin(phi), -sign * np.cos(phi)))

    lines_color = 'g'
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color=lines_color)
    ax.plot((p1[0], p2[0]), (p1[1], p2[1]), color=lines_color)
    dot_color = 'royalblue'
    ax.plot(p0[0], p0[1], '.', zorder=50,
            color=dot_color)
    ax.plot(p2[0], p2[1], '.', zorder=50,
            color=dot_color)

    arrow = mpatches.FancyArrowPatch(
        (0, 0), p2,
        arrowstyle='->,head_width=.25', mutation_scale=30, lw=4, zorder=30,
        color='C3')
    ax.add_patch(arrow)

    ax.plot((p1[0], p1[0]), (p1[1] - 0.02, p1[1] + 0.2), '--', color='0.5', zorder=-1)

    arc_size = 0.2
    angle_color = 'k'
    angle0 = -90 * sign
    arc = mpatches.Arc(
        p1, arc_size, arc_size, theta1=angle0, theta2=angle0 + np.rad2deg(phi),
        lw=5, zorder=-10, color=angle_color)
    ax.add_patch(arc)
    arc_size += 0.02
    arc = mpatches.Arc(
        p1, arc_size, arc_size,
        theta1=angle0 - np.rad2deg(phi), theta2=angle0,
        lw=5, zorder=-10, color=angle_color)
    ax.add_patch(arc)
    y_shift = 0.07
    x_shift = 0.02
    ax.text(p1[0] + x_shift, p1[1] - sign * y_shift,
            r'$ \phi $', ha='left', va='center', color=angle_color)
    ax.text(p1[0] - x_shift, p1[1] - sign * y_shift,
            r'$ \phi $', ha='right', va='center', color=angle_color)

    middle01 = (p0 + p1) / 2
    ax.text(middle01[0] - 0.02, middle01[1] + sign * 0.05,
            r'$ C_1 $', ha='center', va='center', color=lines_color)
    middle12 = (p1 + p2) / 2
    ax.text(middle12[0] + 0.02, middle12[1] + sign * 0.05,
            r'$ C_2 $', ha='center', va='center', color=lines_color)
    ax.text(p2[0] + 0.02, p2[1] + sign * 0.05,
            r'$ \mathbf{p} $', ha='center', va='center')

    # ax.set_aspect('equal', 'datalim')
    ax.set_aspect('equal', 'box')
    # ax.set_xlim(-0.1, 1.6)
    # ax.set_ylim(-0.45, 0.96)
    # plotutils.remove_tick_labels(ax)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.grid()

    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(os.path.join(FIGURES_PATH, 'lemma-displacement-by-zigzag.pdf'))

    plt.show()


def main():
    # plot_ratio_field()
    # plot_following_curve()
    plot_vector_by_zigzag()


if __name__ == '__main__':
    main()
