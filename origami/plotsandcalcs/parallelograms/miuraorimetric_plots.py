import fractions
import itertools
import os

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import origami.plotsandcalcs
from origami.miuraori import SimpleMiuraOri
from origami.quadranglearray import QuadrangleArray, plot_flat_quadrangles
from origami.utils import plotutils
from origami.utils.plotutils import set_3D_labels, set_pi_ticks

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH, "RFFQM/Figures")


def plot_simple_crease_pattern():
    ori = SimpleMiuraOri(np.ones(6), np.ones(6))
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=90, elev=-100)
    ori.plot(ax)

    set_3D_labels(ax)

    plt.savefig(os.path.join(FIGURES_PATH, "simple_pattern.png"))

    plt.show()


def plot_parallelograms_example():
    ls_x = np.array([1, 1.5, 0.7, 1, 2, 0.5, 0.8])
    ls_y = np.array([0.6, 1.1, 1.9, 0.9, 1.1, 1]) * 0.7
    ori = SimpleMiuraOri(ls_x, ls_y, angle=2.6)

    quads = QuadrangleArray(ori.dots, ori.rows + 1, ori.columns + 1)
    fig, ax = plot_flat_quadrangles(quads)
    print(fig.get_figwidth(), fig.get_figheight())
    fig.set_figwidth(10)
    fig.set_figheight(9)
    ax.set_axis_off()
    pad = -0.28
    ax.set_position((pad, pad, 1 - 2 * pad, 1 - 2 * pad))

    # fig.tight_layout()
    mpl.rcParams["savefig.bbox"] = "standard"
    fig.savefig(os.path.join(FIGURES_PATH, "parallelograms-example.svg"))
    fig.savefig(os.path.join(FIGURES_PATH, "parallelograms-example.pdf"))

    plt.show()


def plot_FFF_unit():
    ori = SimpleMiuraOri([1, 1.5], [1.5, 0.8])
    fig = plt.figure()

    # ax: Axes3D = fig.add_subplot(111, projection='3d', azim=90, elev=-100)
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-110, elev=40)

    ori.set_omega(0.9)
    ori.plot(ax, alpha=0.4)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    set_3D_labels(ax)

    edge_points = ori.dots[
                  :,
                  [
                      ori.indexes[0, 0],
                      ori.indexes[0, -1],
                      ori.indexes[-1, 0],
                      ori.indexes[-1, -1],
                  ],
                  ]
    ax.scatter3D(
        edge_points[0, :], edge_points[1, :], edge_points[2, :], color="r", s=220
    )

    plt.savefig(os.path.join(FIGURES_PATH, "FFF_unit.png"))

    plt.show()


def plot_unperturbed_unit_cell():
    ori = SimpleMiuraOri([1.4, 1.4], [1.0, 1.0], angle=-0.6)
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=25, azim=-133)

    # ori.set_omega(-1.2)
    ori.set_omega(1.2)
    _, wire = ori.plot(ax, alpha=0.4)
    wire.remove()
    # wire.set_color("r")
    # wire.set_alpha(0.2)

    mountain_creases = [
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1)),
        ((1, 2), (0, 2)),
    ]
    valley_creases = [
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 1)),
        ((1, 0), (2, 0)),
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((1, 2), (2, 2)),
    ]

    creases_with_color = itertools.chain(
        itertools.zip_longest([], valley_creases, fillvalue="b"),
        itertools.zip_longest([], mountain_creases, fillvalue="r"),
    )

    for color, crease in creases_with_color:
        start, end = crease
        index_start, index_end = ori.indexes[start], ori.indexes[end]
        line = np.array([ori.dots[:, index_start], ori.dots[:, index_end]])
        l = ax.plot(line[:, 0], line[:, 1], line[:, 2], color)[0]
        l.set_zorder(10)

    lim = 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_aspect("equal")

    set_3D_labels(ax)

    edge_points = ori.dots[
                  :,
                  [
                      ori.indexes[0, 0],
                      ori.indexes[0, -1],
                      ori.indexes[-1, 0],
                      ori.indexes[-1, -1],
                  ],
                  ]
    sc = ax.plot3D(
        edge_points[0, :], edge_points[1, :], edge_points[2, :], "g.", markersize=25, alpha=1.0
    )[0]
    sc.set_zorder(15)

    dot = ori.dots[:, ori.indexes[0, 0]]
    ax.text(dot[0], dot[1], dot[2] - 0.3, "A", fontsize=30,
            va='center', ha='center')
    dot = ori.dots[:, ori.indexes[-1, 0]]
    ax.text(dot[0] - 0.1, dot[1] + 0.2, dot[2] - 0.1, "J", fontsize=30)
    dot = ori.dots[:, ori.indexes[0, -1]]
    ax.text(dot[0], dot[1] - 0.05, dot[2] - 0.3, "E", fontsize=30,
            va='center', ha='center')
    dot = ori.dots[:, ori.indexes[-1, -1]]
    ax.text(dot[0], dot[1] + 0.3, dot[2] + 0.07, "G", fontsize=30,
            va='center', ha='center')

    A_dot = ori.dots[:, ori.indexes[0, 0]]
    J_dot = ori.dots[:, ori.indexes[-1, 0]]
    AJ_arrow = plotutils.Arrow3D((A_dot[0], J_dot[0]),
                                 (A_dot[1] + 0.1, J_dot[1] - 0.1),
                                 (A_dot[2], J_dot[2]),
                                 arrowstyle='->,head_width=.25', mutation_scale=30, lw=2.5)
    ax.add_patch(AJ_arrow)

    E_dot = ori.dots[:, ori.indexes[0, -1]]
    AE_arrow = plotutils.Arrow3D((A_dot[0] + 0.1, E_dot[0] - 0.1),
                                 (A_dot[1], E_dot[1]),
                                 (A_dot[2], E_dot[2]),
                                 arrowstyle='->,head_width=.25', mutation_scale=30, lw=2.5)
    ax.add_patch(AE_arrow)

    plotutils.remove_tick_labels(ax)

    # fig.tight_layout()
    # mpl.rcParams["savefig.bbox"] = "standard"

    plt.savefig(os.path.join(FIGURES_PATH, "unperturbed-unit-cell.pdf"), pad_inches=0.2)

    plt.show()


def plot_zigzag_with_patterns():
    # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    # mpl.rcParams["savefig.bbox"] = "standard"

    ori = SimpleMiuraOri([3, 3], [1, 2, 1, 3, 2, 1, 2, 0.9], angle=0.4)
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=55, elev=35)

    ori.set_omega(-1.7)
    ori.plot(ax, alpha=0.3)

    edge_points = ori.dots[:, ori.indexes[:, 0]]
    ax.scatter3D(
        edge_points[0, :], edge_points[1, :], edge_points[2, :], color="r", s=120
    )

    # plotutils.set_axis_scaled(ax)
    # lim = 2.5
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)
    ax.set_aspect("equal")
    set_3D_labels(ax)
    ax.set_xticks([-2, -1, 0, 1, 2])

    fig.savefig(os.path.join(FIGURES_PATH, "YZ-zigzag-pattern.svg"))
    fig.savefig(os.path.join(FIGURES_PATH, "YZ-zigzag-pattern.pdf"))
    # plt.show()

    ori = SimpleMiuraOri([1, 2, 1.5, 2, 3, 1], [2, 2], angle=0.6)
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-120, elev=37)

    ori.set_omega(-1.2)
    ori.plot(ax, alpha=0.4)

    edge_points = ori.dots[:, ori.indexes[0, :]]
    ax.scatter3D(
        edge_points[0, :], edge_points[1, :], edge_points[2, :], color="r", s=120
    )

    plotutils.set_axis_scaled(ax)
    set_3D_labels(ax)
    ax.set_xlabel("X", labelpad=20)
    ax.set_zticks([-0.4, 0, 0.4])

    # fig.tight_layout(rect=(0.2,0,0.9,1))
    # fig.tight_layout()

    # fig.savefig(os.path.join(FIGURES_PATH, "XY-zigzag-pattern.svg"), pad_inches=0.2)
    fig.savefig(os.path.join(FIGURES_PATH, "XY-zigzag-pattern.pdf"), pad_inches=0.2)

    plt.show()


def plot_gamma_vs_activation_angle():
    matplotlib.rc("font", size=22)

    fig, ax = plt.subplots()
    xs = np.linspace(0, np.pi, 200)
    ys = np.arccos((3 * np.cos(xs) - 1) / (3 - np.cos(xs)))

    ax.plot(xs, ys)

    set_pi_ticks(ax, "xy")

    ax.set_xlabel(r"$ \omega $")
    ax.set_ylabel(r"$ \gamma\left(\omega;\beta=\pi/4\right) $")

    fig.savefig(os.path.join(FIGURES_PATH, "gamma_vs_activation_angle.png"))

    plt.show()


def plot_theta_vs_activation_angle():
    matplotlib.rc("font", size=22)

    fig, ax = plt.subplots(layout="constrained", figsize=(5, 4))
    ax: matplotlib.axes.Axes = ax
    xs = np.linspace(0, np.pi, 200)
    ys = 1 / 2 * np.arccos(-1 / 2 * np.cos(xs) + 1 / 2)

    ax.plot(xs, ys)
    ax.set_xlabel(r"$ \omega $")
    ax.set_ylabel(r"$ \theta\left(\omega;\vartheta=\pi/4\right) $")
    set_pi_ticks(ax, "x")
    ax.set_yticks([0, np.pi / 8, np.pi / 4])
    ax.set_yticklabels(["0", r"$\frac{1}{8}\pi$", r"$\frac{1}{4}\pi$"])

    fig.savefig(os.path.join(FIGURES_PATH, "theta_vs_activation_angle.png"))
    fig.savefig(os.path.join(FIGURES_PATH, "theta_vs_activation_angle.pdf"))

    plt.show()


def plot_phi_vs_activation_angle():
    matplotlib.rc("font", size=22)

    fig, ax = plt.subplots(layout="constrained", figsize=(5, 4))
    ax: matplotlib.axes.Axes = ax
    xs = np.linspace(0, np.pi, 200)

    gammas = np.arccos((3 * np.cos(xs) - 1) / (3 - np.cos(xs)))
    beta = np.pi / 4
    ys = 1 / 2 * np.arccos(-np.sin(beta) ** 2 * np.cos(gammas) - np.cos(beta) ** 2)

    ax.plot(xs, ys)
    ax.set_xlabel(r"$ \omega $")
    ax.set_ylabel(r"$ \phi\left(\omega;\vartheta=\pi/4\right) $")
    set_pi_ticks(ax, "x")
    set_pi_ticks(ax, "y", pi_range=(0, fractions.Fraction(1, 2)), divisions=4)

    fig.savefig(os.path.join(FIGURES_PATH, 'phi_vs_activation_angle.pdf'))
    fig.savefig(os.path.join(FIGURES_PATH, 'phi_vs_activation_angle.png'))

    plt.show()


if __name__ == "__main__":
    # plot_gamma_vs_activation_angle()
    # plot_phi_vs_activation_angle()
    # plot_theta_vs_activation_angle()
    # plot_FFF_unit()
    # plot_parallelograms_example()
    plot_unperturbed_unit_cell()
    # plot_zigzag_with_patterns()
