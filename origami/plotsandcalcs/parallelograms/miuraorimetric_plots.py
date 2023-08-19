import fractions
import os

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import origami.plotsandcalcs
from origami.miuraori import SimpleMiuraOri
from origami.utils import plotutils
from origami.utils.plotutils import set_pi_ticks, set_3D_labels

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH, 'RFFQM/Figures')


def plot_simple_crease_pattern():
    ori = SimpleMiuraOri(np.ones(6), np.ones(6))
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=90, elev=-100)
    ori.plot(ax)

    set_3D_labels(ax)

    plt.savefig(os.path.join(FIGURES_PATH, 'simple_pattern.png'))

    plt.show()


def plot_parallelograms_example():
    ori = SimpleMiuraOri(
        [1, 1.5, 0.7, 1, 2, 0.3, 0.8],
        [0.6, 1.1, 1.9, 0.9, 1.1, 1], angle=1)
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=-100)
    ori.plot(ax)

    set_3D_labels(ax)

    plt.savefig(os.path.join(FIGURES_PATH, 'parallelograms-example.svg'))

    plt.show()


def plot_FFF_unit():
    ori = SimpleMiuraOri([1, 1.5], [1.5, 0.8])
    fig = plt.figure()

    # ax: Axes3D = fig.add_subplot(111, projection='3d', azim=90, elev=-100)
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-110, elev=40)

    ori.set_omega(0.9)
    ori.plot(ax, alpha=0.4)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    set_3D_labels(ax)

    edge_points = ori.dots[:, [ori.indexes[0, 0],
                               ori.indexes[0, -1],
                               ori.indexes[-1, 0],
                               ori.indexes[-1, -1]]]
    ax.scatter3D(edge_points[0, :], edge_points[1, :], edge_points[2, :], color='r', s=220)

    plt.savefig(os.path.join(FIGURES_PATH, 'FFF_unit.png'))

    plt.show()


def plot_unperturbed_unit_cell():
    ori = SimpleMiuraOri([1.4, 1.4], [1.0, 1.0], angle=-0.6)
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-50, elev=23)

    # ori.set_omega(-1.2)
    ori.set_omega(1.2)
    _, wire = ori.plot(ax, alpha=0.4)
    wire.set_color('r')
    wire.set_alpha(0.2)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    set_3D_labels(ax)

    edge_points = ori.dots[:, [ori.indexes[0, 0],
                               ori.indexes[0, -1],
                               ori.indexes[-1, 0],
                               ori.indexes[-1, -1]]]
    ax.scatter3D(edge_points[0, :], edge_points[1, :], edge_points[2, :], color='g', s=120)

    dot = ori.dots[:, ori.indexes[0, 0]]
    ax.text(dot[0] - 0.2, dot[1], dot[2], "E", fontsize=30)
    dot = ori.dots[:, ori.indexes[-1, 0]]
    ax.text(dot[0] + 0.1, dot[1], dot[2], "K", fontsize=30)
    dot = ori.dots[:, ori.indexes[0, -1]]
    ax.text(dot[0] + 0.1, dot[1], dot[2]+0.2, "A", fontsize=30)
    dot = ori.dots[:, ori.indexes[-1, -1]]
    ax.text(dot[0], dot[1], dot[2]+0.2, "J", fontsize=30)

    plt.savefig(os.path.join(FIGURES_PATH, 'unperturbed-unit-cell.svg'))

    plt.show()


def plot_zigzag_with_patterns():
    ori = SimpleMiuraOri(
        [3, 3],
        [1, 2, 1, 3, 2, 1, 2, 0.9], angle=0.4)
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=55, elev=35)

    ori.set_omega(-1.7)
    ori.plot(ax, alpha=0.3)

    edge_points = ori.dots[:, ori.indexes[:, 0]]
    ax.scatter3D(edge_points[0, :], edge_points[1, :], edge_points[2, :], color='r', s=120)

    # plotutils.set_axis_scaled(ax)
    lim = 2.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_aspect('equal')
    set_3D_labels(ax)

    fig.savefig(os.path.join(FIGURES_PATH, 'YZ-zigzag-pattern.svg'))
    plt.show()

    ori = SimpleMiuraOri(
        [1, 2, 1.5, 2, 3, 1],
        [2, 2], angle=0.6)
    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=89)

    ori.set_omega(-1.2)
    ori.plot(ax, alpha=0.4)

    edge_points = ori.dots[:, ori.indexes[0, :]]
    ax.scatter3D(edge_points[0, :], edge_points[1, :], edge_points[2, :], color='r', s=120)

    plotutils.set_axis_scaled(ax)
    set_3D_labels(ax)

    fig.savefig(os.path.join(FIGURES_PATH, 'XY-zigzag-pattern.svg'))

    plt.show()


def plot_gamma_vs_activation_angle():
    matplotlib.rc('font', size=22)

    fig, ax = plt.subplots()
    xs = np.linspace(0, np.pi, 200)
    ys = np.arccos((3 * np.cos(xs) - 1) / (3 - np.cos(xs)))

    ax.plot(xs, ys)

    set_pi_ticks(ax, 'xy')

    ax.set_xlabel(r'$ \omega $')
    ax.set_ylabel(r'$ \gamma\left(\omega;\beta=\pi/4\right) $')

    fig.savefig(os.path.join(FIGURES_PATH, 'gamma_vs_activation_angle.png'))

    plt.show()


def plot_phi_vs_activation_angle():
    matplotlib.rc('font', size=22)

    fig, ax = plt.subplots()
    ax: matplotlib.axes.Axes = ax
    xs = np.linspace(0, np.pi, 200)
    ys = 1 / 2 * np.arccos(-1 / 2 * np.cos(xs) + 1 / 2)

    ax.plot(xs, ys)
    ax.set_xlabel(r'$ \omega $')
    ax.set_ylabel(r'$ \phi\left(\omega;\beta=\pi/4\right) $')
    set_pi_ticks(ax, 'x')
    ax.set_yticks([0, np.pi / 8, np.pi / 4])
    ax.set_yticklabels(['0', r'$\frac{1}{8}\pi$', r'$\frac{1}{4}\pi$'])

    fig.savefig(os.path.join(FIGURES_PATH, 'phi_vs_activation_angle.png'))

    plt.show()


def plot_theta_vs_activation_angle():
    fig, ax = plt.subplots()
    ax: matplotlib.axes.Axes = ax
    xs = np.linspace(0, np.pi, 200)

    gammas = np.arccos((3 * np.cos(xs) - 1) / (3 - np.cos(xs)))
    beta = np.pi / 4
    ys = 1 / 2 * np.arccos(-np.sin(beta) ** 2 * np.cos(gammas) - np.cos(beta) ** 2)

    ax.plot(xs, ys)
    ax.set_xlabel(r'$ \omega $')
    ax.set_ylabel(r'$ \theta\left(\omega;\beta=\pi/4\right) $')
    set_pi_ticks(ax, 'x')
    set_pi_ticks(ax, 'y', pi_range=(0, fractions.Fraction(1, 2)), divisions=4)

    # fig.savefig(os.path.join(FIGURES_PATH, 'theta_vs_activation_angle.png'))

    plt.show()


if __name__ == '__main__':
    # plot_gamma_vs_activation_angle()
    # plot_phi_vs_activation_angle()
    # plot_theta_vs_activation_angle()
    # plot_FFF_unit()
    # plot_parallelograms_example()
    plot_unperturbed_unit_cell()
    # plot_zigzag_with_patterns()
