import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.plotsandcalcs.articleillustrations.panelcompatibility import plot_angle
from origami.plotsandcalcs.masterpresentation import FIGURES_PATH
from origami.quadranglearray import dots_to_quadrangles
from origami.utils import linalgutils


def plot_panel_illustration():
    ls = [1.3, 0.8, 1.5]
    cs = [1, 1.3, 1.2]
    angle = 1.1
    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    # add some perturbations
    angles_left[0, :] += [-0.2, -0.1, 0.2, -0.1]
    angles_left[1, :] += [-0.1, 0.2, -0.15, -0.2]
    angles_bottom[0, :] += [0.1, 0.2, -0.1]
    angles_bottom[1, :] += [0.2, -0.1, -0.1]

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)

    rot_angle = 0.2
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots

    dots = quads.dots[:2, :]
    dots = np.array(dots, 'float64')
    indexes = quads.indexes

    arc_size = 0.2
    text_pad_x = 0.085
    text_pad_y = 0.08
    angle_color = 'orangered'
    plt.rcParams.update({'font.size': 30})

    def plot_line(i0, j0, i1, j1, color):
        ax.plot((dots[0, indexes[i0, j0]], dots[0, indexes[i1, j1]]),
                (dots[1, indexes[i0, j0]], dots[1, indexes[i1, j1]]),
                color)

    plot_line(1, 1, 1, 2, 'r')
    plot_line(1, 1, 2, 1, 'r')
    plot_line(1, 2, 2, 2, 'b')
    plot_line(2, 1, 2, 2, 'b')
    plot_line(1, 1, 0, 1, '--b')
    plot_line(1, 1, 1, 0, '--r')
    plot_line(2, 1, 2, 0, '--b')
    plot_line(2, 1, 3, 1, '--b')
    plot_line(1, 2, 0, 2, '--r')
    plot_line(1, 2, 1, 3, '--r')
    plot_line(2, 2, 3, 2, '--r')
    plot_line(2, 2, 2, 3, '--b')

    def _plot_angle(ind1, ind0, ind2, name):
        plot_angle(ax, quads, ind1, ind0, ind2, name, arc_size, angle_color, text_pad_x, text_pad_y)

    def plot_alpha_RL(i0, j0, letter):
        _plot_angle((i0, j0 + 1), (i0, j0), (i0 + 1, j0), rf'$ \alpha^R_{letter} $')
        _plot_angle((i0 + 1, j0), (i0, j0), (i0, j0 - 1), rf'$ \alpha^L_{letter} $')

    plot_alpha_RL(1, 1, 'A')
    plot_alpha_RL(1, 2, 'C')
    plot_alpha_RL(2, 1, 'B')
    plot_alpha_RL(2, 2, 'D')

    ax.set_xlim(0.0, 2.4)
    ax.set_ylim(0.8, 3.0)

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, 'panel-compatibility.png'))
    fig.savefig(os.path.join(FIGURES_PATH, 'panel-compatibility.svg'))

    plt.show()


def main():
    plot_panel_illustration()


if __name__ == '__main__':
    main()
