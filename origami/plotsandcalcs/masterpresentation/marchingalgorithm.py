import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from origami.RFFQMOrigami import RFFQM
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.plotsandcalcs.articleillustrations import panelcompatibility
from origami.plotsandcalcs.articleillustrations.marchingalgorithm import draw_creases
from origami.plotsandcalcs.masterpresentation import FIGURES_PATH
from origami.quadranglearray import dots_to_quadrangles, QuadrangleArray
from origami.utils import linalgutils


def plot_marching_algorithm():
    mpl.rcParams['font.size'] = 17

    ls = [1.1, 0.8, 1.1, 0.9, 0.8]
    cs = [1, 1.2, 1.1, 0.9, 1]
    angle = 1.1
    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    # add some perturbations
    left_alpha_pert = np.array([-0.2, -0.1, 0.2, -0.1, 0.1, -0.1])
    left_beta_pert = np.array([-0.1, 0.2, -0.15, -0.2, 0.1, 0.1])
    bottom_alpha_pert = np.array([0.1, -0.2, -0.1, 0.2, 0.1])
    bottom_beta_pert = np.array([-0.3, 0.6, -0.1, 0.1, 0.2])
    pert_factor = 0.6
    angles_left[0, :] += left_alpha_pert * pert_factor
    angles_left[1, :] += left_beta_pert * pert_factor
    angles_bottom[0, :] += bottom_alpha_pert * pert_factor
    angles_bottom[1, :] += bottom_beta_pert * pert_factor

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig = plt.figure(figsize=(7, 7))
    ax: Axes = fig.add_subplot(111)

    rot_angle = -0.1
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots

    draw_creases(ax, quads)

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, 'marching1.svg'))
    fig.savefig(os.path.join(FIGURES_PATH, 'marching1.png'))

    pad_specific_params = {'alpha^L01': {'pad_x': 0.07, 'pad_y': 0.25},
                           'alpha^L03': {'pad_x': -0.02, 'pad_y': 0.20},
                           'alpha^R01': {'pad_x': 0.07, 'pad_y': 0.1},
                           'alpha^R10': {'pad_x': 0.2, 'pad_y': -0.06},
                           'alpha^R00': {'pad_x': 0.25, 'pad_y': -0.05}}

    draw_angles(ax, quads, pad_specific_params)

    fig.savefig(os.path.join(FIGURES_PATH, 'marching2.svg'))
    fig.savefig(os.path.join(FIGURES_PATH, 'marching2.png'))

    draw_lengths(ax, quads, (1, 1), (1, 2), r'$ u_{0,\!0} $', 0.85,
                 -0.0, -0.15, -0.3, -0.1, text_rotate=-20)
    draw_lengths(ax, quads, (1, 1), (2, 1), r'$ v_{0,\!0} $', 0.8,
                 0.11, -0.03, -0.01, -0.01, text_rotate=-15)
    draw_lengths(ax, quads, (1, 4), (1, 5), r'$ u_{0,\!3} $', 1.0,
                 0.05, -0.1, -0.15, -0.15, text_rotate=10)
    draw_lengths(ax, quads, (2, 1), (3, 1), r'$ v_{1,\!0} $',
                 0.75, 0.19, -0.03, -0.08, -0.05, text_rotate=15)

    fig.savefig(os.path.join(FIGURES_PATH, 'marching3.svg'))
    fig.savefig(os.path.join(FIGURES_PATH, 'marching3.png'))

    plt.show()


def draw_lengths(ax: Axes, quads: QuadrangleArray, ind0, ind1, text,
                 length=0.1, shift_x=0.0, shift_y=0.0,
                 text_shift_x=0.0, text_shift_y=0.0, text_rotate=0):
    dots, indexes = quads.dots, quads.indexes
    p0 = np.array(dots[:, indexes[ind0]])
    p1 = np.array(dots[:, indexes[ind1]])
    start = p0 + (1 - length) / 2 * (p1 - p0)
    end = start + length * (p1 - p0)
    start[0] += shift_x
    end[0] += shift_x
    start[1] += shift_y
    end[1] += shift_y
    arrow = FancyArrowPatch(start, end, linestyle='-', color='0.35', arrowstyle='|-|', mutation_scale=5, lw=1.5,
                            zorder=5)
    ax.add_patch(arrow)
    middle = (start + end) / 2
    ax.text(middle[0] + text_shift_x, middle[1] + text_shift_y, text,
            bbox={'facecolor': 'w', 'edgecolor': 'w'}, zorder=10, rotation=text_rotate)


def draw_angles(ax, quads, pad_specific_params):
    angle_color = 'orangered'

    indexes = quads.indexes
    rows, cols = indexes.shape

    arc_size = 0.2
    default_arc_pad = 0.10

    def _plot_angle(ind1, ind0, ind2, angle_name):
        ij_text = '{' + str(i - 1) + r',\!' + str(j - 1) + '}'
        text = rf'$ \{angle_name}_{ij_text} $'

        key = f'{angle_name}{i - 1}{j - 1}'
        if key in pad_specific_params:
            pad_x, pad_y = pad_specific_params[key]['pad_x'], pad_specific_params[key]['pad_y']
        else:
            pad_x = pad_y = default_arc_pad

        panelcompatibility.plot_angle(
            ax, quads, ind1, ind0, ind2,
            text, arc_size, angle_color, pad_x, pad_y)

    i = 1
    for j in range(1, cols - 1):
        _plot_angle((i, j + 1), (i, j), (i + 1, j), r'alpha^R')
        _plot_angle((i + 1, j), (i, j), (i, j - 1), 'alpha^L')
    j = 1
    for i in range(1, rows - 1):
        _plot_angle((i, j + 1), (i, j), (i + 1, j), 'alpha^R')
        _plot_angle((i + 1, j), (i, j), (i, j - 1), 'alpha^L')


def main():
    plot_marching_algorithm()


if __name__ == '__main__':
    main()
