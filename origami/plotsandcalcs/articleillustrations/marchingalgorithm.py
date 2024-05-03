import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl

from origami.RFFQMOrigami import RFFQM
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.plotsandcalcs import articleillustrations
from origami.plotsandcalcs.articleillustrations import FIGURES_PATH
from origami.plotsandcalcs.articleillustrations import panelcompatibility, ANGLE_COLOR1, ANGLE_COLOR2
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
    ori = RFFQM(quads)

    # origamiplots.plot_crease_pattern(ori)
    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)

    rot_angle = -0.1
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = np.array(quads.dots, 'float64')

    draw_creases(ax, quads)

    pad_specific_params = {'beta^L01': {'pad_x': 0.07, 'pad_y': 0.25},
                           'beta^L03': {'pad_x': -0.02, 'pad_y': 0.20},
                           'beta^R01': {'pad_x': 0.07, 'pad_y': 0.1},
                           'beta^R10': {'pad_x': 0.2, 'pad_y': -0.06},
                           'beta^R00': {'pad_x': 0.25, 'pad_y': -0.05}}

    draw_angles(ax, quads, pad_specific_params)

    arrows_dots = np.array([(1.825, 1.500),
                            (4.43, 1.580),
                            (1.81, 2.530),
                            (4.54, 2.40),
                            (1.56, 3.65),
                            (4.45, 3.4),
                            (1.46, 4.65),
                            (4.8, 4.48)])

    draw_arrows(ax, arrows_dots)

    draw_lengths(ax, quads, (1, 1), (1, 2), '$ u_{0,\!0} $', 0.85, -0.0, -0.15, -0.17, -0.07)
    draw_lengths(ax, quads, (1, 1), (2, 1), '$ v_{0,\!0} $', 0.8, 0.11, -0.03, -0.01, -0.01)
    draw_lengths(ax, quads, (1, 4), (1, 5), '$ u_{0,\!3} $', 1.0, 0.05, -0.1, 0.0, -0.05)
    draw_lengths(ax, quads, (2, 1), (3, 1), '$ v_{1,\!0} $', 0.75, 0.19, -0.03, -0.08, -0.05)

    ax.set_aspect('equal')
    ax.set_axis_off()
    # ax.grid()
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, 'marching-illustration.pdf'))

    plt.show()


def draw_lengths(ax: Axes, quads: QuadrangleArray, ind0, ind1, text,
                 length=0.1, shift_x=0.0, shift_y=0.0,
                 text_shift_x=0.0, text_shift_y=0.0):
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
            bbox={'facecolor': 'w', 'edgecolor': 'w'}, zorder=10)


def draw_arrows(ax: Axes, arrows_dots):
    for i in range(0, len(arrows_dots), 2):
        p0 = arrows_dots[i, :]
        p1 = arrows_dots[i + 1, :]
        length = 1.0
        start = p0 + (1 - length) / 2 * (p1 - p0)
        end = start + length * (p1 - p0)
        arrow = FancyArrowPatch(start, end, linestyle='-', arrowstyle='->', mutation_scale=30, lw=2.0, zorder=5)
        ax.add_patch(arrow)
    for i in range(1, len(arrows_dots) - 1, 2):
        p0 = arrows_dots[i, :]
        p1 = arrows_dots[i + 1, :]
        # length = 0.85
        start = (p0[0], p0[1] + 0.05)
        end = (p1[0], p1[1] - 0.05)
        arrow = FancyArrowPatch(start, end, linestyle='--', color='0.35', arrowstyle='->', mutation_scale=30, lw=1.5,
                                zorder=5,
                                connectionstyle='arc,angleA=110,angleB=-70,armA=50,armB=50,rad=40')
        ax.add_patch(arrow)


def draw_angles(ax, quads, pad_specific_params):
    indexes = quads.indexes
    rows, cols = indexes.shape

    arc_size = 0.2
    default_arc_pad = 0.10

    def _plot_angle(ind1, ind0, ind2, base_angle_text, angle_name):
        ij_text = '{' + str(i - 1) + ',\!' + str(j - 1) + '}'
        text = rf'$ {base_angle_text}+\{angle_name}_{ij_text} $'

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
        angle_base_text = r'\pi-\vartheta' if j % 2 == 1 else r'\vartheta'
        angle_color = ANGLE_COLOR2 if j % 2 == 1 else ANGLE_COLOR1
        _plot_angle((i, j + 1), (i, j), (i + 1, j),
                    angle_base_text, r'beta^R')
        _plot_angle((i + 1, j), (i, j), (i, j - 1),
                    angle_base_text, 'beta^L')
    j = 1
    for i in range(1, rows - 1):
        angle_base_text = r'\pi-\vartheta'
        angle_color = ANGLE_COLOR2
        _plot_angle((i, j + 1), (i, j), (i + 1, j),
                    angle_base_text, 'beta^R')
        _plot_angle((i + 1, j), (i, j), (i, j - 1),
                    angle_base_text, 'beta^L')


def draw_creases(ax, quads):
    MVA_to_color = {1: articleillustrations.MOUNTAIN_COLOR,
                    -1: articleillustrations.VALLEY_COLOR,
                    0: 'k'}

    indexes = quads.indexes
    dots = quads.dots
    rows, cols = indexes.shape

    def _draw_crease(i0, j0, i1, j1, MVA, length_factor=1.0, linestyle='-'):
        color = MVA_to_color[MVA]
        draw_crease(ax, quads, i0, j0, i1, j1, color, length_factor, linestyle)

    initial_MVA = -1
    MVA = initial_MVA
    for j in range(1, cols - 1):
        _draw_crease(0, j, 1, j, MVA, linestyle='--')
        MVA *= -1

    MVA = -initial_MVA
    for i in range(1, rows - 1):
        _draw_crease(i, 0, i, 1, MVA, linestyle='--')
        MVA *= -1

    MVA = -initial_MVA
    i = 1
    for j in range(1, cols - 1):
        vertical_MVA = -1 if j % 2 == 0 else 1
        _draw_crease(i, j, i, j + 1, MVA)
        _draw_crease(i, j, i + 1, j, vertical_MVA * MVA, length_factor=0.4)
    j = 1
    vertical_MVA = -1 if j % 2 == 0 else 1
    for i in range(1, cols - 1):
        _draw_crease(i, j, i, j + 1, MVA, length_factor=0.4)
        _draw_crease(i, j, i + 1, j, vertical_MVA * MVA)
        MVA *= -1


def draw_crease(ax, quads, i0, j0, i1, j1, color, length_factor=1.0, linestyle='-'):
    dots, indexes = quads.dots, quads.indexes
    d0 = dots[:, indexes[i0, j0]]
    d1 = dots[:, indexes[i1, j1]]
    d1 = d0 + length_factor * (d1 - d0)
    ax.plot((d0[0], d1[0]), (d0[1], d1[1]), color=color, linestyle=linestyle)


def main():
    plot_marching_algorithm()


if __name__ == '__main__':
    main()
