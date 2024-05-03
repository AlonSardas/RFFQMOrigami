import os
import string

import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import origami.plotsandcalcs
from origami import origamiplots
from origami.RFFQMOrigami import RFFQM
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.plotsandcalcs import articleillustrations
from origami.plotsandcalcs.articleillustrations import panelcompatibility
from origami.plotsandcalcs.articleillustrations.marchingalgorithm import draw_crease
from origami.quadranglearray import dots_to_quadrangles, QuadrangleArray
from origami.utils import plotutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH, 'RFFQM', 'Figures', 'article-illustrations')


def plot_unit_cell():
    mpl.rcParams['font.size'] = 28

    ls = np.ones(6) * 1.0
    cs = np.ones(5) * 1.2
    ls[2] = 0.5
    ls[5] = 0.5
    cs[1] = 0.5
    cs[4] = 0.5
    angle = 1.1
    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    plot_entire_pattern = False
    if plot_entire_pattern:
        ori = RFFQM(quads)
        origamiplots.plot_crease_pattern(ori)
        plt.show()

    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)

    A_ind = (3, 2)
    B_ind = (A_ind[0] + 1, A_ind[1])
    C_ind = (A_ind[0], A_ind[1] + 1)
    D_ind = (A_ind[0] + 1, A_ind[1] + 1)
    E_ind = (C_ind[0], C_ind[1] + 1)
    F_ind = (C_ind[0] + 1, C_ind[1] + 1)
    J_ind = (B_ind[0] + 1, B_ind[1])
    H_ind = (B_ind[0] + 1, B_ind[1] + 1)
    G_ind = (D_ind[0] + 1, D_ind[1] + 1)

    MVA_to_color = {1: articleillustrations.MOUNTAIN_COLOR,
                    -1: articleillustrations.VALLEY_COLOR,
                    0: 'k'}

    def _draw_crease(ind0, ind1, MVA, linestyle='-'):
        color = MVA_to_color[MVA]
        draw_crease(ax, quads, ind0[0], ind0[1], ind1[0], ind1[1], color, linestyle=linestyle)

    _draw_crease(A_ind, C_ind, -1)
    _draw_crease(C_ind, E_ind, -1)
    _draw_crease(A_ind, B_ind, 1)
    _draw_crease(C_ind, D_ind, -1)
    _draw_crease(E_ind, F_ind, 1)
    _draw_crease(B_ind, D_ind, 1)
    _draw_crease(D_ind, F_ind, 1)
    _draw_crease(B_ind, J_ind, -1)
    _draw_crease(D_ind, H_ind, 1)
    _draw_crease(F_ind, G_ind, -1)
    _draw_crease(J_ind, H_ind, -1)
    _draw_crease(H_ind, G_ind, -1)

    # _draw_crease(A_ind, (A_ind[0]-1, A_ind[1]), -1, '--')
    _draw_crease(C_ind, (C_ind[0] - 1, C_ind[1]), 1, '--')
    _draw_crease(A_ind, (A_ind[0], A_ind[1] - 1), -1, '--')
    _draw_crease(B_ind, (B_ind[0], B_ind[1] - 1), 1, '--')
    _draw_crease(E_ind, (E_ind[0], E_ind[1] + 1), -1, '--')
    _draw_crease(J_ind, (J_ind[0] + 1, J_ind[1]), 1, '--')
    _draw_crease(J_ind, (J_ind[0], J_ind[1] - 1), -1, '--')
    _draw_crease(H_ind, (H_ind[0] + 1, H_ind[1]), -1, '--')
    _draw_crease(G_ind, (G_ind[0] + 1, G_ind[1]), 1, '--')

    pad_x, pad_y = 0.02, -0.2
    all_dots_ind = (A_ind, B_ind, C_ind, D_ind, E_ind, F_ind, G_ind, H_ind, J_ind)
    plot_dots_labels(ax, quads, all_dots_ind, pad_x, pad_y)

    dots, indexes = quads.dots, quads.indexes

    def _add_circle(ind):
        circle = mpatches.Circle(dots[:2, indexes[ind]], radius=0.04, facecolor='g', zorder=5)
        ax.add_patch(circle)

    _add_circle(A_ind)
    _add_circle(E_ind)
    _add_circle(G_ind)
    _add_circle(J_ind)

    def _plot_angle(ind1, ind0, ind2, name, angle_color, arc_size, pad_x, pad_y):
        panelcompatibility.plot_angle(ax, quads, ind1, ind0, ind2, name, arc_size, angle_color, pad_x, pad_y,
                                      fontsize=24)

    def plot_eta_RL(ind0, base_angle):
        i0, j0 = ind0

        i, j = (i0 - A_ind[0], j0 - A_ind[1])
        ij_text = f'{i // 2},\!{j // 2}'
        sub_indexes = (i % 2, j % 2)
        letter_dict = {(0, 0): 'A', (1, 0): 'B', (0, 1): 'C', (1, 1): 'D'}
        letter = letter_dict[sub_indexes]
        template = r'$ ' + base_angle + r'+\eta^{%s,\!' + letter + '}_{' + ij_text + '}$'

        eta_R_txt = template % 'R'
        eta_L_txt = template % 'L'

        pad_x = 0.02
        if base_angle == r'\vartheta':
            color = articleillustrations.ANGLE_COLOR1
            pad_y = 0.26
            arc_size = 0.3
        else:
            color = articleillustrations.ANGLE_COLOR2
            pad_y = 0.11
            arc_size = 0.2

        _plot_angle((i0, j0 + 1), (i0, j0), (i0 + 1, j0),
                    eta_R_txt, color, arc_size, pad_x, pad_y)
        _plot_angle((i0 + 1, j0), (i0, j0), (i0, j0 - 1),
                    eta_L_txt, color, arc_size, pad_x, pad_y)

    plot_eta_RL(A_ind, r'\vartheta')
    plot_eta_RL(B_ind, r'\vartheta')
    plot_eta_RL(C_ind, r'\pi-\vartheta')
    plot_eta_RL(D_ind, r'\pi-\vartheta')
    plot_eta_RL(J_ind, r'\vartheta')
    plot_eta_RL(E_ind, r'\vartheta')

    p0 = dots[:2, indexes[C_ind]]
    p1 = dots[:2, indexes[D_ind]]
    middle = (p0 + p1) / 2
    plotutils.draw_elliptic_arrow(ax, (middle[0], middle[1] + 0.08), 0.2, 0.13, 0, 40, -240)
    ax.text(middle[0] + 0.1, middle[1], r'$\omega_{0,\!0}$')

    p0 = dots[:2, indexes[B_ind]]
    p1 = dots[:2, indexes[D_ind]]
    middle = 0.65 * p1 + 0.35 * p0
    plotutils.draw_elliptic_arrow(ax, middle, 0.13, 0.2, 0, -120, 130)
    # plotutils.draw_elliptic_arrow(ax, middle, 0.13, 0.2, 0, 100, -20)
    ax.text(middle[0] - 0.1, middle[1] + 0.17, r'$\gamma_{0,\!0}$')

    def _draw_length(ind0, ind1, name, shift_x, shift_y):
        p0 = dots[:2, indexes[ind0]]
        p1 = dots[:2, indexes[ind1]]
        middle = (p0 + p1) / 2
        ax.text(middle[0] + shift_x, middle[1] + shift_y, name)

    _draw_length(A_ind, C_ind, '$ c_{0,\!0}^L $', shift_x=0.05, shift_y=-0.1)
    _draw_length(C_ind, E_ind, '$ c_{0,\!0}^R $', shift_x=-0.05, shift_y=-0.23)
    _draw_length(A_ind, B_ind, r'$ \ell_{0,\!0}^A $', shift_x=-0.25, shift_y=0.05)
    _draw_length(B_ind, J_ind, r'$ \ell_{0,\!0}^B $', shift_x=-0.33, shift_y=0.09)
    _draw_length(E_ind, F_ind, r'$ \ell_{0,\!1}^A $', shift_x=0.02, shift_y=0.01)
    _draw_length(F_ind, G_ind, r'$ \ell_{0,\!1}^B $', shift_x=0.02, shift_y=0.01)
    _draw_length(J_ind, H_ind, r'$ c_{1,\!0}^L $', shift_x=-0.15, shift_y=-0.2)
    _draw_length(H_ind, G_ind, r'$ c_{1,\!0}^R $', shift_x=0.07, shift_y=-0.28)

    # _draw_panel_type(ax, 2.1, 5.5, 1)
    # _draw_panel_type(ax, 3.15, 5.5, 2)

    ax.set_aspect('equal')
    ax.set_axis_off()
    # ax.grid()
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, 'unit-cell-notation.pdf'))

    plt.show()


def plot_angles_only():
    mpl.rcParams['font.size'] = 28

    ls = np.ones(6) * 1.0
    cs = np.ones(5) * 1.2
    ls[2] = 0.5
    ls[5] = 0.5
    cs[1] = 0.5
    cs[4] = 0.5
    angle = 1.1
    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)

    A_ind = (3, 2)
    B_ind = (A_ind[0] + 1, A_ind[1])
    C_ind = (A_ind[0], A_ind[1] + 1)
    D_ind = (A_ind[0] + 1, A_ind[1] + 1)
    E_ind = (C_ind[0], C_ind[1] + 1)
    F_ind = (C_ind[0] + 1, C_ind[1] + 1)
    J_ind = (B_ind[0] + 1, B_ind[1])
    H_ind = (B_ind[0] + 1, B_ind[1] + 1)
    G_ind = (D_ind[0] + 1, D_ind[1] + 1)

    MVA_to_color = {1: articleillustrations.MOUNTAIN_COLOR,
                    -1: articleillustrations.VALLEY_COLOR,
                    0: 'k'}

    def _draw_crease(ind0, ind1, MVA, linestyle='-'):
        color = MVA_to_color[MVA]
        draw_crease(ax, quads, ind0[0], ind0[1], ind1[0], ind1[1], color, linestyle=linestyle)

    _draw_crease(A_ind, C_ind, -1)
    _draw_crease(C_ind, E_ind, -1)
    _draw_crease(A_ind, B_ind, 1)
    _draw_crease(C_ind, D_ind, -1)
    _draw_crease(E_ind, F_ind, 1)
    _draw_crease(B_ind, D_ind, 1)
    _draw_crease(D_ind, F_ind, 1)
    _draw_crease(B_ind, J_ind, -1)
    _draw_crease(D_ind, H_ind, 1)
    _draw_crease(F_ind, G_ind, -1)
    _draw_crease(J_ind, H_ind, -1)
    _draw_crease(H_ind, G_ind, -1)

    _draw_crease(C_ind, (C_ind[0] - 1, C_ind[1]), 1, '--')
    _draw_crease(A_ind, (A_ind[0], A_ind[1] - 1), -1, '--')
    _draw_crease(B_ind, (B_ind[0], B_ind[1] - 1), 1, '--')
    _draw_crease(E_ind, (E_ind[0], E_ind[1] + 1), -1, '--')
    _draw_crease(J_ind, (J_ind[0] + 1, J_ind[1]), 1, '--')
    _draw_crease(J_ind, (J_ind[0], J_ind[1] - 1), -1, '--')
    _draw_crease(H_ind, (H_ind[0] + 1, H_ind[1]), -1, '--')
    _draw_crease(G_ind, (G_ind[0] + 1, G_ind[1]), 1, '--')
    _draw_crease(F_ind, (F_ind[0], F_ind[1] + 1), 1, '--')

    pad_x, pad_y = 0.02, -0.2
    ABCD_dots_ind = (A_ind, B_ind, C_ind, D_ind)
    plot_dots_labels(ax, quads, ABCD_dots_ind, pad_x, pad_y)

    dots, indexes = quads.dots, quads.indexes

    def _add_circle(ind):
        circle = mpatches.Circle(dots[:2, indexes[ind]], radius=0.04, facecolor='g', zorder=5)
        ax.add_patch(circle)

    _add_circle(A_ind)
    _add_circle(E_ind)
    _add_circle(G_ind)
    _add_circle(J_ind)

    def _plot_angle(ind1, ind0, ind2, name, angle_color, arc_size, pad_x, pad_y):
        panelcompatibility.plot_angle(ax, quads, ind1, ind0, ind2, name, arc_size, angle_color, pad_x, pad_y,
                                      fontsize=24)

    def plot_beta_RL(ind0, base_angle):
        i0, j0 = ind0

        m, n = (i0 - A_ind[0], j0 - A_ind[1])
        mn_text = f'{m}{n}'
        template = r'$ ' + base_angle + r'+\beta^{%s}_{' + mn_text + '}$'

        beta_R_txt = template % 'R'
        beta_L_txt = template % 'L'

        pad_x = 0.02
        if base_angle == r'\vartheta':
            color = articleillustrations.ANGLE_COLOR1
            pad_y = 0.24
            arc_size = 0.3
        else:
            color = articleillustrations.ANGLE_COLOR2
            pad_y = 0.11
            arc_size = 0.2

        _plot_angle((i0, j0 + 1), (i0, j0), (i0 + 1, j0),
                    beta_R_txt, color, arc_size, pad_x, pad_y)
        _plot_angle((i0 + 1, j0), (i0, j0), (i0, j0 - 1),
                    beta_L_txt, color, arc_size, pad_x, pad_y)

    plot_beta_RL(A_ind, r'\vartheta')
    plot_beta_RL(B_ind, r'\vartheta')
    plot_beta_RL(C_ind, r'\pi-\vartheta')
    plot_beta_RL(D_ind, r'\pi-\vartheta')
    plot_beta_RL(J_ind, r'\vartheta')
    plot_beta_RL(E_ind, r'\vartheta')
    plot_beta_RL(F_ind, r'\vartheta')

    _draw_panel_type(ax, 2.1, 5.5, 1)
    _draw_panel_type(ax, 3.15, 5.5, 2)

    ax.set_aspect('equal')
    ax.set_axis_off()
    # ax.grid()
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, 'crease-pattern-angles.pdf'))

    plt.show()


def plot_dots_labels(ax: Axes, quads: QuadrangleArray, dots_indices, pad_x, pad_y):
    dots, indexes = quads.dots, quads.indexes

    for i, dot_ind in enumerate(dots_indices):
        label = string.ascii_uppercase[i]
        # This is mainly because 'I' looks just as a straight line,
        # and it is unclear it stands for the letter
        if label == 'I':
            label = 'J'

        x = dots[0, indexes[dot_ind]] + pad_x
        y = dots[1, indexes[dot_ind]] + pad_y
        ax.text(x, y, label)


def _draw_panel_type(ax: Axes, x, y, type_index):
    ax.text(x, y, f'Type {type_index}', fontsize=26, va='center', ha='center')
    dot_interval = 0.1
    dot_space_middle = 0.13
    three_dots = np.arange(3) * dot_interval
    row_dots = y + np.concatenate((dot_space_middle + three_dots, -dot_space_middle - three_dots))
    xs = np.ones(6) * x
    ax.plot(xs, row_dots, '.k')


def main():
    # plot_angles_only()
    plot_unit_cell()


if __name__ == '__main__':
    main()
