import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import origami.plotsandcalcs
from origami.plotsandcalcs.articleillustrations import panelcompatibility
from origami.utils import plotutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH, 'RFFQM', 'Figures', 'article-illustrations')


def plot_single_vertex():
    R = 1
    angle0 = 0.2
    alpha = 85 / 180 * np.pi
    beta = 65 / 180 * np.pi

    mpl.rcParams['font.size'] = 30
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)

    circle = mpatches.Circle((0, 0), R * 0.99, fill=False, linewidth=3, color='k')
    ax.add_patch(circle)

    VALLEY_COLOR = '#1565C0'  # (21 / 255, 101 / 255, 192 / 255)
    MOUNTAIN_COLOR = '#D32F2f'  # (211 / 255, 47 / 255, 47 / 255)

    def _draw_vec(angle, name, color):
        t = R * np.array([np.cos(angle), np.sin(angle)])
        arrow = mpatches.FancyArrowPatch((0, 0), t, linestyle='-', arrowstyle='-|>', mutation_scale=40, lw=5.0,
                                         zorder=5, color=color)
        ax.add_patch(arrow)
        text_pos = t * 1.1
        ax.text(text_pos[0], text_pos[1], name, ha='center', va='center')

    at_1 = angle0
    at_2 = angle0 + alpha
    at_3 = angle0 + alpha + beta
    at_4 = angle0 - (np.pi - beta)
    _draw_vec(at_1, r'$\mathbf{t}_1$', MOUNTAIN_COLOR)
    _draw_vec(at_2, r'$\mathbf{t}_2$', VALLEY_COLOR)
    _draw_vec(at_3, r'$\mathbf{t}_3$', MOUNTAIN_COLOR)
    _draw_vec(at_4, r'$\mathbf{t}_4$', MOUNTAIN_COLOR)

    def _draw_rotation_arrow(angle, name, text_pos=(0, 0)):
        pos = R * 0.8 * np.array([np.cos(angle), np.sin(angle)])
        ellipse_angle = 90 + np.rad2deg(angle)
        plotutils.draw_elliptic_arrow(ax, pos, 0.2, 0.13, ellipse_angle, -20, 180 + 20)
        ax.text(pos[0] + text_pos[0], pos[1] + text_pos[1], name, fontsize=26, va='center', ha='center')

    _draw_rotation_arrow(at_1, '$\gamma_1$', (0, -0.2))
    _draw_rotation_arrow(at_2, '$\gamma_2$', (0.2, -0.0))
    _draw_rotation_arrow(at_3, '$\gamma_3$', (0, 0.2))
    _draw_rotation_arrow(at_4, '$\gamma_4$', (-0.20, -0.0))

    arc_size = 0.55
    lw = 3
    color = 'grey'
    arc = mpatches.Arc((0, 0), arc_size, arc_size,
                       theta1=np.rad2deg(angle0),
                       theta2=np.rad2deg(angle0 + alpha),
                       lw=lw, zorder=-10, color=color)
    ax.add_patch(arc)
    arc = mpatches.Arc((0, 0), arc_size, arc_size,
                       theta1=np.rad2deg(angle0 + alpha),
                       theta2=np.rad2deg(angle0 + alpha + beta),
                       lw=lw, zorder=-10, color=color)
    ax.add_patch(arc)

    text_size = 28
    ax.text(0.05, 0.05,
            r'$ \alpha^R $', ha='left', va='bottom', color=color, fontsize=text_size)
    ax.text(-0.05, 0.05,
            r'$ \alpha^L $', ha='right', va='bottom', color=color, fontsize=text_size)

    text_size = 30
    ax.text(0.48, 0.61,
            r'$ \mathbf{R}_1(\gamma_1) $', ha='center', va='center', fontsize=text_size)
    ax.text(-0.39, 0.51,
            r'$ \mathbf{R}_2(\gamma_2) $', ha='center', va='center', fontsize=text_size)
    ax.text(-0.59, -0.32,
            r'$ \mathbf{R}_3(\gamma_3) $', ha='center', va='center', fontsize=text_size)
    ax.text(0.367, -0.42,
            r'$ \mathbf{R}_4(\gamma_4) $', ha='center', va='center', fontsize=text_size)

    ax.set_xlim(-1.2 * R, 1.2 * R)
    ax.set_ylim(-1.2 * R, 1.2 * R)
    ax.set_aspect('equal')
    # ax.grid()
    ax.set_axis_off()
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, '4-vertex.pdf'))

    plt.show()


def main():
    plot_single_vertex()


if __name__ == '__main__':
    main()
