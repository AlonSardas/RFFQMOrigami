import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from origami.plotsandcalcs.masterpresentation import FILES_PATH
from origami.utils import plotutils

FIGURES_PATH = os.path.join(FILES_PATH, 'Figures')

R = 1
angle0 = 0.2
alpha = 85 / 180 * np.pi
beta = 65 / 180 * np.pi

VALLEY_COLOR = '#1565C0'  # (21 / 255, 101 / 255, 192 / 255)
MOUNTAIN_COLOR = '#D32F2f'  # (211 / 255, 47 / 255, 47 / 255)


def _draw_vec(ax: Axes, angle, name=None, color='k'):
    t = R * np.array([np.cos(angle), np.sin(angle)])
    arrow = mpatches.FancyArrowPatch((0, 0), t, linestyle='-', arrowstyle='-|>', mutation_scale=40, lw=5.0,
                                     zorder=5, color=color)
    ax.add_patch(arrow)
    text_pos = t * 1.1
    if name:
        ax.text(text_pos[0], text_pos[1], name, ha='center', va='center')


def _draw_rotation_arrow(ax, angle, name, text_pos=(0, 0), textsize=30):
    pos = R * 0.75 * np.array([np.cos(angle), np.sin(angle)])
    ellipse_angle = 90 + np.rad2deg(angle)
    plotutils.draw_elliptic_arrow(ax, pos, 0.2, 0.13, ellipse_angle, -20, 180 + 20)
    ax.text(pos[0] + text_pos[0], pos[1] + text_pos[1], name, fontsize=textsize, va='center', ha='left')


def plot_vertex_for_Kawasaki():
    mpl.rcParams['font.size'] = 20
    fig: Figure = plt.figure(figsize=(5, 5))
    ax: Axes = fig.add_subplot(111)

    circle = mpatches.Circle((0, 0), R * 0.99, fill=False, linewidth=3, color='k')
    ax.add_patch(circle)

    at_1 = angle0
    at_2 = angle0 + alpha
    at_3 = angle0 + alpha + beta
    at_4 = angle0 - (np.pi - beta)
    vec_color = 'g'
    _draw_vec(ax, at_1, color=vec_color)
    _draw_vec(ax, at_2, color=vec_color)
    _draw_vec(ax, at_3, color=vec_color)
    _draw_vec(ax, at_4, color=vec_color)

    arc_size = 0.75
    dsize = 0.08
    lw = 3
    color = 'grey'

    def plot_arc(start, end):
        arc = mpatches.Arc((0, 0), arc_size, arc_size,
                           theta1=np.rad2deg(start),
                           theta2=np.rad2deg(end),
                           lw=lw, zorder=-10, color=color)
        ax.add_patch(arc)

    plot_arc(angle0, angle0 + alpha)
    arc_size += dsize
    plot_arc(angle0 + alpha, angle0 + alpha + beta)
    arc_size += dsize
    plot_arc(angle0 + alpha + beta, angle0 + np.pi + beta)
    arc_size += dsize
    plot_arc(angle0 + np.pi + beta, angle0 + 2 * np.pi)

    text_size = 32
    ax.text(0.05, 0.07,
            r'$ \alpha_1 $', ha='left', va='bottom', color=color, fontsize=text_size)
    ax.text(-0.05, 0.07,
            r'$ \alpha_2 $', ha='right', va='bottom', color=color, fontsize=text_size)
    ax.text(-0.09, -0.05,
            r'$ \alpha_3 $', ha='right', va='top', color=color, fontsize=text_size)
    ax.text(0.05, -0.05,
            r'$ \alpha_4 $', ha='left', va='top', color=color, fontsize=text_size)

    text_size = 30

    ax.set_xlim(-1.2 * R, 1.2 * R)
    ax.set_ylim(-1.2 * R, 1.2 * R)
    ax.set_aspect('equal')
    ax.set_axis_off()
    # fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, '4-vertex-kawasaki.svg'))
    fig.savefig(os.path.join(FIGURES_PATH, '4-vertex-kawasaki.png'))

    plt.show()


def plot_single_vertex():
    mpl.rcParams['font.size'] = 30
    fig: Figure = plt.figure(figsize=(5, 5))
    ax: Axes = fig.add_subplot(111)

    circle = mpatches.Circle((0, 0), R * 0.99, fill=False, linewidth=3, color='k')
    ax.add_patch(circle)

    at_1 = angle0
    at_2 = angle0 + alpha
    at_3 = angle0 + alpha + beta
    at_4 = angle0 - (np.pi - beta)
    _draw_vec(ax, at_1, color=MOUNTAIN_COLOR)
    _draw_vec(ax, at_2, color=VALLEY_COLOR)
    _draw_vec(ax, at_3, color=MOUNTAIN_COLOR)
    _draw_vec(ax, at_4, color=MOUNTAIN_COLOR)

    arc_size = 0.85
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

    text_size = 32
    ax.text(0.05, 0.05,
            r'$ \alpha^R $', ha='left', va='bottom', color=color, fontsize=text_size)
    ax.text(-0.05, 0.08,
            r'$ \alpha^L $', ha='right', va='bottom', color=color, fontsize=text_size)

    ax.set_xlim(-1.2 * R, 1.2 * R)
    ax.set_ylim(-1.2 * R, 1.2 * R)
    ax.set_aspect('equal')
    ax.set_axis_off()
    # fig.tight_layout()

    plot_1DOF = True
    if plot_1DOF:
        _draw_rotation_arrow(ax, at_1, r'$\gamma_2(\gamma_1)$', (-0.35, -0.20))
        _draw_rotation_arrow(ax, at_2, r'$\gamma_3(\gamma_1)$', (0.10, -0.1))
        _draw_rotation_arrow(ax, at_3, r'$\gamma_4(\gamma_1)$', (-0.1, 0.25))
        _draw_rotation_arrow(ax, at_4, r'$\gamma_1$', (-0.25, +0.2))

        fig.savefig(os.path.join(FIGURES_PATH, '4-vertex-1DOF.svg'))
        fig.savefig(os.path.join(FIGURES_PATH, '4-vertex-1DOF.png'))

    else:
        _draw_rotation_arrow(ax, at_1, r'$\gamma_2$', (-0.2, -0.20))
        _draw_rotation_arrow(ax, at_2, r'$\gamma_3$', (0.10, -0.1))
        _draw_rotation_arrow(ax, at_3, r'$\gamma_4$', (-0.1, 0.25))
        _draw_rotation_arrow(ax, at_4, r'$\gamma_1$', (-0.25, +0.2))

        fig.savefig(os.path.join(FIGURES_PATH, '4-vertex.svg'))
        fig.savefig(os.path.join(FIGURES_PATH, '4-vertex.png'))

    plt.show()


def test_plot_3d():
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d",
                                 elev=0, azim=0)

    verts = create_sector_verts(0, np.pi / 3, 50)

    print(verts.shape)

    poly = Poly3DCollection([verts], facecolors='g', alpha=.7)
    ax.add_collection3d(poly)
    plt.show()


def create_sector_verts(alpha, beta, N=50, R=1.0) -> np.ndarray:
    verts = np.zeros((N + 1, 3))
    angles = np.linspace(alpha, beta, N)
    verts[1:, 0] = R * np.cos(angles)
    verts[1:, 1] = R * np.sin(angles)

    return verts


def main():
    plot_vertex_for_Kawasaki()
    # plot_single_vertex()


if __name__ == '__main__':
    main()
