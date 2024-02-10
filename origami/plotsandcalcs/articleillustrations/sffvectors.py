import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami.plotsandcalcs.alternating.utils import create_perturbed_origami
from origami.plotsandcalcs.articleillustrations import FIGURES_PATH
from origami.utils import plotutils


def plot_sff_vectors():
    mpl.rcParams['font.size'] = 32

    L0 = 1
    C0 = 1.4

    rows, cols = 4, 4
    chi = 1 / 2
    xi = 1 / 2
    angle = 1.0
    ori = create_perturbed_origami(angle, chi, xi, L0*2, C0*2, None, None)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=35, azim=-145)

    # print(ori.calc_gamma_by_omega(2))
    ori.set_gamma(2)
    quads = ori.dots
    quads.dots = np.array(quads.dots, dtype='float64')
    # quads.dots[0, :] *= -1
    dots, indexes = quads.dots, quads.indexes

    quads.plot(ax, edge_color='k', alpha=0.4, edge_alpha=0.2)

    # _, wire = quads.plot_with_wireframe(ax, alpha=0.4)
    # wire.set_color("k")
    # wire.set_alpha(0.2)

    # plotutils.set_labels_off(ax)

    edge_points = quads.dots[:, quads.indexes[::2, ::2].flat]
    sc = ax.plot3D(
        edge_points[0, :], edge_points[1, :], edge_points[2, :], "g.", markersize=25, alpha=1.0
    )[0]
    sc.set_zorder(15)

    dots_data = {
        'A': ((0, 0), (0, 0, -0.5)),
        'E': ((0, 2), (0, 0, -0.5)),
        'J': ((2, 0), (0, 0, -0.5))
    }

    # for name, data in dots_data.items():
    #     index, shift = data
    #     dot = quads.dots[:, quads.indexes[index]]
    #     ax.text(dot[0] + shift[0], dot[1] + shift[1], dot[2] + shift[2],
    #             name, fontsize=30)

    color = '#DDCCEEFF'

    def _plot_arrow(i0, i1, name, shift=None):
        p0 = dots[:, quads.indexes[i0]]
        p1 = dots[:, quads.indexes[i1]]
        length = 0.85
        start = p0 + (1 - length) / 2 * (p1 - p0)
        end = start + length * (p1 - p0)

        if '_{x}' in name:
            zdir = 'x'
            x_shift, y_shift = 0, -0.1
        elif '_{y}' in name:
            zdir = 'y'
            x_shift, y_shift = -0.1, 0
        else:
            raise RuntimeError("Unknown arrow direction")
        if shift is None:
            shift = (x_shift, y_shift, -0.2)

        arrow = plotutils.Arrow3D((start[0], end[0]),
                                  (start[1], end[1]),
                                  (start[2], end[2]),
                                  arrowstyle='->,head_width=.25', mutation_scale=30, lw=4, zorder=30)
        ax.add_patch(arrow)
        middle = (p0 + p1) / 2
        ax.text(middle[0] + shift[0], middle[1] + shift[1], middle[2] + shift[2],
                name, va='center', ha='center',
                zdir=zdir,
                bbox={'facecolor': color, 'edgecolor': color}, zorder=30)

    # I added spacing \, before j because it looked connected to (
    _plot_arrow((0, 0), (2, 0), r'$\mathbb{X}_{y}\left(\,x,y\right)$')
    _plot_arrow((0, 0), (0, 2), r'$\mathbb{X}_{x}\left(\,x,y\right)$')
    _plot_arrow((0, 2), (0, 4), r'$\mathbb{X}_{x}\left(\,x+\chi,y\right)$')
    _plot_arrow((2, 0), (4, 0), r'$\mathbb{X}_{y}\left(\,x,y+\xi\right)$')
    _plot_arrow((2, 0), (2, 2), r'$\mathbb{X}_{x}\left(\,x,y+\xi\right)$',
                shift=(0.15, 0.35, 0))

    AE = dots[:, indexes[0, 2]] - dots[:, indexes[0, 0]]
    AJ = dots[:, indexes[2, 0]] - dots[:, indexes[0, 0]]

    Normal = np.cross(AE, AJ) * 0.55
    start = dots[:, indexes[0, 0]]
    end = start + Normal
    arrow = plotutils.Arrow3D((start[0], end[0]),
                              (start[1], end[1]),
                              (start[2] + 0.2, end[2]),
                              arrowstyle='->,head_width=.25', mutation_scale=30, lw=4, zorder=30)
    ax.add_patch(arrow)
    ax.text(end[0], end[1] - 0.25, end[2] + 0.07,
            r'$\vec{N}$', va='center', ha='center',
            bbox={'facecolor': color, 'edgecolor': color}, zorder=30)

    plotutils.remove_tick_labels(ax)
    ax.set_aspect('equal', adjustable='datalim')
    plotutils.set_zoom_by_limits(ax, 1.25)
    xlim = ax.get_xlim()
    xlim_shift = 0.3
    ax.set_xlim3d(xlim[0] + xlim_shift, xlim[1] + xlim_shift)

    plotutils.set_3D_labels(ax)
    ax.set_zlabel('Z', labelpad=-10.4)

    # fig.tight_layout()
    mpl.rcParams["savefig.bbox"] = "standard"

    ax.set_position([0.0, -0.04, 1.0, 1.3])

    plt.savefig(os.path.join(FIGURES_PATH, "sffvectors.pdf"), pad_inches=0.2)
    plt.savefig(os.path.join(FIGURES_PATH, "sffvectors.svg"), pad_inches=0.2)

    plt.show()


def main():
    plot_sff_vectors()


if __name__ == '__main__':
    main()
