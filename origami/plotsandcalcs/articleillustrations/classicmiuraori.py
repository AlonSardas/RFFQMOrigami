import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from origami import origamiplots, quadranglearray
from origami.RFFQMOrigami import RFFQM
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.plotsandcalcs.articleillustrations import FIGURES_PATH
from origami.quadranglearray import QuadrangleArray
from origami.utils import plotutils


def plot_placed_horizontally():
    angle = 1.1
    ls = np.ones(6) * 1.5
    cs = np.ones(8)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)

    fig = plt.figure(figsize=(10, 5))
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=39, azim=-57,
                                 computed_zorder=False)

    ori.dots.center()
    panels, surf = ori.dots.plot_with_wireframe(ax, alpha=0.6)
    surf.set_alpha(0)
    panels.set_fc('C7')
    panels.set_zorder(-10)
    origamiplots.draw_creases(ori.dots, 1, ax)

    ori.set_gamma(ori.calc_gamma_by_omega(0.4))
    ori.dots.dots[0, :] += 8
    # ori.dots.dots[2, :] -= 2
    panel_color = 'C1'
    ori.dots.plot(ax, panel_color, alpha=1.0, edge_alpha=1)
    # surf.set_alpha(0.3)

    ori.set_gamma(ori.calc_gamma_by_omega(1.5))
    ori.dots.dots[0, :] += 15
    # ori.dots.dots[2, :] -= 4
    ori.dots.plot(ax, panel_color, alpha=1.0, edge_alpha=1)
    # wire.set_alpha(0.2)

    ori.set_gamma(ori.calc_gamma_by_omega((np.pi - 0.12)))
    ori.dots.dots[0, :] += 19
    # ori.dots.dots[2, :] -= 6
    # ori.dots.plot(ax, panel_color, alpha=1, edge_alpha=1)
    ax.computed_zorder=False
    quadranglearray.plot_panels_manual_zorder(ori.dots, ax, panel_color, edge_color='g', z_shift=-0.01, edge_width=1)
    # ori.dots.plot(ax, panel_color, alpha=1, edge_alpha=1)
    # surf.set_alpha(0.1)

    plotutils.set_labels_off(ax)
    ax.set_aspect("equal")
    ax.set(zticks=[-0.5, 0, 0.5])
    plotutils.remove_tick_labels(ax)

    # ax.axis('off')
    # ax.grid(True)
    # ax.set_zmargin(-0.4)
    y_pad = 0.4
    ax.set_position([0.05, -y_pad, 0.9, 1 + 2 * y_pad])

    # fig.tight_layout()
    mpl.rcParams["savefig.bbox"] = "standard"
    # fig.set_facecolor("aliceblue")
    fig.savefig(os.path.join(FIGURES_PATH, "classical-miura-ori.svg"))
    fig.savefig(os.path.join(FIGURES_PATH, "classical-miura-ori.pdf"))
    fig.savefig(os.path.join(FIGURES_PATH, "classical-miura-ori.png"), dpi=300)

    plt.show()


def plot_stacked_in_layers():
    angle = 1.1
    ls = np.ones(6) * 1.5
    cs = np.ones(8)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)

    fig = plt.figure(figsize=(10, 10))
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-60, elev=32, computed_zorder=False)

    ori.dots.center()
    panels, surf = ori.dots.plot_with_wireframe(ax, alpha=0.6)
    surf.remove()
    panels.set_fc('C7')
    panels.set_zorder(-10)
    origamiplots.draw_creases(ori, 1, ax)
    ori.set_gamma(ori.calc_gamma_by_omega(0.5))
    ori.dots.dots[2, :] += 2
    _, surf = ori.dots.plot_with_wireframe(ax, alpha=0.6)

    ori.set_gamma(ori.calc_gamma_by_omega(1.5))
    ori.dots.dots[2, :] += 5
    panels, surf = ori.dots.plot_with_wireframe(ax, alpha=0.5)
    panels.set_fc('C1')
    # surf.set_alpha(0.1)
    # panels.set_zorder(-10)

    ori.set_gamma(ori.calc_gamma_by_omega((np.pi - 0.01)))
    ori.dots.dots[2, :] += 13
    panels, surf = ori.dots.plot_with_wireframe(ax, alpha=0.3)
    panels.set_fc('C0')
    surf.set_alpha(0.1)

    plotutils.set_labels_off(ax)
    ax.set_aspect("equal")
    ax.set(zticks=[-0.5, 0, 0.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # ax.axis('off')
    # ax.grid(True)
    # ax.set_zmargin(-0.4)
    y_pad = 0.4
    ax.set_position([0.05, -y_pad, 0.9, 1 + 2 * y_pad])

    mpl.rcParams["savefig.bbox"] = "standard"
    fig.savefig(os.path.join(FIGURES_PATH, "classic-miura-ori2.svg"))
    fig.savefig(os.path.join(FIGURES_PATH, "classic-miura-ori2.pdf"))

    plt.show()


def main():
    plot_placed_horizontally()
    # plot_stacked_in_layers()


if __name__ == "__main__":
    main()
