import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami import origamiplots, marchingalgorithm, quadranglearray, zigzagmiuraori
from origami.RFFQMOrigami import RFFQM
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.plotsandcalcs.masterpresentation import FIGURES_PATH
from origami.quadranglearray import QuadrangleArray
from origami.utils import plotutils


def create_radial():
    angle = 1.3
    ls = [4, 1, 4, 1.5, 4, 2, 4, 3, 2, 4, 1, 4]
    cs = np.ones(8)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, :] += 0.05
    angles_left[:, :] += 0.05

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)

    fig, ax = origamiplots.plot_crease_pattern(ori, background_color='0.9')
    fig.set_size_inches(5, 6)
    fig.savefig(os.path.join(FIGURES_PATH, 'radial-example-flat.png'), transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'radial-example-flat.svg'))

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=45, azim=149)

    ori.set_gamma(1.5)
    ori.dots.plot(ax, panel_color='C1', alpha=1)

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    # fig.savefig(os.path.join(FIGURES_PATH, 'radial-example-folded.svg'))
    # fig.savefig(os.path.join(FIGURES_PATH, 'radial-example-folded.png'), dpi=300)
    plotutils.save_fig_cropped(fig,
                               os.path.join(FIGURES_PATH, 'radial-example-folded.svg'),
                               0.7, 0.65, translate_x=0.3, translate_y=-0.3, transparent=True)

    plt.show()


def create_cylindrical():
    angle = 0.7 * np.pi
    ls = np.ones(14)
    cs = np.ones(10)
    ls[::2] += 0.1 * np.arange(len(ls) // 2)

    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)

    angles_left[:, 1::2] += 0.3
    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    quads = quadranglearray.dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)

    fig, ax = origamiplots.plot_crease_pattern(ori, background_color=True)
    fig.set_size_inches(4, 7)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-small-flat.svg"), bbox_inches='tight', transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-small-flat.png"), bbox_inches='tight')

    ori.set_gamma(-1.9)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=24, azim=-141)
    ori.dots.plot(ax, panel_color='C1', alpha=1.0)
    ax.set_axis_off()
    plotutils.set_axis_scaled(ax)

    fig.tight_layout()
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, "cylinder-small-folded.svg"),
                               0.82, 0.78, transparent=True)

    plt.show()


def create_cylindrical2():
    cols = 20
    ls = np.concatenate([np.ones(5) * 1.9, 2.1 + np.arange(15) * 0.3])
    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(len(ls) + 1) * 0.2 * np.pi
    angles[::2] += 0.1 * np.pi
    # cs = np.ones(10) * 0.1
    dx = 2
    dots = zigzagmiuraori.create_zigzag_dots(angles, cols, ls, dx)
    zigzag = zigzagmiuraori.ZigzagMiuraOri(dots, len(ls) + 1, cols)
    ori = RFFQM(zigzag.get_quads())

    # origamiplots.plot_interactive(ori)

    fig, ax = origamiplots.plot_crease_pattern(ori, background_color=True)
    fig.set_size_inches(4, 7)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-flat2.svg"), bbox_inches='tight', transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-flat2.png"), bbox_inches='tight')

    ori.set_gamma(-2.1)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=16, azim=-148)
    ax.computed_zorder = False
    # ori.dots.plot(ax, panel_color='C1', alpha=1.0)
    quadranglearray.plot_panels_manual_zorder(ori.dots, ax, panel_color='C1', alpha=1.0, edge_color='g')
    # ori.dots.plot(ax, panel_color='C1', alpha=1.0)
    ax.set_axis_off()
    plotutils.set_axis_scaled(ax)

    fig.tight_layout()
    # plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, "cylinder-folded2.png"),
    #                            0.82, 0.78)
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, "cylinder-folded2.svg"),
                               0.88, 0.78, transparent=True)

    plt.show()


def main():
    # create_radial()
    # create_cylindrical()
    create_cylindrical2()


if __name__ == '__main__':
    main()
