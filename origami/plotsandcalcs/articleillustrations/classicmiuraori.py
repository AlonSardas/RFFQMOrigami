import os

import matplotlib as mpl
from matplotlib.colors import LightSource
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import origami.plotsandcalcs
from origami import origamiplots
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.quadranglearray import QuadrangleArray
from origami.RFFQMOrigami import RFFQM
from origami.utils import plotutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH, "RFFQM/Figures")


def main():
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
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-60, elev=32, computed_zorder=False)

    ori.dots.center()
    panels, surf = ori.dots.plot(ax, 0.6)
    surf.set_alpha(0)
    panels.set_fc('C7')
    panels.set_zorder(-10)
    origamiplots.draw_creases(ori, 1, ax)
    ori.set_gamma(ori.calc_gamma_by_omega(1))
    ori.dots.dots[0, :] += 8
    _, surf = ori.dots.plot(ax, 0.6)
    
    # surf.set_alpha(0)
    # origamiplots.draw_creases(ori, -1, ax)

    ori.set_gamma(ori.calc_gamma_by_omega((np.pi - 0.01)))
    ori.dots.dots[0, :] += 13
    panels, surf = ori.dots.plot(ax, 0.3)

    # this is used to set the graph color to blue
    blue = np.array([0., 0., 1.])
    zs = ori.dots.dots[2, ori.dots.indexes]
    rgb = np.tile(blue, (zs.shape[0], zs.shape[1], 1))

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
    ax.set_position([0.05, -y_pad, 0.9, 1+2*y_pad])

    # fig.tight_layout()
    mpl.rcParams["savefig.bbox"] = "standard"
    # fig.set_facecolor("aliceblue")
    fig.savefig(os.path.join(FIGURES_PATH, "classic-miura-ori.svg"))
    fig.savefig(os.path.join(FIGURES_PATH, "classic-miura-ori.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
