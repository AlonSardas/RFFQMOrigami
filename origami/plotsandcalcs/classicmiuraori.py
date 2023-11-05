import os

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from origami.utils import plotutils
import origami.plotsandcalcs
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.quadranglearray import QuadrangleArray
from origami.RFFQMOrigami import RFFQM

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
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-60, elev=32)

    ori.dots.center()
    ori.dots.plot(ax, 0.6)
    ori.set_gamma(2)
    ori.dots.dots[0, :] += 8
    ori.dots.plot(ax, 0.6)
    ori.set_gamma(np.pi - 0.05)
    ori.dots.dots[0, :] += 13
    ori.dots.plot(ax, 0.6)

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
