import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami.plotsandcalcs.masterpresentation import FIGURES_PATH
from origami.utils import plotutils


def plot_smooth_vase():
    # See Mathematica code from https://commons.wikimedia.org/wiki/File:Rotationskoerper_animation.gif
    def f(phi, z):
        x = (2 + np.sin(z)) * np.cos(phi)
        y = (2 + np.sin(z)) * np.sin(phi)
        return x, y, z

    phis = np.linspace(0, 2 * np.pi, 100)
    zs = np.linspace(0, 2 * np.pi, 40)
    phis, zs = np.meshgrid(phis, zs)
    xs, ys, zs = f(phis, zs)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=25, azim=-130)
    ax.plot_surface(xs, ys, zs, color='#4EA72E', antialiased=False, linewidth=0)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    # plotutils.set_3D_labels(ax)

    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'smooth-vase.svg'),
                               0.7, 0.9, transparent=True)

    # fig.savefig(os.path.join(FIGURES_PATH, 'smooth-vase.png'))
    # fig.savefig(os.path.join(FIGURES_PATH, 'smooth-vase.svg'))

    # plt.show()


def main():
    plot_smooth_vase()


if __name__ == '__main__':
    main()
