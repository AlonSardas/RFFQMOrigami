import os

import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d import Axes3D

from origami import plotsandcalcs, quadranglearray, origamiplots
from origami.alternatingpert.utils import create_perturbed_origami
from origami.utils import plotutils

PANELS_COLOR = 'C1'
EDGE_COLOR = 'g'
EDGE_ALPHA = 1.0
EDGE_WIDTH = 2


def animate_miura_ori():
    theta = 1.3
    Nx, Ny = 4, 4
    C_tot, L_tot = 5, 5
    W0 = 1.5
    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, None, None)
    ori.set_gamma(ori.calc_gamma_by_omega(0))

    # origamiplots.plot_interactive(ori)

    fig = plt.figure(figsize=(6, 3))
    ax: Axes3D = fig.add_subplot(111, projection="3d",
                                 elev=22, azim=-119)
    zoom = 1.9
    ax.set_position([-(zoom - 1 + 0.05) / 2, -(zoom - 1) / 2, zoom, zoom])
    # panels = ori.dots.plot(ax, panel_color=PANELS_COLOR,
    #                        edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA, edge_width=EDGE_WIDTH)
    # panels.remove()
    ax.set_axis_off()
    ax.computed_zorder = False

    artists = []
    do = 0.03
    # do = 0.33
    for omega in np.arange(0.05, W0, do):
        ori.set_gamma(omega)
        panels = ori.dots.plot(ax, panel_color=PANELS_COLOR,
                               edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA, edge_width=EDGE_WIDTH)
        # panels = quadranglearray.plot_panels_manual_zorder(ori.dots, ax)
        artists.append([panels])

    PAUSE_FRAMES = 35
    artists.extend([[panels]] * PAUSE_FRAMES)

    for omega in np.arange(W0, 2.72, do):
        ori.set_gamma(omega)
        panels = ori.dots.plot(ax, panel_color=PANELS_COLOR,
                               edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA, edge_width=EDGE_WIDTH)
        artists.append([panels])
        # panels = quadranglearray.plot_panels_manual_zorder(ori.dots, ax)
        # artists.append(panels)

    for omega in np.arange(2.72, np.pi - 0.03, do):
        ori.set_gamma(omega)
        panels = quadranglearray.plot_panels_manual_zorder(ori.dots, ax, panel_color=PANELS_COLOR,
                                                           edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA,
                                                           edge_width=EDGE_WIDTH)

        artists.append(panels)

    artists.extend([panels] * PAUSE_FRAMES)

    plotutils.set_axis_scaled(ax)

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=55)
    ani.save(filename=os.path.join(FILES_PATH, "miura-ori-anim.avi"), writer="ffmpeg")
    # ani.save(filename=os.path.join(FILES_PATH, "miura-ori-anim.mp4"), writer="ffmpeg")

    # plt.show()


def main():
    animate_miura_ori()


if __name__ == '__main__':
    main()
