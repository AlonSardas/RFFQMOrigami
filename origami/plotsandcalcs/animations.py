import numpy as np
from matplotlib import pyplot as plt, animation

from origami import marchingalgorithm, quadranglearray, RFFQMOrigami, origamiplots
from origami.utils import plotutils


def main():
    angle = 0.7 * np.pi  # The base angle of the Miura-Ori
    ls = np.ones(4)
    cs = np.ones(4)

    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)
    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)

    quads = quadranglearray.dots_to_quadrangles(dots, indexes)
    ori = RFFQMOrigami.RFFQM(quads)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    artists = []
    omegas = np.linspace(0.05, np.pi - 0.05, 100)
    for i, omega in enumerate(omegas):
        ori.set_gamma(omega)
        # poly3d = ori.dots.plot(ax, panel_color='C1', alpha=1.0,
        #                        edge_alpha=1.0, edge_color='g')
        # artists.append([poly3d])
        panels = quadranglearray.plot_panels_manual_zorder(ori.dots, ax, reverse_x_dir=True)
        artists.append(panels)

    ax.set_aspect('equal')
    plotutils.remove_tick_labels(ax)

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=20)
    ani.save(filename="/tmp/html_example.html", writer="html")
    ani.save(filename="/tmp/ffmpeg_example.mkv", writer="ffmpeg")

    plt.show()


if __name__ == '__main__':
    main()
