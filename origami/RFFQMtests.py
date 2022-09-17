import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from origami import zigzagplots
from origami.RFFQMOrigami import RFFQM
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import QuadrangleArray


def create_basic_crease():
    n = 6
    dx = 1
    dy = 2

    angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    # angles = np.array([0.2, 0.3, 0.2]) * np.pi

    dots = zigzagplots.create_zigzag_dots(angles, n, dy, dx)
    rows, cols = len(angles), n

    quads = QuadrangleArray(dots, rows, cols)
    origami = RFFQM(quads)
    plot_interactive(origami)


def add_slider(ax, origami: RFFQM):
    init_omega = 0.5
    dots = origami.set_omega(init_omega)
    dots.assert_valid()
    dots.plot(ax)  # We plot to find the limits of the axis to use

    lim = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

    # Make a horizontal slider to control the frequency.
    omega_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    omega_slider = matplotlib.widgets.Slider(
        ax=omega_slider_ax,
        label='Omega',
        valmin=0,
        valmax=np.pi,
        valinit=init_omega,
    )

    def update_omega(omega):
        ax.clear()
        dots = origami.set_omega(omega, should_center=False)
        dots.plot(ax, alpha=0.7)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim / 2, lim / 2)

    omega_slider.on_changed(update_omega)
    return omega_slider


def plot_interactive(origami: RFFQM):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    # We need to assign the return value to variable slider for the slider object
    # stay alive and keep functioning
    # noinspection PyUnusedLocal
    slider = add_slider(ax, origami)

    plt.show()


def create_radial():
    angle = 1
    ls = np.ones(5) * 2
    cs = np.ones(20)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, :] += 0.1
    # angles_left[:, 0] += 0.1

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    origami = RFFQM(quads)
    plot_interactive(origami)


def create_sphere():
    angle = 0.51 * np.pi
    ls = [1, 2, 1, 3, 1, 4, 1, 5, 1, 7, 1, 1, 7, 1, 6, 1, 5, 1, 4]
    cs = np.ones(20)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, :] += 0.1
    # angles_left[:, 0] += 0.1

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    origami = RFFQM(quads)
    plot_interactive(origami)


def main():
    # logutils.enable_logger()
    # create_basic_crease()
    # create_radial()
    create_sphere()


if __name__ == '__main__':
    main()
