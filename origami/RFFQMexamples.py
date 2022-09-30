import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami import zigzagplots, continuousorigami
from origami.RFFQMOrigami import RFFQM
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import QuadrangleArray
from origami.utils import linalgutils, plotutils


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
    # dots.assert_valid()
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
        quads = origami.set_omega(omega, should_center=True)
        # dots.center()
        quads.plot(ax, alpha=0.85)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

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
    angle = 2
    # ls = np.ones(5) * 2
    ls = [2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4]
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


def create_sphere_interactive():
    zigzag_angle = 2.5
    xs = np.arange(0.05, 0.9, 0.03)
    ys = 1 - np.sqrt(1 - xs ** 2)

    ls_sphere, _, _ = continuousorigami.follow_curve(xs, ys, (np.pi - zigzag_angle) / 2)
    ls = np.append(ls_sphere, np.ones(20) * 0.01)

    angle = 0.51 * np.pi
    cs = np.ones(20) * 0.01

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, :] += 0.05

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    origami = RFFQM(quads)
    plot_interactive(origami)


def create_sphere():
    zigzag_angle = 2.5
    xs = np.arange(0.05, 0.9, 0.03)
    ys = 1 - np.sqrt(1 - xs ** 2)

    ls_sphere, _, _ = continuousorigami.follow_curve(xs, ys, (np.pi - zigzag_angle) / 2)
    ls = np.append(ls_sphere, np.ones(20) * 0.01)

    angle = 0.51 * np.pi
    cs = np.ones(20) * 0.015

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, :] += 0.05
    # angles_left[:, 0] += 0.1
    # angles_left[0, 6] = 0.5107 * np.pi

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    origami = RFFQM(quads)

    origami.set_omega(zigzag_angle)

    # This aligns the sphere
    dots = origami.dots.dots
    indexes = origami.dots.indexes
    v1 = dots[:, indexes[50, 0]] - dots[:, indexes[50, 5]]
    v2 = dots[:, indexes[50, 5]] - dots[:, indexes[50, 10]]
    n = np.cross(v1, v2)
    R = linalgutils.create_alignment_rotation_matrix(n, [0, 0, 1])
    origami.dots.dots = R @ dots

    dots = origami.dots.dots
    min_z = dots[2, :].min()
    dots[2, :] += 1 - min_z - 2

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    origami.dots.plot(ax, alpha=0.6)

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0.0, alpha=0.2, color='r')

    # Calculate the distance from targeted sphere
    # relevant_dots = dots[:, indexes[1:len(ls_sphere):2, 1::2].flat]
    relevant_dots = dots[:, indexes[1:len(ls_sphere):2, 1::2].flat]
    d = np.sqrt(np.sum(relevant_dots ** 2, axis=0))
    print(d.shape)
    print(d)
    print(f'among {len(d)} points, mean: {d.mean()}, and std: {d.std()}')

    plotutils.set_axis_scaled(ax)

    plt.show()


# noinspection SpellCheckingInspection
def create_MARS_Barreto():
    """
    For example of MARS_Barreto, see:

    Paulo Taborda Barreto. Lines meeting on a surface: The “Mars” paperfold-
    ing. In Koryo Miura, editor, Proceedings of the 2nd International Meeting of
    Origami Science and Scientific Origami, pages 323–331, Otsu, Japan, November–
    December 1994.
    """
    angle = 0.7 * np.pi
    ls = np.ones(10)
    cs = np.ones(10)

    angles_left = np.ones((2, len(ls) + 1)) * np.pi / 2
    angles_bottom = np.ones((2, len(ls))) * np.pi / 2

    angles_left[0, 1::2] = angle
    angles_left[1, 0::2] = angle
    angles_bottom[0, 0::2] = np.pi - angle
    angles_bottom[1, 1::2] = angle

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
    # create_sphere_interactive()
    # create_sphere()
    create_MARS_Barreto()


if __name__ == '__main__':
    main()
