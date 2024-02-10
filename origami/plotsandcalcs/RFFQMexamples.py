import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import origami
import origami.plotsandcalcs
from origami import origamimetric
from origami.RFFQMOrigami import RFFQM
from origami.origamiplots import plot_interactive
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import QuadrangleArray, plot_flat_quadrangles
from origami.utils import linalgutils, zigzagutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM', 'Compatibility', 'Figures')


def create_basic_crease():
    n = 6
    dx = 1
    dy = 2

    angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    # angles = np.array([0.2, 0.3, 0.2]) * np.pi

    dots = origami.zigzagmiuraori.create_zigzag_dots(angles, n, dy, dx)
    rows, cols = len(angles), n

    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)
    plot_interactive(ori)


def create_radial_simple():
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
    ori = RFFQM(quads)
    plot_interactive(ori)


def create_radial():
    angle = 1.3
    # ls = np.ones(5) * 2
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

    fig, ax = plot_flat_quadrangles(quads)
    ax.set_axis_off()
    ax.set_aspect('equal')
    # ax.set_box_aspect(None, zoom=1.7)
    ax.dist = 5
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'radial-example-flat.svg'))

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-156, elev=-140)

    ori.set_gamma(1.5)
    ori.dots.plot_with_wireframe(ax, alpha=0.4)
    
    # ax.set_ylim(-10, 15)
    # ax.set_box_aspect(None, zoom=2)
    # ax.dist is deprecated but I couldn't combine set_box_aspect with set_aspect('equal')
    ax.dist = 8.5
    ax.set_aspect('equal')
    # ax.set(xticks=[], yticks=[], zticks=[])
    ax.set(zticks=[-3, 0, 3])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    # ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'radial-example-folded.svg'))

    plt.show()


def create_sphere_interactive():
    zigzag_angle = 2.5
    xs = np.arange(0.05, 0.9, 0.03)
    ys = 1 - np.sqrt(1 - xs ** 2)

    ls_sphere, _, _ = zigzagutils.follow_curve(xs, ys, (np.pi - zigzag_angle) / 2)
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

    ls_sphere, _, _ = zigzagutils.follow_curve(xs, ys, (np.pi - zigzag_angle) / 2)
    ls = np.append(ls_sphere, np.ones(20) * 0.01)
    print(ls_sphere)
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
    ori = RFFQM(quads)

    ori.set_gamma(zigzag_angle)

    # Trying to verify that the curvature is indeed constant:
    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
    fig, ax = plt.subplots()
    print(Ks.shape)
    print(Ks[5:-5, 5:-5])
    # ax.plot([1, 4, 5, 2], '.')
    # ax.plot(Ks[4:len(ls_sphere)//2, 1:8].flat, '.')
    ax.plot(Ks[5:, 1:].flat, '.')
    # print(Ks.transpose())

    # This aligns the sphere
    dots = ori.dots.dots
    indexes = ori.dots.indexes
    v1 = dots[:, indexes[50, 0]] - dots[:, indexes[50, 5]]
    v2 = dots[:, indexes[50, 5]] - dots[:, indexes[50, 10]]
    n = np.cross(v1, v2)
    R = linalgutils.create_alignment_rotation_matrix(n, [0, 0, 1])
    ori.dots.dots = R @ dots

    dots = ori.dots.dots
    min_z = dots[2, :].min()
    dots[2, :] += 1 - min_z - 2

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-120, elev=23)
    ori.dots.plot(ax, alpha=0.7)

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0.0, alpha=0.1, color='r')

    # Calculate the distance from targeted sphere
    # relevant_dots = dots[:, indexes[1:len(ls_sphere):2, 1::2].flat]
    relevant_dots = dots[:, indexes[1:len(ls_sphere):2, 1::2].flat]
    d = np.sqrt(np.sum(relevant_dots ** 2, axis=0))
    # print(d.shape)
    # print(d)
    print(f'among {len(d)} points, mean: {d.mean()}, and std: {d.std()}')

    lim = 0.75
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_aspect('equal')
    # plotutils.set_axis_scaled(ax)

    fig.savefig(os.path.join(FIGURES_PATH, 'sphere-radial.svg'))
    fig.savefig(os.path.join(FIGURES_PATH, 'sphere-radial.png'))

    plt.show()


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

    print(angles_left, angles_bottom)

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
