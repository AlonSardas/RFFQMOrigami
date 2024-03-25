"""
Produce design of simple example to be used in a laser cutter
"""

import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import origami.plotsandcalcs
from origami import origamiplots
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.plotsandcalcs.alternating import curvatures
from origami.plotsandcalcs.alternating.utils import create_F_from_list, create_MM_from_list, create_perturbed_origami, \
    create_perturbed_origami_by_list, plot_perturbations, plot_perturbations_by_list
from origami.quadranglearray import QuadrangleArray
from origami.RFFQMOrigami import RFFQM
from origami.utils import linalgutils, plotutils

FIGURES_PATH = os.path.join(
    origami.plotsandcalcs.BASE_PATH, "RFFQM", "OrigamiProduction", "Designs"
)


def draw_miura_cell():
    angle = 1.1
    ls = np.ones(2) * 1
    cs = np.ones(2) * 1.3

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)

    fig, ax = plot_for_printing(ori)

    fig.savefig(os.path.join(FIGURES_PATH, 'simple-cell.svg'), transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'simple-cell.ps'), transparent=True)

    plt.show()


def draw_miura_ori():
    angle = 1.1
    ls = np.ones(10) * 1
    cs = np.ones(10) * 1.3
    # cs[-1] = 0.5

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)

    # origamiplots.plot_interactive(ori)
    # fig, ax = origamiplots.plot_crease_pattern(ori)
    # fig.savefig(os.path.join(FIGURES_PATH, 'Miura-Ori-pattern.png'), transparent=False)

    prepare_patterns_for_printing(quads, 'MiuraOri2', file_format='ps')

    plt.show()


def draw_wavy_pattern():
    rows, cols = 14, 18
    L0 = 8
    C0 = 8.5
    chi = 1 / cols * 2
    xi = 1 / rows * 2
    L_tot = L0 / xi
    C_tot = C0 / chi

    kx_func = lambda t: 1 / C_tot * 4.0 * np.tanh(-(t - 0.5) * 3)
    ky_func = lambda t: 1 / L_tot * 1.2

    delta0 = 0.40
    Delta0 = 0.7

    W0 = -2.1
    theta = 1.1

    xs, deltas = curvatures.get_deltas_for_kx(L_tot, C_tot, W0, theta, kx_func, delta0, chi)
    ys, Deltas = curvatures.get_Deltas_for_ky(L_tot, C_tot, W0, theta, ky_func, Delta0, xi)
    fig, axes = plt.subplots(2)
    plot_perturbations_by_list(axes, xs, deltas, ys, Deltas)

    ori = create_perturbed_origami_by_list(
        theta, L0, C0, deltas, Deltas)
    quads = ori.dots

    rot_angle = 0.41
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = quads.dots.astype('float64')

    # plot_for_printing(ori)
    # plt.show()

    prepare_patterns_for_printing(ori.dots, 'wavy')
    origamiplots.plot_interactive(ori)
    # plt.show()


def draw_MARS_pattern():
    angle = 0.7 * np.pi
    ls = np.ones(8)
    cs = np.ones(8)

    angles_left = np.ones((2, len(ls) + 1)) * np.pi / 2
    angles_bottom = np.ones((2, len(ls))) * np.pi / 2

    angles_left[0, 1::2] = np.pi - angle
    angles_left[1, 0::2] = np.pi - angle
    angles_bottom[0, 0::2] = angle
    angles_bottom[1, 1::2] = np.pi - angle

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)

    quads = ori.dots

    rot_angle = -0.25
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = np.array(quads.dots, 'float64')

    origamiplots.plot_interactive(ori)
    # return

    plot_for_printing(ori)
    plt.show()

    prepare_patterns_for_printing(ori.dots, 'MARS')


def draw_spherical_cap():
    kx = -0.10
    ky = -0.10

    # L0 = 1.0
    # C0 = 1
    W0 = 2.2
    theta = 1.2

    Delta0 = 0.6
    delta0 = 0.4

    Nx, Ny = 6, 7
    L_tot, C_tot = 15, 15

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    # xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    fig, axes = plt.subplots(2)
    plot_perturbations(axes, delta_func, Delta_func, Nx, Ny)
    plt.show()

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)

    quads = ori.dots
    rot_angle = delta0
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = np.array(quads.dots, 'float64')

    prepare_patterns_for_printing(ori.dots, 'spherical-cap-3')

    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    origamiplots.plot_interactive(ori)


def draw_spherical_cap_small():
    kx = -0.12
    ky = -0.12

    W0 = 2.2
    theta = 1.2

    Delta0 = 0.8
    delta0 = 0.6

    Nx, Ny = 4, 4
    L_tot, C_tot = 15, 15

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    # xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    fig, axes = plt.subplots(2)
    plot_perturbations(axes, delta_func, Delta_func, Nx, Ny)
    # plt.show()

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)
    _rotate_dots(ori.dots, delta0)

    prepare_patterns_for_printing(ori.dots, 'spherical-cap-small')
    fig, ax = plot_for_printing(ori)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-small-all.svg'))

    ori.set_gamma(ori.calc_gamma_by_omega(-W0))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=26, azim=-138)
    ori.dots.plot(ax, 'C1', edge_color='k')
    plotutils.set_axis_scaled(ax)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-small-folded.svg'))

    origamiplots.plot_interactive(ori)


def _rotate_dots(quads, rot_angle):
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots


def draw_saddle():
    kx = 0.10
    ky = -0.10

    W0 = 2.2
    theta = 1.2

    Delta0 = +0.5
    delta0 = -0.4

    Nx, Ny = 6, 7
    L_tot, C_tot = 15, 15

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    # xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    fig, axes = plt.subplots(2)
    plot_perturbations(axes, delta_func, Delta_func, Nx, Ny)
    plt.show()

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)

    quads = ori.dots
    rot_angle = delta0
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = np.array(quads.dots, 'float64')

    prepare_patterns_for_printing(ori.dots, 'saddle')

    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    origamiplots.plot_interactive(ori)


def prepare_patterns_for_printing(quads: QuadrangleArray, name, file_format='ps'):
    """
    Creates 2 images that are used to dictate the mountain and valley creases of the pattern.
    The 1st image contains only the mountain creases with a rectangular frame around the entire pattern.
    After carving the mountains, the rectangular frame should be cut and the entire pattern is flipped.
    Next the 2nd image is used to carve the valleys.

    Args:
        quads: The flat quadrangle array of the crease pattern
        name: The base name for the output files. The files are saved in FIGURES_PATH
        file_format: The file format used to save the image
    """
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)

    ax.set_axis_off()

    creases = origamiplots.draw_inner_creases(quads, ax)
    for crease, MV, HV in creases:
        if HV == 'V':
            crease.set_color('k')
        else:
            crease.set_color('silver')

    for crease, MV, HV in creases:
        if MV == -1:
            # Make valleys invisible
            crease.set_visible(False)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    frame = ax.plot((x_lim[1], x_lim[0], x_lim[0], x_lim[1], x_lim[1]),
                    (y_lim[1], y_lim[1], y_lim[0], y_lim[0], y_lim[1]), '-r')[0]

    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, f'{name}1.{file_format}'), transparent=True, metadata={'DocumentPaperSizes': None})

    for crease, MV, HV in creases:
        visible = crease.get_visible()
        crease.set_visible(not visible)  # Flip visibility

    x_lim = ax.get_xlim()

    origamiplots.draw_frame_creases(quads, ax, 'r')

    # flip horizontally
    ax.set_xlim(x_lim[1], x_lim[0])
    frame.set_color('#FFFF00')

    fig.savefig(os.path.join(FIGURES_PATH, f'{name}2.{file_format}'), transparent=True)
    return fig, ax


def plot_for_printing(ori: RFFQM) -> Tuple[Figure, Axes]:
    if not np.isclose(ori.gamma, 0):
        raise ValueError(f"The given origami is not flat. gamma={ori.gamma}")
    if not np.all(np.isclose(ori.dots.dots[2, :], 0)):
        raise ValueError(f"The given origami is not on XY plane!")
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    origamiplots.draw_inner_creases(ori.dots, ax, -1)

    origamiplots.draw_frame_creases(ori.dots, ax, 'k')

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def main():
    # draw_miura_cell()
    # draw_miura_ori()
    # draw_wavy_pattern()
    # draw_MARS_pattern()
    draw_spherical_cap_small()
    # draw_saddle()


if __name__ == '__main__':
    main()
