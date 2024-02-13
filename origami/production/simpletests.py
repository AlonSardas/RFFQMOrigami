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

import origami.plotsandcalcs
from origami import origamiplots
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.plotsandcalcs.alternating import curvatures
from origami.plotsandcalcs.alternating.utils import create_F_from_list, create_MM_from_list, create_perturbed_origami, \
    create_perturbed_origami_by_list
from origami.quadranglearray import QuadrangleArray
from origami.RFFQMOrigami import RFFQM
from origami.utils import linalgutils

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
    ls = np.ones(6) * 1
    cs = np.ones(6) * 1.3
    cs[-1] = 0.5

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    ori = RFFQM(quads)

    # origamiplots.plot_interactive(ori)
    fig, ax = origamiplots.plot_crease_pattern(ori)
    fig.savefig(os.path.join(FIGURES_PATH, 'Miura-Ori-pattern.png'), transparent=False)

    prepare_patterns_for_printing(quads, 'MiuraOri', format='ps')

    plt.show()


def draw_wavy_pattern():
    rows, cols = 14, 16
    L0 = 7
    C0 = 8.5
    chi = 1 / cols * 2
    xi = 1 / rows * 2
    L_tot = L0 / xi
    C_tot = C0 / chi

    kx_func = lambda t: 1 / C_tot * 4.2 * np.tanh(-(t - 0.5) * 3)
    ky_func = lambda t: 1 / L_tot * 1.2

    delta0 = 0.40
    Delta0 = 4.0

    W0 = -2.1
    theta = 1.1

    xs, deltas = curvatures.get_deltas_for_kx(L_tot, C_tot, W0, theta, kx_func, delta0, chi)
    ys, Deltas = curvatures.get_Deltas_for_ky(L_tot, C_tot, W0, theta, ky_func, Delta0 / L0, xi)

    ori = create_perturbed_origami_by_list(
        theta, L0, C0, deltas, Deltas)
    quads = ori.dots

    rot_angle = 0.41
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = quads.dots.astype('float64')

    plot_for_printing(ori)
    plt.show()

    prepare_patterns_for_printing(ori.dots, 'wavy')
    plt.show()


def draw_MARS_pattern():
    angle = 0.7 * np.pi
    ls = np.ones(5)
    cs = np.ones(5)

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
    ori = RFFQM(quads)

    quads = ori.dots

    rot_angle = -0.25
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = np.array(quads.dots, 'float64')

    origamiplots.plot_interactive(ori)
    return

    plot_for_printing(ori)
    plt.show()

    prepare_patterns_for_printing(ori.dots, 'MARS')


def draw_spherical_cap():
    rows, cols = 20, 30

    kx = 0.10
    ky = 0.10

    L0 = 1.0
    C0 = 1
    W0 = 2.4
    theta = 1.1

    Delta0 = -0.5
    delta0 = -0.2

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = L0 * Ny, C0 * Nx

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    # xs, deltas, ys, Deltas = get_pert_list_by_func(delta_func, Delta_func, Nx, Ny)
    # plot_perturbations_by_list(xs, deltas, ys, Deltas)

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)

    quads = ori.dots
    rot_angle = delta0
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots
    quads.dots = np.array(quads.dots, 'float64')
    
    prepare_patterns_for_printing(ori.dots, 'spherical-cap-2')


def prepare_patterns_for_printing(quads: QuadrangleArray, name, format='ps'):
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)

    ax.set_axis_off()

    creases = origamiplots.draw_inner_creases(quads, ax)
    for crease in creases:
        r, g, b, a = crease.get_color()
        if b == 1.0:
            # Make valleys invisible
            crease.set_visible(False)
        crease.set_color('k')

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    frame = ax.plot((x_lim[1], x_lim[0], x_lim[0], x_lim[1], x_lim[1]),
                    (y_lim[1], y_lim[1], y_lim[0], y_lim[0], y_lim[1]), '-r')[0]

    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, f'{name}1.{format}'), transparent=True, metadata={'DocumentPaperSizes': None})

    for crease in creases:
        visible = crease.get_visible()
        crease.set_visible(not visible)  # Flip visibility

    x_lim = ax.get_xlim()

    origamiplots.draw_frame_creases(quads, ax, 'r')

    # flip horizontally
    ax.set_xlim(x_lim[1], x_lim[0])
    frame.set_color('#FFFF00')

    fig.savefig(os.path.join(FIGURES_PATH, f'{name}2.{format}'), transparent=True)
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
    draw_spherical_cap()


if __name__ == '__main__':
    main()
