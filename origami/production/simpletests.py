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
from origami.quadranglearray import QuadrangleArray
from origami.RFFQMOrigami import RFFQM

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
    fig.savefig(os.path.join(FIGURES_PATH, 'Miura-Ori1.ps'), transparent=True)

    for crease in creases:
        visible = crease.get_visible()
        crease.set_visible(not visible)  # Flip visibility

    x_lim = ax.get_xlim()

    origamiplots.draw_frame_creases(ori.dots, ax, 'r')

    # flip horizontally
    ax.set_xlim(x_lim[1], x_lim[0])
    frame.set_color('#FFFF00')

    fig.savefig(os.path.join(FIGURES_PATH, 'Miura-Ori2.ps'), transparent=True)

    plt.show()


def plot_for_printing(ori: RFFQM) -> Tuple[Figure, Axes]:
    if not np.isclose(ori.gamma, 0):
        raise ValueError(f"The given origami is not flat. gamma={ori.gamma}")
    if not np.all(np.isclose(ori.dots.dots[2, :], 0)):
        raise ValueError(f"The given origami is not on XY plane!")
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    origamiplots.draw_inner_creases(ori.dots, ax, 'k')

    origamiplots.draw_frame_creases(ori.dots, ax, 'r')

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def main():
    # draw_miura_cell()
    draw_miura_ori()


if __name__ == '__main__':
    main()
