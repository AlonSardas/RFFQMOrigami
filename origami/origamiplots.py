from typing import Sequence, Tuple

import matplotlib
from matplotlib.lines import Line2D
import matplotlib.widgets
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

from origami.miuraori import SimpleMiuraOri
from origami.quadranglearray import QuadrangleArray
from origami.RFFQMOrigami import RFFQM
from origami.utils import plotutils


def add_slider_miuraori(ax, ori: SimpleMiuraOri, should_plot_normals=False):
    lim = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

    init_omega = 1

    # Make a horizontal slider
    omega_slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    omega_slider = matplotlib.widgets.Slider(
        ax=omega_slider_ax,
        label='Omega',
        valmin=0,
        valmax=np.pi,
        valinit=init_omega,
    )

    ori.plot(ax)

    def update_omega(omega):
        ax.clear()
        ori.set_omega(omega)
        if should_plot_normals:
            ori.plot_normals(ax)
            ori.plot(ax, alpha=0.3)
        else:
            ori.plot(ax, alpha=1)
        # ax.set_autoscale_on(False)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim / 2, lim / 2)

        plotutils.set_3D_labels(ax)

    omega_slider.on_changed(update_omega)
    # update_omega(np.pi/2)
    update_omega(init_omega)
    return omega_slider


def plot_interactive_miuraori(ori: SimpleMiuraOri):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ori.plot(ax)

    # We need to assign the return value to variable slider for the slider object
    # stay alive and keep functioning
    # noinspection PyUnusedLocal
    slider = add_slider_miuraori(ax, ori, should_plot_normals=False)

    plt.show()


def add_slider(ax, ori: RFFQM):
    # dots.r
    # if np.isclose(origami.gamma, 0):
    #     init_omega = 0.5
    #     dots = origami.set_gamma(init_omega)
    dots = ori.dots
    dots.rotate_and_center()
    init_omega = ori.gamma

    # dots.assert_valid()
    dots.plot(ax)  # We plot to find the limits of the axis to use

    lim = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

    # Make a horizontal slider
    omega_slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    omega_slider = matplotlib.widgets.Slider(
        ax=omega_slider_ax,
        label='Omega',
        valmin=0,
        valmax=np.pi,
        valinit=init_omega,
    )

    def update_omega(omega):
        ax.clear()
        quads = ori.set_gamma(omega, should_center=True)
        # dots.center()
        quads.plot(ax, alpha=0.85)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    update_omega(init_omega)
    omega_slider.on_changed(update_omega)
    return omega_slider


def plot_interactive(ori: RFFQM):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    # We need to assign the return value to variable slider for the slider object
    # stay alive and keep functioning
    # noinspection PyUnusedLocal
    slider = add_slider(ax, ori)

    plt.show()


def plot_crease_pattern(ori: RFFQM, initial_MVA=1) -> Tuple[Figure, Axes]:
    """
    Plot the crease lines on a flat 2D axes.

    Args:
        ori (RFFQM): The origami the plot
        initial_MVA (int, optional): The first mountain-valley assignment (MVA) to use
            1,-1 flip between them. 0 to ignore the MVA

    Returns:
        Tuple[Figure, Axes]: The figure and axes created
    """
    if not np.isclose(ori.gamma, 0):
        raise ValueError(f"The given origami is not flat. gamma={ori.gamma}")
    if not np.all(np.isclose(ori.dots.dots[2, :], 0)):
        raise ValueError(f"The given origami is not on XY plane!")
    fig: Figure = plt.figure()
    # ax: Axes3D = fig.add_subplot(111, projection='3d', elev=90, azim=-90)
    ax: Axes = fig.add_subplot(111)

    draw_creases(ori, initial_MVA, ax)

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def draw_creases(ori: RFFQM, initial_MVA, ax: Axes | Axes3D):
    indexes = ori.dots.indexes
    dots = ori.dots.dots

    rows, cols = indexes.shape

    MVA_to_color = {1: 'r', -1: 'b', 0: 'k'}

    def draw_crease(dot, i, j, _MVA):
        if isinstance(ax, Axes3D):
            ax.plot([dot[0], dots[0, indexes[i, j]]],
                    [dot[1], dots[1, indexes[i, j]]],
                    [dot[2], dots[2, indexes[i, j]]],
                    color=MVA_to_color[_MVA])
        else:
            ax.plot([dot[0], dots[0, indexes[i, j]]],
                    [dot[1], dots[1, indexes[i, j]]],
                    color=MVA_to_color[_MVA])

    MVA = initial_MVA
    for j in range(1, cols):
        dot = dots[:, indexes[0, j]]
        draw_crease(dot, 0, j - 1, -MVA)

    MVA = initial_MVA
    for i in range(1, rows):
        dot = dots[:, indexes[i, 0]]
        draw_crease(dot, i - 1, 0, MVA)
        MVA *= -1

    MVA = initial_MVA
    for i in range(1, rows):
        for j in range(1, cols):
            vertical_MVA = 1 if j % 2 == 0 else -1

            dot_index = indexes[i, j]
            dot = dots[:, dot_index]
            draw_crease(dot, i, j - 1, MVA)
            draw_crease(dot, i - 1, j, vertical_MVA * MVA)
        MVA *= -1


def draw_inner_creases_no_MVA(quads: QuadrangleArray, ax: Axes, color):
    creases = draw_inner_creases(quads, 0)
    for crease in creases:
        crease.set_color(color)
    return creases


def draw_inner_creases(quads, ax: Axes, initial_MVA=-1) -> Sequence[Line2D]:
    creases = []

    indexes = quads.indexes
    dots = quads.dots
    rows, cols = indexes.shape

    MVA_to_color = {1: 'r', -1: 'b', 0: 'k'}

    def draw_crease(dot, i, j, MVA):
        color = MVA_to_color[MVA]
        crease = ax.plot([dot[0], dots[0, indexes[i, j]]],
                         [dot[1], dots[1, indexes[i, j]]],
                         color)[0]
        creases.append(crease)

    MVA = initial_MVA
    for j in range(1, cols - 1):
        dot = dots[:, indexes[0, j]]
        draw_crease(dot, 1, j, MVA)
        MVA *= -1

    MVA = -initial_MVA
    for i in range(1, rows - 1):
        dot = dots[:, indexes[i, 0]]
        draw_crease(dot, i, 1, MVA)
        MVA *= -1

    MVA = -initial_MVA
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            vertical_MVA = -1 if j % 2 == 0 else 1
            dot_index = indexes[i, j]
            dot = dots[:, dot_index]
            draw_crease(dot, i, j + 1, MVA)
            draw_crease(dot, i + 1, j, vertical_MVA * MVA)
        MVA *= -1

    return creases


def draw_frame_creases(quads: QuadrangleArray, ax: Axes, color):
    dots, indexes = quads.dots, quads.indexes
    rows, cols = quads.rows, quads.cols

    if not np.all(np.isclose(dots[2, :], 0)):
        raise ValueError(f"The given origami is not on XY plane!")

    def draw_crease(dot, i, j, c):
        ax.plot([dot[0], dots[0, indexes[i, j]]],
                [dot[1], dots[1, indexes[i, j]]],
                color=c)

    for j in range(1, cols):
        dot = dots[:, indexes[0, j]]
        draw_crease(dot, 0, j - 1, color)

    for j in range(1, cols):
        dot = dots[:, indexes[-1, j]]
        draw_crease(dot, -1, j - 1, color)

    for i in range(1, rows):
        dot = dots[:, indexes[i, 0]]
        draw_crease(dot, i - 1, 0, color)

    for i in range(1, rows):
        dot = dots[:, indexes[i, -1]]
        draw_crease(dot, i - 1, -1, color)


def plot_smooth_interpolation(quads: QuadrangleArray, ax: Axes3D):
    dots, indexes = quads.dots, quads.indexes
    dots = dots[:, indexes[::2, ::2]]
    left, right = np.min(dots[0, :, :]), np.max(dots[0, :, :])
    bottom, top = np.min(dots[1, :, :]), np.max(dots[1, :, :])

    interp = interpolate.SmoothBivariateSpline(
        dots[0, :, :].flatten(),
        dots[1, :, :].flatten(),
        dots[2, :, :].flatten())

    xs = np.linspace(left, right, 30)
    ys = np.linspace(bottom, top, 30)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = interp(Xs, Ys, grid=False)

    return ax.plot_surface(Xs, Ys, Zs, linewidth=0, antialiased=False)
