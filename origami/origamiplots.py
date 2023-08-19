import matplotlib
import matplotlib.widgets
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami.RFFQMOrigami import RFFQM
from origami.miuraori import SimpleMiuraOri
from origami.utils import plotutils


def add_slider_miuraori(ax, ori: SimpleMiuraOri, should_plot_normals=False):
    lim = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

    init_omega = 1

    # Make a horizontal slider
    omega_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
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
        quads = ori.set_gamma(omega, should_center=True)
        # dots.center()
        quads.plot(ax, alpha=0.85)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    update_omega(init_omega)
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


def plot_crease_pattern(ori: RFFQM):
    if not np.isclose(ori.gamma, 0):
        raise ValueError(f"The given origami is not flat. gamma={ori.gamma}")
    if not np.all(np.isclose(ori.dots.dots[2, :], 0)):
        raise ValueError(f"The given origami is not on XY plane!")
    fig: Figure = plt.figure()
    # ax: Axes3D = fig.add_subplot(111, projection='3d', elev=90, azim=-90)
    ax: Axes = fig.add_subplot(111)
    indexes = ori.dots.indexes
    dots = ori.dots.dots

    rows, cols = indexes.shape

    MVA_to_color = {1: 'r', -1: 'b'}
    MVA = 1

    def draw_crease(dot, i, j, _MVA):
        ax.plot([dot[0], dots[0, indexes[i, j]]],
                [dot[1], dots[1, indexes[i, j]]],
                # [dot[2], dots[2, indexes[i, j]]],
                color=MVA_to_color[_MVA])

    for j in range(1, cols):
        dot = dots[:, indexes[0, j]]
        draw_crease(dot, 0, j - 1, -MVA)

    for i in range(1, rows):
        dot = dots[:, indexes[i, 0]]
        draw_crease(dot, i - 1, 0, MVA)
        MVA *= -1

    for i in range(1, rows):
        for j in range(1, cols):
            vertical_MVA = 1 if j % 2 == 0 else -1

            dot_index = indexes[i, j]
            dot = dots[:, dot_index]
            draw_crease(dot, i, j - 1, MVA)
            draw_crease(dot, i - 1, j, vertical_MVA * MVA)
        MVA *= -1

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax
