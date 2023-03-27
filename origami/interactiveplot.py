import matplotlib
import matplotlib.widgets
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from origami.RFFQMOrigami import RFFQM
from origami.miuraori import SimpleMiuraOri
from origami.utils import plotutils


def add_slider_miuraori(ax, ori: SimpleMiuraOri, should_plot_normals=False):
    lim = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

    init_omega = 1

    # Make a horizontal slider to control the frequency.
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


def add_slider(ax, origami: RFFQM):
    init_omega = 0.5
    dots = origami.set_gamma(init_omega)
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
        quads = origami.set_gamma(omega, should_center=True)
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
