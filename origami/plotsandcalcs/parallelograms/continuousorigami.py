import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from origami.utils import plotutils
from origami.utils.zigzagutils import calc_zigzag_points_by_lengths, plot_zigzag

PI = np.pi

FIGURES_PATH = '../../../../RFFQM/Figures/continuous-origami'


def plot_ratio_field():
    phi = 0.6
    xs = np.linspace(0, 2 * PI, 200)
    Hs = (np.cos(xs) * np.tan(phi) + 1) / (1 - np.cos(xs) * np.tan(phi))

    fig, ax = plt.subplots()
    fig: Figure = fig
    ax: Axes = ax
    ax.plot(xs, Hs)
    plotutils.set_pi_ticks(ax, 'X', pi_range=(0, 2))
    ax.set_xlabel('x')
    ax.set_ylabel('H(x)')

    fig.savefig(os.path.join(FIGURES_PATH, 'H-vs-x.png'))


def plot_following_curve():
    phi = 0.6

    def H_func(x): return (np.cos(x) * np.tan(phi) + 1) / (1 - np.cos(x) * np.tan(phi))

    dx = 0.5
    xs = np.arange(0, 2 * PI, dx)
    Hs = H_func(xs[:-1] + dx / 2)

    cs = np.zeros((len(xs) - 1) * 2)
    cs[::2] = dx / ((1 + Hs) * np.sin(phi))
    cs[1::2] = cs[::2] * Hs
    # print(cs)

    xs, ys, mid_xs, mid_ys = calc_zigzag_points_by_lengths(cs, phi)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()

    points, middles, _ = plot_zigzag(ax, xs, ys, mid_xs, mid_ys)

    points.set_markersize(12)
    middles.set_markersize(12)

    smooth_xs = np.linspace(0, 2 * PI, 200)
    smooth_ys = np.sin(smooth_xs)

    ax.plot(smooth_xs, smooth_ys, '--')

    ax.set_xlabel('x')
    ax.set_ylabel(r'$ y=\sin(x) $')

    fig.savefig(os.path.join(FIGURES_PATH, 'following-sin-curve.png'))

    plt.show()


def main():
    # plot_ratio_field()
    plot_following_curve()


if __name__ == '__main__':
    main()
