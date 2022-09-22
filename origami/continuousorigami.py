import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from origami.utils import plotutils

PI = np.pi

FIGURES_PATH = '../../RFFQM/Figures/continuous-origami'


def follow_curve(xs, ys, zigzag_angle) -> (np.ndarray, np.ndarray, np.ndarray):
    ls = np.zeros((len(xs) - 1) * 2)
    middle_points_xs = np.zeros((len(xs) - 1))
    middle_points_ys = np.zeros((len(xs) - 1))

    phi = zigzag_angle
    for i in range(len(xs) - 1):
        a_x = xs[i]
        a_y = ys[i]
        b_x = xs[i + 1]
        b_y = ys[i + 1]

        dx = b_x - a_x
        dy = b_y - a_y

        c1 = dx / (2 * np.sin(phi)) - dy / (2 * np.cos(phi))
        c2 = dx / (2 * np.sin(phi)) + dy / (2 * np.cos(phi))

        if c1 < 0 or c2 < 0:
            raise RuntimeError('The given angle is not small enough to make the zigzag')

        # print(c1)

        ls[2 * i] = c1
        ls[2 * i + 1] = c2
        middle_points_xs[i] = a_x + c1 * np.sin(phi)
        middle_points_ys[i] = a_y - c1 * np.cos(phi)

    return ls, middle_points_xs, middle_points_ys


def calc_zigzag_points_by_lengths(cs, zigzag_angle, x0=0, y0=0):
    if len(cs) % 2 == 1:
        raise RuntimeError(f'length of cs must be even, got {len(cs)}')
    xs = np.zeros(len(cs) // 2 + 1)
    ys = np.zeros(len(xs))
    mid_xs = np.zeros(len(cs) // 2)
    mid_ys = np.zeros(len(mid_xs))

    xs[0] = x0
    ys[0] = y0

    for i in range(len(cs) // 2):
        c = cs[2 * i]
        mid_xs[i] = xs[i] + c * np.sin(zigzag_angle)
        mid_ys[i] = ys[i] - c * np.cos(zigzag_angle)

        c = cs[2 * i + 1]
        xs[i + 1] = mid_xs[i] + c * np.sin(zigzag_angle)
        ys[i + 1] = mid_ys[i] + c * np.cos(zigzag_angle)

    return xs, ys, mid_xs, mid_ys


def plot_zigzag(ax: Axes, xs, ys, mid_xs, mid_ys):
    points = ax.plot(xs, ys, '.')[0]
    middles = ax.plot(mid_xs, mid_ys, '.')[0]

    all_ys = np.zeros(len(xs) + len(mid_xs))
    all_zs = np.zeros(len(all_ys))

    all_ys[::2] = xs
    all_zs[::2] = ys
    all_ys[1::2] = mid_xs
    all_zs[1::2] = mid_ys

    zigzag = ax.plot(all_ys, all_zs, '-')[0]

    return points, middles, zigzag


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
