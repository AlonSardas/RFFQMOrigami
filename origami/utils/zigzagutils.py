from typing import Tuple

import numpy as np
from matplotlib.axes import Axes


def follow_curve(xs: np.ndarray, ys: np.ndarray, zigzag_angle: float) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def calc_zigzag_points_by_lengths(cs, zigzag_angle: float, x0: float = 0, y0: float = 0):
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
