import logging
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami.utils import plotutils, linalgutils

logger = logging.getLogger('origami.dots')
logger.setLevel(logging.DEBUG)


class QuadrangleArray(object):
    def __init__(self, dots: np.ndarray, rows: int, cols: int):
        if dots.shape[0] == 2:
            dots3D = np.zeros((3, dots.shape[1]), dtype=dots.dtype)
            dots3D[:2, :] = dots
            dots = dots3D
        self.dots = dots
        self.rows, self.cols = rows, cols
        self.indexes: np.ndarray = np.arange(rows * cols).reshape((rows, cols))

    def plot(self, ax: Axes3D, alpha=1.0):
        return plot_dots(self.dots, self.indexes, ax, alpha)

    def rotate_and_center(self):
        self.dots = center_dots(self.dots, self.indexes)

    def center(self):
        self.dots -= self.dots.mean(axis=1)[:, None]

    def is_valid(self, flat_quadrangles: Optional['QuadrangleArray'] = None) -> Tuple[bool, str]:
        return is_valid(flat_quadrangles, self.dots, self.indexes)

    def assert_valid(self):
        valid, reason = self.is_valid()
        if not valid:
            raise RuntimeError(f'Not a valid quadrangles array. Reason: {reason}')
        else:
            logger.debug('Configuration is valid')

    def copy(self):
        new_dots = self.dots.copy()
        return QuadrangleArray(new_dots, self.rows, self.cols)


def plot_dots(dots: np.ndarray, indexes: np.ndarray, ax: Axes3D, alpha=1.0):
    plotutils.set_3D_labels(ax)

    rows, cols = indexes.shape
    surf = ax.plot_surface(dots[0, :].reshape((rows, cols)),
                           dots[1, :].reshape((rows, cols)),
                           dots[2, :].reshape((rows, cols)), alpha=alpha, linewidth=100)
    wire = ax.plot_wireframe(dots[0, :].reshape((rows, cols)),
                             dots[1, :].reshape((rows, cols)),
                             dots[2, :].reshape((rows, cols)),
                             alpha=alpha, color='g', linewidth=2)

    return surf, wire


def center_dots(dots: np.ndarray, indexes):
    rows, cols = indexes.shape

    v1: np.ndarray = dots[:, indexes[0, 0]] - dots[:, indexes[-1, 0]]
    v2: np.ndarray = dots[:, indexes[0, 0]] - dots[:, indexes[0, -1 - (1 - cols % 2)]]
    n1: np.ndarray = np.cross(v1, v2)
    logger.debug(f'centering dots. normal before={n1}')

    n = np.array([0, 0, 1])

    angle = linalgutils.calc_angle(n, n1)
    if np.isclose(angle, 0):
        pass
    elif np.isclose(angle, np.pi):
        R = linalgutils.create_XZ_rotation_matrix(np.pi)
        dots = R @ dots
    else:
        rot_axis = np.cross(n, n1)
        R = linalgutils.create_rotation_around_axis(rot_axis, -angle)
        dots = R @ dots

    v1: np.ndarray = dots[:, indexes[0, 0]] - dots[:, indexes[-1, 0]]
    v2: np.ndarray = dots[:, indexes[0, 0]] - dots[:, indexes[0, -1 - (1 - cols % 2)]]
    n1: np.ndarray = np.cross(v1, v2)
    logger.debug(f'normal after rotation={n1}')

    vec = dots[:, indexes[-1, 0]] - dots[:, indexes[0, 0]]

    logger.debug(f'xy vec: {vec}')

    vec[2] = 0  # cast on XY plane

    angle_xy = linalgutils.calc_angle(vec, [0, 1, 0])
    if vec[0] < 0:  # Small patch to determine the direction of the rotation
        angle_xy *= -1
    R_xy = linalgutils.create_XY_rotation_matrix(angle_xy)
    dots = R_xy @ dots

    vec = dots[:, indexes[-1, 0]] - dots[:, indexes[0, 0]]

    logger.debug(f'xy vec after rotation: {vec}')

    dots -= dots.mean(axis=1)[:, None]

    return dots


def is_valid(flat_dots, dots: np.ndarray, indexes: np.ndarray) -> Tuple[bool, str]:
    """
    check that all panels are actually quadrangle, i.e. on a single plane
    """
    rows, cols = indexes.shape
    for y in range(rows - 1):
        for x in range(cols - 1):
            v1 = dots[:, indexes[y, x + 1]] - dots[:, indexes[y, x]]
            v2 = dots[:, indexes[y + 1, x]] - dots[:, indexes[y, x]]

            n1 = np.cross(v1, v2)

            v3 = dots[:, indexes[y + 1, x + 1]] - dots[:, indexes[y, x + 1]]
            n2 = np.cross(v1, v3)

            angle = linalgutils.calc_angle(n1, n2)
            if np.isclose(angle, np.pi, atol=1e-5):
                return False, \
                    f'Panel {x},{y} has 2 opposite normals. ' \
                    f'Most likely that 2 creases intersect in the flat configuration'
            if not np.isclose(angle, 0, atol=1e-5):
                return False, \
                    f'Panel {x},{y} is not planar. Angle between 2 normals is {angle}'

    return True, 'All quadrangle are indeed planar'


def dots_to_quadrangles(dots: np.ndarray, indexes: np.ndarray) -> QuadrangleArray:
    cols, rows = indexes.shape[0], indexes.shape[1]
    return QuadrangleArray(dots, cols, rows)


def plot_flat_quadrangles(quads: QuadrangleArray) -> Tuple[Figure, Axes3D]:
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
    quads.plot(ax)
    plotutils.set_axis_scaled(ax)
    plotutils.set_3D_labels(ax)
    return fig, ax
