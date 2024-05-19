import logging
from typing import Optional, Tuple

import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from origami.utils import plotutils, linalgutils

logger = logging.getLogger("origami.dots")


class QuadrangleArray(object):
    def __init__(self, dots: np.ndarray, rows: int, cols: int):
        """
        Store the data of N=rows*columns dots, and their x,y,z positions in 3D space

        :param dots: 2D of shape 3XN with the coordinates for each dot.
            If the given data is 2XN, the last z coordinate is filled with 0.
        :param rows: the number of rows in the quadrangle array
        :param cols: the number of columns in the quadrangle array
        """
        if len(dots.shape) != 2:
            raise ValueError("The array of dots should be 2 dimensional, with rows as the number of dots")

        if dots.shape[1] != rows * cols:
            raise ValueError(f"There should be {rows}x{cols} dots, got {dots.shape[1]}")

        if dots.shape[0] == 2:
            dots3D = np.zeros((3, dots.shape[1]), dtype=dots.dtype)
            dots3D[:2, :] = dots
            dots = dots3D
        self.dots = dots
        self.rows, self.cols = rows, cols
        self.indexes: np.ndarray = np.arange(rows * cols).reshape((rows, cols))

    def plot_with_wireframe(self, ax: Axes3D, color=None, alpha=1.0, lightsource=None):
        return plot_panels_and_edges_with_wireframe(self.dots, self.indexes, ax, color, alpha, lightsource)

    def plot(self, ax: Axes3D, panel_color=None, edge_color='g',
             alpha=1.0, edge_alpha=1.0, edge_width=1.0, lightsource=None) -> Poly3DCollection:
        return plot_panels(self.dots, self.indexes, ax, panel_color,
                           edge_color, alpha, edge_alpha, edge_width, lightsource)

    def rotate_and_center(self):
        self.dots = center_dots(self.dots, self.indexes)

    def center(self):
        self.dots -= self.dots.mean(axis=1)[:, None]

    def is_valid(
            self, flat_quadrangles: Optional["QuadrangleArray"] = None
    ) -> Tuple[bool, str]:
        return is_valid(flat_quadrangles, self.dots, self.indexes)

    def assert_valid(self):
        valid, reason = self.is_valid()
        if not valid:
            raise RuntimeError(f"Not a valid quadrangles array. Reason: {reason}")
        else:
            logger.debug("Configuration is valid")

    def copy(self):
        new_dots = self.dots.copy()
        return QuadrangleArray(new_dots, self.rows, self.cols)


def from_mesh_gird(Xs: np.ndarray, Ys: np.ndarray, Zs: np.ndarray):
    if Xs.shape != Ys.shape:
        raise ValueError("Data size must be the same")
    if Xs.shape != Zs.shape:
        raise ValueError("Data size must be the same")
    rows, cols = Xs.shape
    dots = np.array((Xs.flat, Ys.flat, Zs.flat))
    return QuadrangleArray(dots, rows, cols)


def plot_panels(dots: np.ndarray, indexes: np.ndarray,
                ax: Axes3D, panels_color=None, edge_color=None,
                alpha=1.0, edge_alpha=1.0, edge_width=1.5,
                lightsource=None) -> Poly3DCollection:
    # It seems that plotting with float128 is not supported by matplotlib
    if hasattr(np, 'float128') and dots.dtype == np.float128:
        dots = np.array(dots, np.float64)

    rows, cols = indexes.shape
    surf = ax.plot_surface(
        dots[0, :].reshape((rows, cols)),
        dots[1, :].reshape((rows, cols)),
        dots[2, :].reshape((rows, cols)),
        alpha=0.4,
        linewidth=edge_width,
        # antialiased=True,
        color=panels_color,
        edgecolor=edge_color,
        lightsource=lightsource,
        rstride=1,
        cstride=1
    )

    # This is a patch to make the panels and edges have different alphas
    surf.set_alpha(edge_alpha)
    surf.set_edgecolor(surf.get_edgecolor())
    surf.set_alpha(alpha)

    return surf


def plot_panels_manual_zorder(quads: QuadrangleArray,
                              ax: Axes3D, panel_color='C0', edge_color='k',
                              alpha=1.0, edge_alpha=1.0, edge_width=1.5,
                              lightsource=None,
                              z0=0.0, z_shift=0.01,
                              reverse_x_dir=False):
    # This function is a last attempt to overcome the wrong zordering
    # of matplotlib 3d plots.
    # It is still a bad solution, since it will work only for limited view point.
    # In hindsight, it might be better to migrate the plots to plotly
    # (or mayavi, but it seemed less convenient).
    if ax.computed_zorder:
        raise RuntimeError("The method for manually setting the zorder works only when "
                           "computed_zorder is set to False")

    dots, indexes = quads.dots, quads.indexes
    if hasattr(np, 'float128') and dots.dtype == np.float128:
        dots = np.array(dots, np.float64)

    rows, cols = indexes.shape
    xs = dots[0, :].reshape((rows, cols))
    ys = dots[1, :].reshape((rows, cols))
    zs = dots[2, :].reshape((rows, cols))

    z = z0
    panels = []
    for j in range(0, cols):
        if reverse_x_dir:
            j = cols - 1 - j
        for i in range(0, rows, 1):
            if j % 2 == 0:
                i = rows - 1 - i
            # print(i, 2 * i, 2 * i + 2)
            surf = ax.plot_surface(
                xs[i:i + 2, j:j + 2],
                ys[i:i + 2, j:j + 2],
                zs[i:i + 2, j:j + 2],
                alpha=0.4,
                linewidth=edge_width,
                # antialiased=True,
                color=panel_color,
                edgecolor=edge_color,
                lightsource=lightsource,
                rstride=1,
                cstride=1,
                zorder=z
            )
            z -= z_shift

            # This is a patch to make the panels and edges have different alphas
            surf.set_alpha(edge_alpha)
            surf.set_edgecolor(surf.get_edgecolor())
            surf.set_alpha(alpha)
            panels.append(surf)

    return panels


def plot_panels_and_edges_with_wireframe(dots: np.ndarray, indexes: np.ndarray, ax: Axes3D, color=None, alpha=1.0,
                                         lightsource=None):
    # This function is not recommended and it remains for legacy.
    # Plotting the wireframe separately makes the edges that are supposed to be hidden
    # visible on top of the panels
    plotutils.set_3D_labels(ax)

    # It seems that plotting with float128 is not supported by matplotlib
    if dots.dtype == np.float128:
        dots = np.array(dots, np.float64)

    rows, cols = indexes.shape
    surf = ax.plot_surface(
        dots[0, :].reshape((rows, cols)),
        dots[1, :].reshape((rows, cols)),
        dots[2, :].reshape((rows, cols)),
        alpha=alpha,
        linewidth=1, color=color, lightsource=lightsource,
        # antialiased=False
    )
    wire = ax.plot_wireframe(
        dots[0, :].reshape((rows, cols)),
        dots[1, :].reshape((rows, cols)),
        dots[2, :].reshape((rows, cols)),
        alpha=alpha,
        color="g",
        linewidth=2,
    )

    return surf, wire
    # return surf, None


def plot_2D_polygon(quads: QuadrangleArray, ax: Axes, color='0.9') -> mpatches.Polygon:
    dots, indexes = quads.dots, quads.indexes
    poly_xyz = np.concatenate([dots[:, indexes[0, :]],
                               dots[:, indexes[1:, -1]],
                               dots[:, indexes[-1, -2::-1]],
                               dots[:, indexes[-2:0:-1, 0]]], axis=1)
    poly_xy = poly_xyz[:2, :]
    poly = mpatches.Polygon(poly_xy.transpose(), color=color, lw=0)
    ax.add_patch(poly)
    return poly


def center_dots(dots: np.ndarray, indexes):
    rows, cols = indexes.shape

    v_y: np.ndarray = dots[:, indexes[-1, 0]] - dots[:, indexes[0, 0]]
    v_x: np.ndarray = dots[:, indexes[0, -1 - (1 - cols % 2)]] - dots[:, indexes[0, 0]]
    n1: np.ndarray = np.cross(v_x, v_y)
    logger.debug(f"centering dots. normal before={n1}")

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

    v_y: np.ndarray = dots[:, indexes[-1, 0]] - dots[:, indexes[0, 0]]
    v_x: np.ndarray = dots[:, indexes[0, -1 - (1 - cols % 2)]] - dots[:, indexes[0, 0]]
    n1: np.ndarray = np.cross(v_x, v_y)
    logger.debug(f"normal after rotation={n1}")

    vec = dots[:, indexes[-1, 0]] - dots[:, indexes[0, 0]]

    logger.debug(f"xy vec: {vec}")

    vec[2] = 0  # cast on XY plane

    angle_xy = linalgutils.calc_angle(vec, [0, 1, 0])
    if vec[0] < 0:  # Small patch to determine the direction of the rotation
        angle_xy *= -1
    R_xy = linalgutils.create_XY_rotation_matrix(angle_xy)
    dots = R_xy @ dots

    vec = dots[:, indexes[-1, 0]] - dots[:, indexes[0, 0]]

    logger.debug(f"xy vec after rotation: {vec}")

    dots -= dots.mean(axis=1)[:, None]

    return dots


def is_valid(flat_dots, dots: np.ndarray, indexes: np.ndarray) -> Tuple[bool, str]:
    """
    check that all panels are actually quadrangles, i.e. all 4 vertices are on a single plane
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
                return (
                    False,
                    f"Panel {x},{y} has 2 opposite normals. "
                    f"Most likely that 2 creases intersect in the flat configuration",
                )
            if not np.isclose(angle, 0, atol=1e-5):
                return (
                    False,
                    f"Panel {x},{y} is not planar. Angle between 2 normals is {angle}",
                )

    return True, "All quadrangle are indeed planar"


def dots_to_quadrangles(dots: np.ndarray, indexes: np.ndarray) -> QuadrangleArray:
    cols, rows = indexes.shape[0], indexes.shape[1]
    return QuadrangleArray(dots, cols, rows)


def plot_flat_quadrangles(quads: QuadrangleArray) -> Tuple[Figure, Axes3D]:
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-90, elev=90)
    quads.plot_with_wireframe(ax)
    plotutils.set_axis_scaled(ax)
    return fig, ax


def _triangulate(quads: QuadrangleArray):
    rows, cols = quads.rows, quads.cols
    dots, indexes = quads.dots, quads.indexes
    panels_x, panels_y = cols - 1, rows - 1

    triangles = np.zeros((2 * panels_x * panels_y, 3))

    ind = 0
    for i in range(panels_y):
        for j in range(panels_y):
            triangles[ind] = [indexes[i, j], indexes[i, j + 1], indexes[i + 1, j]]
            ind += 1
            triangles[ind] = [indexes[i, j + 1], indexes[i + 1, j + 1], indexes[i + 1, j]]
            ind += 1

    return triangles
