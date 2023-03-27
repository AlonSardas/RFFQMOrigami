"""
This module takes QuadrangleArray and calculate its metric in the 3D space.
We use a unit cell of 2X2 quadrangles to calculate dx,dy. The metric is then
g11 = du^2
g22 = dv^2
g12 = g21 = du * dv
We assume coordinates that go along the QuadrangleArray, that is, a step to the right
corresponds to increase of 1 in the u coordinate.

Once we have the metric (in a discrete manner), we can also calculate
the Gaussian curvature
"""
from typing import Tuple

import numpy as np

from origami.quadranglearray import QuadrangleArray


def calc_curvature_and_metric(quads: QuadrangleArray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Gaussian curvature and the metric entries for each unit cell.
    For the formulas used here, see:
    https://en.wikipedia.org/wiki/Gaussian_curvature
    https://en.wikipedia.org/wiki/Second_fundamental_form

    :param quads: The dots of the quadrilateral mesh origami
    :return: K, g11, g12, g22
    """
    indexes = quads.indexes
    rows = (quads.rows + 1) // 2
    cols = (quads.cols + 1) // 2
    metric_dot_xs: np.ndarray = quads.dots[0, indexes[::2, ::2]].reshape(rows, cols)
    metric_dot_ys: np.ndarray = quads.dots[1, indexes[::2, ::2]].reshape(rows, cols)
    metric_dot_zs: np.ndarray = quads.dots[2, indexes[::2, ::2]].reshape(rows, cols)

    du_xs = np.diff(metric_dot_xs, axis=1)[:-1, :]
    du_ys = np.diff(metric_dot_ys, axis=1)[:-1, :]
    du_zs = np.diff(metric_dot_zs, axis=1)[:-1, :]

    dv_xs = np.diff(metric_dot_xs, axis=0)[:, :-1]
    dv_ys = np.diff(metric_dot_ys, axis=0)[:, :-1]
    dv_zs = np.diff(metric_dot_zs, axis=0)[:, :-1]

    # We use the second fundamental for to calculate the Gaussian curvature
    # it seems easier than working with Christoffel symbols
    dudu_xs = np.diff(du_xs, axis=1)[:-1, :]
    dudu_ys = np.diff(du_ys, axis=1)[:-1, :]
    dudu_zs = np.diff(du_zs, axis=1)[:-1, :]

    dvdv_xs = np.diff(dv_xs, axis=0)[:, :-1]
    dvdv_ys = np.diff(dv_ys, axis=0)[:, :-1]
    dvdv_zs = np.diff(dv_zs, axis=0)[:, :-1]

    dvdu_xs = np.diff(du_xs, axis=0)[:, :-1]
    dvdu_ys = np.diff(du_ys, axis=0)[:, :-1]
    dvdu_zs = np.diff(du_zs, axis=0)[:, :-1]

    # To match the shape:
    du_xs, du_ys, du_zs, dv_xs, dv_ys, dv_zs = [
        a[:-1, :-1] for a in [du_xs, du_ys, du_zs, dv_xs, dv_ys, dv_zs]]

    g11 = du_xs ** 2 + du_ys ** 2 + du_zs ** 2
    g22 = dv_xs ** 2 + dv_ys ** 2 + dv_zs ** 2
    g12 = du_xs * dv_xs + du_ys * dv_ys + du_zs * dv_zs

    N_xs = du_ys * dv_zs - dv_ys * du_zs
    N_ys = dv_xs * du_zs - du_xs * dv_zs
    N_zs = du_xs * dv_ys - dv_xs * du_ys

    N_norms = np.sqrt(N_xs ** 2 + N_ys ** 2 + N_zs ** 2)
    N_xs /= N_norms
    N_ys /= N_norms
    N_zs /= N_norms

    b11 = dudu_xs * N_xs + dudu_ys * N_ys + dudu_zs * N_zs
    b22 = dvdv_xs * N_xs + dvdv_ys * N_ys + dvdv_zs * N_zs
    b12 = dvdu_xs * N_xs + dvdu_ys * N_ys + dvdu_zs * N_zs

    Ks = (b11 * b22 - b12 ** 2) / (g11 * g22 - g12 ** 2)

    return Ks, g11, g12, g22
