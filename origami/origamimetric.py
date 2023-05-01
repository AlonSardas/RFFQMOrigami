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

-------------------------------------

We also implement a discrete way to calculate the shape operator using the method written
in
Grinspun, E., Gingold, Y., Reisman, J. and Zorin, D. (2006),
Computing discrete shape operators on general meshes.
Computer Graphics Forum, 25: 547-556.
https://doi.org/10.1111/j.1467-8659.2006.00974.x

and also, see the notation in
Xiangxin Dang, Fan Feng, Paul Plucinsky, Richard D. James, Huiling Duan, Jianxiang Wang,
Inverse design of deployable origami structures that approximate a general surface,
International Journal of Solids and Structures,
Volumes 234–235,
2022,
111224,
ISSN 0020-7683,
https://doi.org/10.1016/j.ijsolstr.2021.111224.

And another similar one:
van Rees, Wim M. and Matsumoto, Elisabetta A. and Gladman, A. Sydney and Lewis, Jennifer A. and Mahadevan, L.
Mechanics of biomimetic 4D printed structures
"""
from typing import Tuple

import numpy as np

from origami.quadranglearray import QuadrangleArray
from origami.utils import linalgutils


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


def calc_curvature_by_triangles(quads: QuadrangleArray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a discrete version of the shape operator and using it the
    Gaussian curvature (its determinant). See figure:
    https://pubs.rsc.org/image/article/2018/sm/c8sm00990b/c8sm00990b-f1_hi-res.gif

    There is more than one way to define the triangles based on the origami,
    the method used here is based on what defined in the article:
    `Inverse design of deployable origami structures that approximate a general surface`
    Each unit cell of 2X2 panels is translated into 4 triangles. Then the shape operator
    is calculated based on these triangles.

    :param quads: The dots of the quadrilateral mesh origami
    :return: K, b11, b12, b22
    """
    dots, indexes = quads.dots, quads.indexes
    triangles_rows = ((quads.rows - 1) // 2) * 2
    triangles_cols = ((quads.cols - 1) // 2) * 2
    tri_indexes = np.arange(triangles_rows * triangles_cols).reshape((triangles_rows, triangles_cols))
    triangles_normals = np.zeros((3, triangles_rows * triangles_cols))

    for i in range(0, triangles_rows, 2):
        for j in range(0, triangles_cols, 2):
            index0 = indexes[i + 2, j + 1]

            index1 = indexes[i, j]
            index2 = indexes[i, j + 1]
            triangles_normals[:, tri_indexes[i, j]] = \
                _calc_normal_for_triangle(quads, index0, index1, index2)

            index1 = indexes[i, j + 1]
            index2 = indexes[i, j + 2]
            triangles_normals[:, tri_indexes[i, j + 1]] = \
                _calc_normal_for_triangle(quads, index0, index1, index2)

            index1 = indexes[i, j + 2]
            index2 = indexes[i + 2, j + 2]
            triangles_normals[:, tri_indexes[i + 1, j + 1]] = \
                _calc_normal_for_triangle(quads, index0, index1, index2)

            index1 = indexes[i + 2, j]
            index2 = indexes[i, j]
            triangles_normals[:, tri_indexes[i + 1, j]] = \
                _calc_normal_for_triangle(quads, index0, index1, index2)
    interior_triangle_matrix_shape = (triangles_rows - 2, triangles_cols - 2)
    b11 = np.zeros(interior_triangle_matrix_shape)
    b12 = np.zeros(interior_triangle_matrix_shape)
    b22 = np.zeros(interior_triangle_matrix_shape)
    Ks = np.zeros(interior_triangle_matrix_shape)
    # For each interior triangle
    for i in range(1, triangles_rows - 1):
        for j in range(1, triangles_cols - 1):
            if i % 2 == 0:
                if j % 2 == 0:
                    index0 = indexes[i, j]
                    index1 = indexes[i, j + 1]
                    index2 = indexes[i + 2, j + 1]
                else:
                    index0 = indexes[i, j]
                    index1 = indexes[i, j + 1]
                    index2 = indexes[i + 2, j]
            else:
                if j % 2 == 0:
                    index0 = indexes[i - 1, j]
                    index1 = indexes[i + 1, j + 1]
                    index2 = indexes[i + 1, j]
                else:
                    index0 = indexes[i + 1, j]
                    index1 = indexes[i - 1, j + 1]
                    index2 = indexes[i + 1, j + 1]
            v0, v1, v2 = dots[:, index0], dots[:, index1], dots[:, index2]
            # e0: np.ndarray = (v1 - v0) / np.linalg.norm(v1 - v0)
            # e1: np.ndarray = (v2 - v1) / np.linalg.norm(v2 - v1)
            # e2: np.ndarray = (v2 - v0) / np.linalg.norm(v2 - v0)
            e0: np.ndarray = v1 - v0
            e1: np.ndarray = v2 - v1
            e2: np.ndarray = v2 - v0
            e0n = e0 / np.linalg.norm(e0)
            e1n = e1 / np.linalg.norm(e1)
            e2n = e2 / np.linalg.norm(e2)

            nt = triangles_normals[:, tri_indexes[i, j]]

            N0I = np.cross(e0n, nt)
            N1I = np.cross(e1n, nt)
            N2I = np.cross(e2n, nt)

            if i % 2 == 0:
                if j % 2 == 0:
                    tri_index0 = tri_indexes[i - 1, j]
                    tri_index1 = tri_indexes[i, j + 1]
                    tri_index2 = tri_indexes[i + 1, j]
                else:
                    tri_index0 = tri_indexes[i - 1, j]
                    tri_index1 = tri_indexes[i + 1, j]
                    tri_index2 = tri_indexes[i, j - 1]
            else:
                if j % 2 == 0:
                    tri_index0 = tri_indexes[i - 1, j]
                    tri_index1 = tri_indexes[i + 1, j]
                    tri_index2 = tri_indexes[i, j - 1]
                else:
                    tri_index0 = tri_indexes[i - 1, j]
                    tri_index1 = tri_indexes[i, j + 1]
                    tri_index2 = tri_indexes[i + 1, j]

            N0O = -np.cross(e0n, triangles_normals[:, tri_index0])
            N1O = -np.cross(e1n, triangles_normals[:, tri_index1])
            N2O = -np.cross(e2n, triangles_normals[:, tri_index2])

            n0 = (N0I + N0O) / np.linalg.norm(N0I + N0O)
            n1 = (N1I + N1O) / np.linalg.norm(N1I + N1O)
            n2 = (N2I + N2O) / np.linalg.norm(N2I + N2O)

            should_assert = False
            if should_assert:
                print(i, j)
                assert np.all(np.isclose(nt, np.cross(e0, e1) / np.linalg.norm(np.cross(e0, e1)))), \
                    "The normal do not coincides"
                assert np.isclose(np.linalg.norm(N0I), 1), "The normal is not perpendicular"
                assert np.isclose(np.linalg.norm(N1I), 1), "The normal is not perpendicular"
                assert np.isclose(np.linalg.norm(N2I), 1), "The normal is not perpendicular"
                assert np.isclose(np.linalg.norm(N0O), 1), "The normal is not perpendicular"
                assert np.isclose(np.linalg.norm(N1O), 1), "The normal is not perpendicular"
                assert np.isclose(np.linalg.norm(N2O), 1), "The normal is not perpendicular"
                assert np.isclose(n0.dot(e0), 0), "The normal is not perpendicular"
                assert np.isclose(n1.dot(e1), 0), "The normal is not perpendicular"
                assert np.isclose(n2.dot(e2), 0), "The normal is not perpendicular"

            # print(N2I, N2O)

            tb11 = 2 * (n0 - n2).dot(e1)
            tb22 = 2 * (n1 - n0).dot(e2)
            tb12 = 2 * (n1 - n0).dot(e1)

            g11 = e1.dot(e1)
            g22 = e2.dot(e2)
            g12 = e1.dot(e2)

            b11[i - 1, j - 1] = tb11
            b22[i - 1, j - 1] = tb22
            b12[i - 1, j - 1] = tb12
            Ks[i - 1, j - 1] = (tb11 * tb22 - tb12 ** 2) / (g11 * g22 - g12 ** 2)

    print(b12)
    return Ks, b11, b12, b22


def _calc_normal_for_triangle(quads, index0, index1, index2) -> np.ndarray:
    dots, indexes = quads.dots, quads.indexes
    v0 = dots[:, index0]
    v1 = dots[:, index1]
    v2 = dots[:, index2]
    return linalgutils.calc_normal(v1 - v0, v2 - v0)
