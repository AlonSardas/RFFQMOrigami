import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami import quadranglearray, origamimetric
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import create_angles_func_vertical_alternation, set_perturbations_by_func
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles
from origami.utils import plotutils
from origami.utils.plotutils import imshow_with_colorbar


def test_shape_operator():
    rows, cols = 21, 31
    x = np.linspace(-1 / 2, 1 / 2, cols)
    y = np.linspace(-1 / 2, 1 / 2, rows)
    xs, ys = np.meshgrid(x, y)
    ys[:, 1::2] += 0.01
    zs = np.sqrt(1 ** 2 - xs ** 2 - ys ** 2)

    dots = np.zeros((3, rows * cols))
    dots[0, :] = xs.flat
    dots[1, :] = ys.flat
    dots[2, :] = zs.flat

    quads = quadranglearray.QuadrangleArray(dots, rows, cols)

    quadranglearray.plot_flat_quadrangles(quads)

    Ks, _, _, _ = origamimetric.calc_curvature_and_metric(quads)
    Ks_by_shape, _, _, _ = origamimetric.calc_curvature_by_triangles(quads)
    Ks_by_christoffel, _, _, _ = origamimetric.calc_curvature_by_christoffel(quads)

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks')
    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks_by_shape, 'Ks by triangles')
    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks_by_christoffel, 'Ks by christoffel')

    plt.show()


def test_metric_calc():
    # rows = 40
    # cols = 40
    rows = 40
    cols = 40

    F = lambda x: 0.001 * (x - cols / 2 + 0.05)
    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

    MM = lambda y: 0.005 * ((y - rows / 4)) ** 2
    dMM = lambda y: MM(y + 1) - MM(y)
    ddMM = lambda y: dMM(y + 1) - dMM(y)
    dddMM = lambda y: ddMM(y + 1) - ddMM(y)

    FF = lambda x: F(x * 2)
    dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)
    ddFF = lambda x: dFF(x + 0.5) - dFF(x - 0.5)
    dddFF = lambda x: ddFF(x + 0.5) - ddFF(x - 0.5)

    angle = np.pi / 2 - 0.1

    L0 = 1
    C0 = 0.5
    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0

    ys = np.arange(len(ls) // 2)
    ls[1::2] = L0 + dMM(ys) + 1 / 2 * ddMM(ys)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    ori = RFFQM(quads)

    W0 = 3.0
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)
    # KsOther, _, _, _ = origamimetric.calc_curvature_by_triangles(ori.dots)

    fig, axes = plt.subplots(2, 2)
    imshow_with_colorbar(fig, axes[0, 0], g11, "g11")
    imshow_with_colorbar(fig, axes[0, 1], g22, "g22")
    imshow_with_colorbar(fig, axes[1, 0], g12, "g12")
    imshow_with_colorbar(fig, axes[1, 1], Ks, "K")

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
    quads = ori.dots
    dots = quads.dots
    indexes = quads.indexes
    # quads.plot(ax, alpha=0.1)
    plotutils.set_axis_scaled(ax)
    dots64 = dots.astype('float64')

    rows = (quads.rows + 1) // 2
    cols = (quads.cols + 1) // 2
    metric_dot_xs: np.ndarray = dots64[0, indexes[::2, ::2]].reshape(rows, cols)
    metric_dot_ys: np.ndarray = dots64[1, indexes[::2, ::2]].reshape(rows, cols)
    metric_dot_zs: np.ndarray = dots64[2, indexes[::2, ::2]].reshape(rows, cols)

    ax.scatter(metric_dot_xs,
               metric_dot_ys,
               metric_dot_zs, alpha=1)
    # ax.scatter(dots64[0, :],
    #            dots64[1, :],
    #            dots64[2, :], alpha=0.1)
    du_xs, du_ys, du_zs, dv_xs, dv_ys, dv_zs = \
        origamimetric._calc_du_dv(metric_dot_xs, metric_dot_ys, metric_dot_zs)

    # We use the second fundamental for to calculate the Gaussian curvature
    # it seems easier than working with Christoffel symbols
    (dudu_xs, dudu_ys, dudu_zs, dvdu_xs, dvdu_ys, dvdu_zs, dvdv_xs, dvdv_ys, dvdv_zs) = \
        origamimetric._calc_2nd_derivatives(du_xs, du_ys, du_zs, dv_xs, dv_ys, dv_zs)

    print(dv_xs.shape)

    # print('dv_xs', dv_xs)
    print('dv_ys', dv_ys)
    print('dv_zs', dv_zs)

    # print('dvdv_xs', dvdv_xs)
    print('dvdv_ys', dvdv_ys)
    print('dvdv_zs', dvdv_zs)

    # To match the shape:
    du_xs, du_ys, du_zs, dv_xs, dv_ys, dv_zs = [
        a[:-1, :-1] for a in [du_xs, du_ys, du_zs, dv_xs, dv_ys, dv_zs]]

    N_xs, N_ys, N_zs = origamimetric._calc_normals(du_xs, du_ys, du_zs, dv_xs, dv_ys, dv_zs)

    j0 = 3
    dvdv_x1, dvdv_y1, dvdv_z1 = dv_xs[1, j0] - dv_xs[0, j0], dv_ys[1, j0] - dv_ys[0, j0], dv_zs[1, j0] - dv_zs[0, j0]
    dvdv_x2, dvdv_y2, dvdv_z2 = dv_xs[10, j0] - dv_xs[9, j0], dv_ys[10, j0] - dv_ys[9, j0], dv_zs[10, j0] - dv_zs[9, j0]
    print("dvdv1", dvdv_x1, dvdv_y1, dvdv_z1)
    print("dvdv2", dvdv_x2, dvdv_y2, dvdv_z2)

    print("here")

    for i in range(rows - 2):
        for j in range(cols - 2):
            x0, y0, z0 = metric_dot_xs[i, j], metric_dot_ys[i, j], metric_dot_zs[i, j]
            nx, ny, nz = N_xs[i, j], N_ys[i, j], N_zs[i, j]
            dvdv_x, dvdv_y, dvdv_z = dvdv_xs[i, j], dvdv_ys[i, j], dvdv_zs[i, j]
            dv_x, dv_y, dv_z = dv_xs[i, j], dv_ys[i, j], dv_zs[i, j]
            nx /= 5
            ny /= 5
            nz /= 5
            dvdv_x *= 5
            dvdv_y *= 5
            dvdv_z *= 5
            # ax.plot([x0, x0 + nx], [y0, y0 + ny], [z0, z0 + nz])
            # ax.plot([x0, x0 + dvdv_x], [y0, y0 + dvdv_y], [z0, z0 + dvdv_z])
            # ax.plot([x0, x0 + dv_x], [y0, y0 + dv_y], [z0, z0 + dv_z])
    print("there")

    plotutils.set_3D_labels(ax)

    # fig, ax = plt.subplots()
    # ax.plot(metric_dot_ys[:, j0], metric_dot_zs[:, j0], '.')

    Ks, g11, g12, g22 = origamimetric.calc_curvature_by_christoffel(ori.dots)

    fig, axes = plt.subplots(2, 2)
    imshow_with_colorbar(fig, axes[0, 0], g11, "g11")
    imshow_with_colorbar(fig, axes[0, 1], g22, "g22")
    imshow_with_colorbar(fig, axes[1, 0], g12, "g12")
    imshow_with_colorbar(fig, axes[1, 1], Ks, "K - christoffel")

    plt.show()

    b22 = dvdv_xs * N_xs + dvdv_ys * N_ys + dvdv_zs * N_zs
    print(b22)

    Ks_christoffel = origamimetric.calc_curvature_by_christoffel(ori.dots)

    # ax.scatter([1, 2, 3], [4, 4, 4], [5, 2, 4], '*')

    # fig, ax = plt.subplots()
    # imshow_with_colorbar(fig, ax, KsOther, "Ks by triangles")

    dF = FF(1) - FF(0)
    ddMM = MM(2) + MM(0) - 2 * MM(1)
    # expectedK = -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dF * \
    #             ddMM * (cos(W0) - 2 * csc(angle) ** 2 + 1)
    # print(expectedK)

    # plot_interactive(ori)
    plt.show()


def main():
    test_shape_operator()
    # test_metric_calc()


if __name__ == '__main__':
    main()
