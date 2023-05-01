import matplotlib.pyplot as plt
import numpy as np

from origami import quadranglearray, origamimetric
from origami.utils import plotutils


def test_shape_operator():
    rows, cols = 21, 31
    x = np.linspace(-1/2, 1/2, cols)
    y = np.linspace(-1/2, 1/2, rows)
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

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks')
    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks_by_shape, 'Ks by triangles')

    plt.show()


def main():
    test_shape_operator()


if __name__ == '__main__':
    main()
