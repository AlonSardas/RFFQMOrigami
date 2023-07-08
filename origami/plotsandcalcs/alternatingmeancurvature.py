import matplotlib.pyplot as plt
import numpy as np

from origami import origamimetric
from origami.plotsandcalcs.alternatingbetterapprox import create_expected_curvatures_func
from origami.plotsandcalcs.continuousmetricalternating import create_perturbed_origami
from origami.quadranglearray import plot_flat_quadrangles
from origami.utils.plotutils import imshow_with_colorbar


def test_cylinders():
    """
    In cylinders, we expect to have 0 Gaussian curvature and
    non-vanishing mean curvature
    """
    rows = 50
    cols = 30
    angle = np.pi / 2 - 0.1
    W0 = 2.9
    L0 = 1
    C0 = 0.5
    F = lambda x: 0.0 * x
    # MM = lambda y: 0.00 * (y-5) ** 2 + 0.1*y*np.sin(y/4+0.2)
    MM = lambda y: 0.01 * (y - 5) ** 2

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    # plot_flat_quadrangles(ori.dots)

    plt.show()
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, angle, F, MM)
    compare_curvatures(Ks, Hs, expected_K, expected_H)

    rows = 30
    cols = 80
    angle = np.pi / 2 - 0.2
    W0 = 3.0
    L0 = 1
    C0 = 0.5
    F = lambda x: 0.005 * (x - cols / 2)
    MM = lambda y: 0 * y
    # MM = lambda y: 0.01 * y**2

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, angle, F, MM)
    compare_curvatures(Ks, Hs, expected_K, expected_H)


def compare_curvatures(Ks, Hs, expected_K_func, expected_H_func):
    fig, axes = plt.subplots(2, 2)

    len_ys, len_xs = Ks.shape
    xs, ys = np.arange(len_xs), np.arange(len_ys)
    Xs, Ys = np.meshgrid(xs, ys)

    im = imshow_with_colorbar(fig, axes[0, 0], Ks, "K")
    vmin, vmax = im.get_clim()
    im2 = imshow_with_colorbar(fig, axes[1, 0], expected_K_func(Xs, Ys), "expected K")
    # im2.set_clim(vmin, vmax)

    im = imshow_with_colorbar(fig, axes[0, 1], Hs, "H")
    vmin, vmax = im.get_clim()
    im2 = imshow_with_colorbar(fig, axes[1, 1], expected_H_func(Xs, Ys), "expected H")
    # im2.set_clim(vmin, vmax)


def main():
    test_cylinders()
    plt.show()


if __name__ == "__main__":
    main()
