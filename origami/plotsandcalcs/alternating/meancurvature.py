import os.path

import matplotlib.pyplot as plt
import numpy as np

import origami
from origami import origamimetric
from origami.plotsandcalcs.alternating.betterapprox import compare_curvatures
from origami.plotsandcalcs.alternating.betterapproxcurvatures import create_expected_curvatures_func
from origami.plotsandcalcs.alternating.utils import create_perturbed_origami
from origami.quadranglearray import plot_flat_quadrangles

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM/ContinuousMetric/AlternatingFigures/MeanCurvature/')


def test_cylinders():
    """
    In cylinders, we expect to have 0 Gaussian curvature and
    non-vanishing mean curvature
    """
    rows = 30
    cols = 80
    angle = np.pi / 2 - 0.1
    W0 = 3.0
    L0 = 1
    C0 = 0.5
    F = lambda x: 0.01 * (x - 1.0 * cols / 2)
    MM = lambda y: 0.0000 * y ** 2
    # MM = lambda y: 0.01 * y**2

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    omegas = ori.calc_omegas_vs_x()
    # This is a trick to correct that actual W0 by the place F vanishes
    W0 = omegas[len(omegas) // 2]

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, angle, F, MM)
    fig, _ = compare_curvatures(Ks, Hs, expected_K, expected_H)
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'simple-mean-x-axis.pdf'))
    plt.show()

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


def test_x_curvature():
    rows = 12
    cols = 120
    angle = np.pi / 2 - 0.1
    W0 = 2.9
    L0 = 1
    C0 = 0.5
    func = lambda x: 0.02 * np.cos(x / 16)  # - 0.2 * np.sin(x / 7) - 0.00001 * (x - 30) ** 2
    F = lambda x: func(x) - func(0)
    MM = lambda y: 0 * y

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    plot_flat_quadrangles(ori.dots)

    plt.show()
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    geometry = origamimetric.OrigamiGeometry(ori.dots)
    Ks, Hs = geometry.get_curvatures_by_shape_operator()
    expected_K, expected_H = create_expected_curvatures_func(L0, C0, W0, angle, F, MM)
    compare_curvatures(Ks, Hs, expected_K, expected_H)


def main():
    test_cylinders()
    # test_x_curvature()
    plt.show()


if __name__ == "__main__":
    main()
