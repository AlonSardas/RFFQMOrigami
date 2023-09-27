import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import origami.plotsandcalcs
from origami import origamimetric
from origami import quadranglearray
from origami.utils import plotutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH,
                            'RFFQM/BasicGeometry')


def test_periodic_by_extended_ruled_surface():
    L = 1
    ny, nx = 70, 70
    N = nx * ny
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))
    fig:Figure = fig

    plt.rcParams['axes.titley'] = 1.2
    plt.rcParams['axes.titlepad'] = -10

    def calc_periodic_surface(Ax, Ay, i):
        xs, ys = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, L, ny))
        zs = Ax * np.sin(2 * np.pi * xs / 0.45) + Ay * np.sin(2 * np.pi * ys / 0.67)

        dots = np.array([xs.reshape(N), ys.reshape(N), zs.reshape(N)])

        quads = quadranglearray.QuadrangleArray(dots, ny, nx)
        quads.center()
        # quadranglearray.plot_flat_quadrangles(quads)

        Ks, _, _, _, = origamimetric.calc_curvature_and_metric(quads)

        dx = L / nx
        dy = L / ny

        h_x = np.gradient(zs, axis=1) / dx
        h_y = np.gradient(zs, axis=0) / dy
        h_xx = np.gradient(h_x, axis=1) / dx
        h_yy = np.gradient(h_y, axis=0) / dy

        expected_Ks_linear = h_xx * h_yy
        expected_Ks_exact = (h_xx * h_yy) / (1 + h_x ** 2 + h_y ** 2) ** 2

        im1 = plotutils.imshow_with_colorbar(fig, axes[0, i], Ks, f"Ks: Ax={Ax},Ay={Ay}")
        # label1 = "Ks linearized\n" "$ h_{xx}^2+h_{yy}^2 $"
        im2 = plotutils.imshow_with_colorbar(fig, axes[2, i], expected_Ks_linear, "approximated Monge")
        im3 = plotutils.imshow_with_colorbar(fig, axes[1, i], expected_Ks_exact, "exact by Monge")
        im1.set_extent([0, L, 0, L])
        im2.set_extent([0, L, 0, L])
        im3.set_extent([0, L, 0, L])

    calc_periodic_surface(0.01, 0.01, 0)
    calc_periodic_surface(0.1, 0.01, 1)
    fig.suptitle("Surfaces of the form:\n" r"h(x,y)=$A_x \sin(x/ \alpha) + A_y \sin(x/ \beta)$")
    # fig.supxlabel("Surfaces of the form:\n" r"$A_x \sin(x/ \alpha) + A_y \sin(x/ \beta)$")
    fig.tight_layout()
    # fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(FIGURES_PATH, 'periodic-perturbation-comparison.png'))
    plt.show()


def main():
    test_periodic_by_extended_ruled_surface()


if __name__ == "__main__":
    main()
