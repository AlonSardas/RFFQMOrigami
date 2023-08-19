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
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig:Figure = fig

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

        plotutils.imshow_with_colorbar(fig, axes[0, i], Ks, f"Ks: Ax={Ax},Ay={Ay}")
        # label1 = "Ks linearized\n" "$ h_{xx}^2+h_{yy}^2 $"
        plotutils.imshow_with_colorbar(fig, axes[2, i], expected_Ks_linear, "approximated monge")
        plotutils.imshow_with_colorbar(fig, axes[1, i], expected_Ks_exact, "exact by monge")

    calc_periodic_surface(0.01, 0.01, 0)
    calc_periodic_surface(0.1, 0.01, 1)
    fig.suptitle("Surfaces of the form:\n" r"h(x,y)=$A_x \sin(x/ \alpha) + A_y \sin(x/ \beta)$")
    # fig.supxlabel("Surfaces of the form:\n" r"$A_x \sin(x/ \alpha) + A_y \sin(x/ \beta)$")
    # fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'periodic-perturbation-comparison.png'))
    plt.show()


def main():
    test_periodic_by_extended_ruled_surface()


if __name__ == "__main__":
    main()
