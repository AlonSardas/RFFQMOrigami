import os.path
from fractions import Fraction

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import origami.plotsandcalcs
from origami import quadranglearray, origamiplots, RFFQMOrigami, marchingalgorithm
from origami.origamiplots import plot_interactive_miuraori
from origami.utils import plotutils
from origami.zigzagmiuraori import ZigzagMiuraOri, create_zigzag_dots

FIGURES_PATH = os.path.join(
    origami.plotsandcalcs.BASE_PATH, "RFFQM", "Figures", "zigzag-origami"
)


def create_basic_crease1():
    n = 3
    dx = 1
    dy = 2

    rows = 3
    angles = 0.4 * np.ones(rows) * np.pi

    dots = create_zigzag_dots(angles, n, dy, dx)

    return dots, len(angles), n


def create_basic_crease2():
    n = 7
    dx = 1
    dy = 1.5

    angles = np.array([0.25, 0.3, 0.65, 0.6, 0.65]) * np.pi
    angles = np.pi - angles

    dots = create_zigzag_dots(angles, n, dy, dx)

    return dots, len(angles), n


def create_full_cylinder():
    n = 7
    rows = 21
    dx = 1
    dy = 4

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    angles = np.pi - angles
    dots = create_zigzag_dots(angles, n, dy, dx)
    return dots, len(angles), n


def create_cylinder_changing_dxs():
    n = 10
    rows = 21
    dxs = np.array([1, 2, 1, 3, 1, 4, 1, 5, 1])
    dy = 20

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    dots = create_zigzag_dots(angles, n, dy, dxs)
    return dots, len(angles), n


def create_spiral():
    cols = 5
    rows = 40
    dx = 1
    ls = 2 + np.arange(rows - 1) * 0.15

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    dots = create_zigzag_dots(angles, cols, ls, dx)
    return dots, rows, cols


def create_changing_cs_example():
    rows = 10
    # dxs = np.array([4, 7, 4, 8, 4, 5, 4, 2, 7, 5, 4, 2, 4, 3, 4, 4, 4, 5, 4]) * 0.6
    dxs = np.array([1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7]) * 0.6
    dxs = np.append(dxs, dxs[::-1])
    cols = len(dxs) + 1
    ls = 20

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.3 * np.pi
    # angles[::2] += 0.1 * np.pi
    angles[::2] += 0.4 * np.pi
    dots = create_zigzag_dots(angles, cols, ls, dxs)
    return dots, rows, cols


def create_spiral_changing_cs():
    rows = 40
    dxs = np.array([1, 0.5] * 30) * 0.1
    cols = len(dxs) + 1
    ls = 4 + np.arange(rows - 1) * 0.15

    # angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi
    angles = np.ones(rows) * 0.1 * np.pi
    angles[::2] += 0.1 * np.pi
    dots = create_zigzag_dots(angles, cols, ls, dxs)
    return dots, rows, cols


def plot():
    # logutils.enable_logger()

    # dots, rows, cols = create_full_cylinder()
    # dots, rows, cols = create_spiral()
    # dots, rows, cols = create_changing_cs_example()
    # dots, rows, cols = create_spiral_changing_cs()
    # dots, rows, cols = create_cylinder_changing_dxs()
    dots, rows, cols = create_basic_crease1()
    ori = ZigzagMiuraOri(dots, rows, cols)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ori.plot(ax)

    ori.set_omega(1)
    valid, reason = quadranglearray.is_valid(ori.initial_dots, ori.dots, ori.indexes)
    if not valid:
        # raise RuntimeError(f'Not a valid folded configuration. Reason: {reason}')
        print(f"Not a valid folded configuration. Reason: {reason}")

    plot_interactive_miuraori(ori)


def plot_simple_example():
    dots, rows, cols = create_basic_crease2()
    ori = ZigzagMiuraOri(dots, rows, cols)

    ori2 = RFFQMOrigami.RFFQM(ori.get_quads())
    quads = ori2.dots
    fig, ax = origamiplots.plot_crease_pattern(ori2)
    quadranglearray.plot_2D_polygon(quads, ax)
    fig.savefig(os.path.join(FIGURES_PATH, "simple-example-flat.pdf"))
    # return

    ori2.set_gamma(ori2.calc_gamma_by_omega(-0.5))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=167, elev=47)
    ori2.dots.plot(ax, alpha=1, edge_width=2.5)
    ax.set_aspect("equal")
    ax.set_zticks([-0.5, 0, 0.5])
    plotutils.remove_tick_labels(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlim(-2, 2)

    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, "simple-example-folded.pdf"),
                               1, 0.78)
    # fig.savefig(
    #     os.path.join(FIGURES_PATH, "simple-example-folded.pdf"), pad_inches=-0.2
    # )

    plt.show()


def plot_full_cylinder():
    dots, rows, cols = create_full_cylinder()
    ori = ZigzagMiuraOri(dots, rows, cols)

    fig, ax = plot_flat_configuration(ori)
    # ax.set_axis_off()
    # ax.zaxis.set_label_position('none')
    # ax.zaxis.set_ticks_position('none')
    ax.set_zticks([])
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('')
    ax.tick_params(axis='y', which='major', pad=10)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-flat2.png"), pad_inches=-0.3)
    ori2 = RFFQMOrigami.RFFQM(ori.get_quads())
    fig, ax = origamiplots.plot_crease_pattern(ori2)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-flat.svg"), pad_inches=-0.3)

    ori.set_omega(1.85)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-30, elev=21)
    quads = ori.get_quads()
    quads.plot(ax, panel_color='C1', alpha=0.9)
    plotutils.set_3D_labels(ax)
    ax.set_xlabel('X', labelpad=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-folded.png"), pad_inches=0.3, dpi=300)

    plt.show()


def plot_cylinder2():
    angle = 0.54 * np.pi
    ls = 0.7 * np.ones(18)
    cs = np.ones(10)

    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)

    angles_left[:, 1::2] += 0.15
    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    quads = quadranglearray.dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQMOrigami.RFFQM(quads)

    fig, ax = origamiplots.plot_crease_pattern(ori, background_color='0.9')
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-flat2.pdf"))

    ori.set_gamma(-0.5)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=27, azim=148)
    ori.dots.plot(ax, panel_color='C1', alpha=0.9)
    plotutils.set_3D_labels(ax, z_pad=0)
    plotutils.set_axis_scaled(ax)
    plotutils.remove_tick_labels(ax)

    fig.tight_layout()
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, "cylinder-folded2.pdf"),
                               0.95, 0.80, translate_x=0.5, translate_y=-0.20)
    # fig.savefig(os.path.join(FIGURES_PATH, "cylinder-folded2.pdf"), pad_inches=0.3)

    plt.show()


def plot_cylinder_small():
    angle = 0.7 * np.pi
    ls = np.ones(10)
    cs = np.ones(10)

    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)

    angles_left[:, 1::2] += 0.3
    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    quads = quadranglearray.dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQMOrigami.RFFQM(quads)

    fig, ax = origamiplots.plot_crease_pattern(ori)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-small-flat.svg"))

    ori.set_gamma(-1.9)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=27, azim=148)
    ori.dots.plot(ax, panel_color='C1', alpha=0.9)
    plotutils.set_3D_labels(ax)
    plotutils.set_axis_scaled(ax)
    # ax.set_xlabel('X', labelpad=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-small-folded.png"), pad_inches=0.3, dpi=300)

    plt.show()


def plot_spiral():
    dots, rows, cols = create_spiral()
    ori = ZigzagMiuraOri(dots, rows, cols)

    fig, _ = plot_flat_configuration(ori)
    fig.savefig(os.path.join(FIGURES_PATH, "spiral-flat.png"))

    ori.set_omega(2.2)

    quads = ori.get_quads()

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-20, elev=15)
    quads.plot(ax, panel_color='C1', alpha=1)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xticks([-2, 0, 2])
    plotutils.set_3D_labels(ax)
    fig.tight_layout()
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, "spiral-folded.pdf"),
                               1, 0.8, translate_x=0.4, translate_y=-0.1)
    # fig.savefig(os.path.join(FIGURES_PATH, "spiral-folded.pdf"))

    plt.show()


def plot_flat_configuration(ori):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=90, elev=-100)
    ori.set_omega(0)
    ori.plot(ax)
    ax.set_zlim(-1, 1)
    return fig, ax


def plot_theta_vs_alpha_2_subplots():
    # alpha and beta are kind of the same, beta=pi-alpha.

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    xs = np.linspace(0, np.pi, 200)

    def plot_for_omega(ax, omega):
        s_a = np.sin(xs)
        s_a_2 = s_a ** 2
        c_o = np.cos(omega)
        ys = (
                1
                / 2
                * np.arccos((2 - 3 * s_a_2 + s_a_2 * c_o) / (-2 + s_a_2 + s_a_2 * c_o))
        )

        ax.plot(xs, ys)

        plotutils.set_pi_ticks(ax, "x")
        plotutils.set_pi_ticks(ax, "y", (0, Fraction(1, 2)))

        ax.set_xlabel(r"$ \beta $")
        # ax.set_ylabel(r"$ \theta\left(\omega=" + str(omega) + r"\right) $ vs  $ \beta $")
        ax.set_ylabel(r"$ \theta $ vs $\beta$")
        ax.set_title(r"$ \omega=" + str(omega) + " $")

    plot_for_omega(axes[0], omega=0.7)
    plot_for_omega(axes[1], omega=3)

    fig.savefig(os.path.join(FIGURES_PATH, "theta_vs_beta.pdf"))

    plt.show()


def plot_theta_vs_alpha():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    xs = np.linspace(0, np.pi, 400)

    def plot_for_omega(omega):
        s_a = np.sin(xs)
        s_a_2 = s_a ** 2
        c_o = np.cos(omega)
        ys = (
                1
                / 2
                * np.arccos((2 - 3 * s_a_2 + s_a_2 * c_o) / (-2 + s_a_2 + s_a_2 * c_o))
        )

        ax.plot(xs, ys, label=r"$ \omega=" + str(omega) + " $")

    plotutils.set_pi_ticks(ax, "x")
    plotutils.set_pi_ticks(ax, "y", (0, Fraction(1, 2)))

    ax.set_xlabel(r"$ \vartheta $")
    ax.set_ylabel(r"$ \phi $")

    plot_for_omega(omega=0.3)
    plot_for_omega(omega=1)
    plot_for_omega(omega=3)

    ax.legend()

    fig.savefig(os.path.join(FIGURES_PATH, "phi_vs_angle.pdf"))

    plt.show()


def plot_unit_cell():
    dxs = np.array([1, 1.5])
    cols = 3
    rows = 2
    ls = 1.1
    angles = [0.6 * np.pi, 0.7 * np.pi]
    dots = create_zigzag_dots(angles, cols, ls, dxs)

    ori = ZigzagMiuraOri(dots, rows, cols)

    fig, ax = plot_flat_configuration(ori)
    edge_points = ori.dots[
                  :,
                  [
                      ori.indexes[0, 0],
                      ori.indexes[0, -1],
                      ori.indexes[-1, 0],
                      ori.indexes[-1, -1],
                  ],
                  ]
    ax.scatter3D(
        edge_points[0, :], edge_points[1, :], edge_points[2, :], color="r", s=220
    )
    fig.savefig(os.path.join(FIGURES_PATH, "zigzag-FFF-unit.png"))
    plt.show()


def plot_different_scaling():
    n = 7

    def plot_zigzag(base, F, rows, dx, dy):
        angles = np.ones(rows) * base
        angles[::2] += F
        angles = np.pi - angles
        dots = create_zigzag_dots(angles, n, dy, dx)
        ori = ZigzagMiuraOri(dots, rows, n)

        ori.set_omega(1.5)

        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-30, elev=21)
        ori.plot(ax, alpha=0.7)
        plotutils.set_3D_labels(ax)

    plot_zigzag(0.1 * np.pi, 0.1 * np.pi, 10, 1, 4)
    plot_zigzag(0.1 * np.pi, 0.1 * np.pi, 50, 1, 4)
    plot_zigzag(0.1 * np.pi, 0.1 * np.pi / 5, 50, 1 / 5, 4 / 5)
    plt.show()


def main():
    # plot()
    plot_spiral()
    # plot_full_cylinder()
    # plot_cylinder_small()
    # plot_simple_example()
    # plot_cylinder2()
    # plot_theta_vs_alpha()
    # plot_unit_cell()
    # plot_different_scaling()


if __name__ == "__main__":
    main()
