import os.path
from fractions import Fraction

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import origami.plotsandcalcs
from origami import quadranglearray
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
    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, ax = plot_flat_configuration(origami)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(os.path.join(FIGURES_PATH, "simple-example-flat.pdf"), pad_inches=-1)

    origami.set_omega(0.5)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=167, elev=47)
    origami.plot(ax, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_zticks([-0.5, 0, 0.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plotutils.set_labels_off(ax)
    # ax.set_zlim(-2, 2)

    fig.savefig(
        os.path.join(FIGURES_PATH, "simple-example-folded.pdf"), pad_inches=-0.2
    )

    plt.show()


def plot_full_cylinder():
    dots, rows, cols = create_full_cylinder()
    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, ax = plot_flat_configuration(origami)
    # ax.set_axis_off()
    # ax.zaxis.set_label_position('none')
    # ax.zaxis.set_ticks_position('none')
    ax.set_zticks([])
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('')
    ax.tick_params(axis='y', which='major', pad=10)
    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-flat.png"), pad_inches=-0.3)

    origami.set_omega(1.85)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-30, elev=21)
    origami.plot(ax, alpha=0.7)
    plotutils.set_3D_labels(ax)
    ax.set_xlabel('X', labelpad=10)

    fig.savefig(os.path.join(FIGURES_PATH, "cylinder-folded.png"))

    plt.show()


def plot_spiral():
    dots, rows, cols = create_spiral()
    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, _ = plot_flat_configuration(origami)
    fig.savefig(os.path.join(FIGURES_PATH, "spiral-flat.png"))

    origami.set_omega(2.2)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-20, elev=15)
    origami.plot(ax, alpha=0.7)
    ax.set_xlim(-3, 3)
    fig.savefig(os.path.join(FIGURES_PATH, "spiral-folded.png"))

    plt.show()


def plot_flat_configuration(origami):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=90, elev=-100)
    origami.set_omega(0)
    origami.plot(ax)
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
    # alpha and beta are kind of the same, beta=pi-alpha.

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

    ax.set_xlabel(r"$ \beta $")
    ax.set_ylabel(r"$ \theta $ vs $\beta$")

    plot_for_omega(omega=0.3)
    plot_for_omega(omega=1)
    plot_for_omega(omega=3)

    ax.legend()

    fig.savefig(os.path.join(FIGURES_PATH, "theta_vs_beta.pdf"))

    plt.show()


def plot_unit_cell():
    dxs = np.array([1, 1.5])
    cols = 3
    rows = 2
    ls = 1.1
    angles = [0.6 * np.pi, 0.7 * np.pi]
    dots = create_zigzag_dots(angles, cols, ls, dxs)

    origami = ZigzagMiuraOri(dots, rows, cols)

    fig, ax = plot_flat_configuration(origami)
    edge_points = origami.dots[
                  :,
                  [
                      origami.indexes[0, 0],
                      origami.indexes[0, -1],
                      origami.indexes[-1, 0],
                      origami.indexes[-1, -1],
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
    # plot_spiral()
    # plot_full_cylinder()
    # plot_simple_example()
    # plot_theta_vs_alpha()
    # plot_unit_cell()
    plot_different_scaling()


if __name__ == "__main__":
    main()
