import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami import origamiplots, marchingalgorithm, origamimetric, quadranglearray
from origami.RFFQMOrigami import RFFQM
from origami.plotsandcalcs import articleillustrations
from origami.alternatingpert.utils import create_perturbed_origami, create_perturbed_origami_by_list
from origami.quadranglearray import dots_to_quadrangles
from origami.utils import plotutils

FIGURES_PATH = os.path.join(articleillustrations.FIGURES_PATH, 'intro-fig')

EDGE_COLOR = 'g'
EDGE_ALPHA = 1.0
EDGE_WIDTH = 3


def plot_unperturbed_Miura_Ori():
    theta = 1.3
    Nx, Ny = 4, 4
    C_tot, L_tot = 5, 5
    W0 = 1.5
    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, None, None)
    ori.set_gamma(ori.calc_gamma_by_omega(0))

    # fig, ax = plt.subplots()
    # ori.dots.plot(ax, alpha=0.6, edge_alpha=0, panel_color=articleillustrations.FLAT_PANELS_COLOR)
    # origamiplots.draw_creases(ori.dots, 1, ax)
    fig, ax = origamiplots.plot_crease_pattern(ori)
    fig.savefig(os.path.join(FIGURES_PATH, 'unperturbed-Miura-Ori.pdf'))

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    fig: Figure = plt.figure(layout='compressed')
    ax: Axes3D = fig.add_subplot(111, projection="3d",
                                 elev=22, azim=-119)
    panels = ori.dots.plot(ax, panel_color=articleillustrations.PANELS_COLOR,
                           edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA, edge_width=EDGE_WIDTH)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'unperturbed-Miura-Ori-folded.png'),
                               1.2, 0.6)
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'unperturbed-Miura-Ori-folded.pdf'),
                               1.2, 0.6)
    # fig.savefig(os.path.join(FIGURES_PATH, 'unperturbed-Miura-Ori-folded.pdf'))
    plt.show()

    plt.show()


def plot_general_perturbations():
    theta = 1.3
    gamma0 = 1.2

    cs = np.array((1, 2, 1.3, 1.4, 1, 0.8, 1, 1, 0.8, 1.4, 2, 2.2, 1))
    ls = np.array((2, 1.3, 1, 1.8, 0.9, 2, 1, 0.6, 1.0, 0.6))

    angles_bottom = np.array(
        [[2, 1.0, 2.2, 1, 2.1, 1.1, 2.1, 1.0, 2.1, 0.9, 2.0, 1.1, 2.3],
         [2.1, 1.0, 1.9, 1.2, 2.0, 1.11, 2.2, 1.0, 2.1, 1.0, 2.1, 1.1, 2.1]])
    angles_left = np.array(
        [[1, 1.1, 0.8, 1.2, 0.9, 1.4, 1.4, 1.3, 1.2, 1.2, 1.3],
         [1.0, 1, 1, 0.9, 1.3, 0.9, 1.3, 1.1, 1.2, 1.2, 1.2]])
    # print(angles_left.shape, angles_bottom.shape)
    # angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, theta)
    # print(angles_left.shape, angles_bottom.shape)

    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)

    fig, ax = origamiplots.plot_crease_pattern(ori, rotate_angle=-0.4)
    fig.set_facecolor('#FFFFFF00')
    fig.savefig(os.path.join(FIGURES_PATH, 'general-perturbations-flat.png'), transparent=True)
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'general-perturbations-flat.pdf'), 0.95, 0.95,
                               transparent=True)

    ori.set_gamma(gamma0)

    fig: Figure = plt.figure(layout='compressed')
    ax: Axes3D = fig.add_subplot(111, projection="3d",
                                 elev=54, azim=-39)
    ori.dots.plot(ax, panel_color=articleillustrations.PANELS_COLOR,
                  edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA, edge_width=EDGE_WIDTH)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'general-perturbations-folded.png'),
                               1.05, 0.70, translate_x=-0.25, translate_y=-0.3, transparent=True)
    # plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'general-perturbations-folded.pdf'),
    #                            1.05, 0.52, transparent=True)

    origamiplots.plot_interactive(ori)


def plot_alternating_perturbations():
    theta = 1.3
    W0 = 1.1

    Deltas = np.array((-0.5, 0.1, -0.3, 0.2, 0.4, 1, 1.5, 1.0, 1.2))
    deltas = 0.9 * np.array((0.5, 0.3, 0.1, -0.2, 0.2, 0.1, -0.2, 0.1, 0.2, -0.3, -0.1, -0.2, 0.1))

    C0, L0 = 1.5, 1

    ori = create_perturbed_origami_by_list(theta, L0, C0, deltas, Deltas)
    fig, ax = origamiplots.plot_crease_pattern(ori, rotate_angle=deltas[0])
    fig.savefig(os.path.join(FIGURES_PATH, 'alternating-perturbations-flat.pdf'))

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    fig: Figure = plt.figure(layout='compressed')
    ax: Axes3D = fig.add_subplot(111, projection="3d",
                                 elev=31, azim=135)
    lightsource = LightSource(azdeg=45)
    ori.dots.plot(ax, panel_color=articleillustrations.PANELS_COLOR,
                  edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA, edge_width=EDGE_WIDTH,
                  lightsource=lightsource)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'alternating-perturbations-folded.png'),
                               1.2, 0.6, transparent=True)
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'alternating-perturbations-folded.pdf'),
                               1.2, 0.6)

    origamiplots.plot_interactive(ori)


def create_random_pattern():
    theta = 1.3
    Nx, Ny = 9, 9
    W0 = 1.5

    def generate_random_pert():
        ls = np.random.uniform(1.5, 3, 2 * Ny)
        cs = np.random.uniform(1.5, 3, 2 * Nx)
        print(
            f"""ls = np.{repr(ls)}
    cs = np.{repr(cs)}""")

        angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, theta)
        left_pert = np.random.uniform(-1, 1, size=angles_left.shape)
        bottom_pert = np.random.uniform(-1, 1, size=angles_bottom.shape)
        print(f"""
    left_pert = np.{repr(left_pert)}
    bottom_pert = np.{repr(bottom_pert)}
    """)
        return ls, cs, left_pert, bottom_pert

    ls, cs, left_pert, bottom_pert = generate_random_pert()

    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, theta)

    angles_bottom[:, ::2] += 0.05

    pert_factor_left = 0.015
    pert_factor_bottom = 0.01
    ang_pert_left = pert_factor_left * left_pert
    ang_pert_bottom = pert_factor_bottom * bottom_pert
    angles_left += ang_pert_left
    angles_bottom += ang_pert_bottom
    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)


def plot_periodic_smooth():
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    Xs, Ys = np.meshgrid(xs, ys)

    ox, oy = 3.0, 3.0

    z_factor = 0.02
    Zs = -z_factor * (np.sin(2 * np.pi * Xs * ox) + np.sin(2 * np.pi * Ys * oy))

    quads = quadranglearray.from_mesh_gird(Xs, Ys, Zs)
    origamimetric.skip_half_dots = False
    geometry = origamimetric.OrigamiGeometry(quads)
    K1, K2 = geometry.get_principal_curvatures()
    # fig, axes = plt.subplots(1, 2)
    # plotutils.imshow_with_colorbar(fig, axes[0], K1, 'K1')
    # plotutils.imshow_with_colorbar(fig, axes[1], K2, 'K2')

    # This is a try to recover the principal curvatures in the different directions
    C11, C12, C21, C22 = geometry.get_shape_operator()

    # fig, ax = plt.subplots()
    # plotutils.imshow_with_colorbar(fig, ax, C11, 'C11')
    # plt.show()
    # return

    Ky = K1 * (C11 > C22) + K2 * (C11 <= C22)
    Kx = K2 * (C11 > C22) + K1 * (C11 <= C22)

    # mpl.rcParams['font.size'] = 40

    def plot_kappas(data, xy):
        fig, ax = plt.subplots(layout="compressed")
        ax.imshow(data)
        ax.invert_yaxis()
        plotutils.remove_tick_labels(ax)
        # ax.set_title(fr'$ \kappa_{xy} $')
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FIGURES_PATH, f'periodic-smooth-k{xy}.pdf'))
        fig.savefig(os.path.join(FIGURES_PATH, f'periodic-smooth-k{xy}.svg'), pad_inches=-0.05)

    plot_kappas(Kx, 'x')
    plot_kappas(Ky, 'y')

    should_plot_curvatures = False
    if should_plot_curvatures:
        Ks, Hs = geometry.get_curvatures_by_shape_operator()
        fig, axes = plt.subplots(1, 2)
        plotutils.imshow_with_colorbar(fig, axes[0], Ks, 'Ks')
        plotutils.imshow_with_colorbar(fig, axes[1], Hs, 'Hs')

    # plt.show()

    fig: Figure = plt.figure(layout='compressed')
    ax: Axes3D = fig.add_subplot(111, projection="3d",
                                 # elev=31, azim=135)
                                 elev=27, azim=-124)
    ax.plot_surface(Xs, Ys, Zs, rcount=50, ccount=50, linewidth=1, antialiased=True)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()

    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'periodic-smooth.png'), 1.2, 0.65, transparent=True)

    plt.show()


def main():
    # plot_unperturbed_Miura_Ori()
    # plot_general_perturbations()
    plot_alternating_perturbations()
    # plot_periodic_smooth()


if __name__ == '__main__':
    main()
