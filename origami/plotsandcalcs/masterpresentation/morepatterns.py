import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D

from origami import origamiplots
from origami.RFFQMOrigami import RFFQM
from origami.alternatingpert import curvatures
from origami.alternatingpert.utils import get_pert_list_by_func, create_perturbed_origami
from origami.angleperturbation import set_perturbations_by_func
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.plotsandcalcs.compatibility.continuouspertalternating import create_angles_func_alternating_with_G
from origami.plotsandcalcs.masterpresentation import FIGURES_PATH
from origami.quadranglearray import dots_to_quadrangles
from origami.utils import plotutils


def plot_spherical_cap():
    rows, cols = 20, 30
    kx = -0.12
    ky = -0.12

    W0 = 2.4
    theta = 1.1

    Delta0 = 0.5
    delta0 = +0.2

    Nx, Ny = cols // 2, rows // 2
    L_tot, C_tot = 10, 15

    delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
    Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

    ori = create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)
    ori.set_gamma(0)

    fig, _ = origamiplots.plot_crease_pattern(ori, background_color=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.svg'), transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap-crease-pattern.png'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=-155, azim=-61,
                                 computed_zorder=False)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    quads = ori.dots
    quads.dots[2, :] -= np.max(quads.dots[2, :])
    light_source = LightSource(azdeg=315 - 90 - 90, altdeg=45)
    quads.plot(ax, panel_color='C1',
               edge_color='g', lightsource=light_source)

    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()

    bbox = fig.get_tightbbox()
    new_bbox = bbox.expanded(1.08, 0.70)
    new_bbox = Bbox.from_bounds(new_bbox.x0 + 0.05, new_bbox.y0, new_bbox.width, new_bbox.height - 0.2)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.svg'), bbox_inches=new_bbox, transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.pdf'), bbox_inches=new_bbox)
    fig.savefig(os.path.join(FIGURES_PATH, 'spherical-cap.png'), bbox_inches=new_bbox, dpi=300)
    # plt.show()
    # origamiplots.plot_interactive(ori)


def plot_twist():
    rows = 20
    cols = 20

    F1 = lambda x: 0 * x
    F2 = lambda x: 0 * x
    G = lambda y: 0.025 * (y - cols / 2)

    angle = 1
    ls = np.ones(rows)
    cs = np.ones(cols) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    pert_func = create_angles_func_alternating_with_G(F1, F2, G)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom, 'delta+eta')

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)

    fig, ax = origamiplots.plot_crease_pattern(ori, rotate_angle=-G(0), background_color=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'twist-flat.svg'), transparent=True)
    fig.savefig(os.path.join(FIGURES_PATH, 'twist-flat.png'))

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=20, azim=-123)

    ori.set_gamma(2.3)
    light_source = LightSource(azdeg=315 - 45, altdeg=45)
    ori.dots.plot(ax, panel_color='C1', alpha=1, lightsource=light_source)

    ax.set_aspect('equal')
    fig.tight_layout()
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'twist-folded.svg'),
                               1, 0.75, pad_x=-0.1, translate_x=0.1, transparent=True)
    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'twist-folded.png'),
                               1, 0.75, pad_x=-0.1, translate_x=0.1, transparent=True)


def main():
    # plot_spherical_cap()
    plot_twist()


if __name__ == '__main__':
    main()
