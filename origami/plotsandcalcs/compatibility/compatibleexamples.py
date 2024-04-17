import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import origami.plotsandcalcs
from origami import RFFQMOrigami, origamiplots
from origami.angleperturbation import set_perturbations_by_func_v1
from origami.origamiplots import plot_interactive
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH, 'RFFQM/Compatibility/Figures')


def create_miura_ori():
    # angle = np.pi * 1 / 3
    angle = 1

    # ls = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ls = [2] * 10
    # cs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cs = [1, 1, 1, 1, 1, 1, 1, 1]

    angles_left = np.ones((2, len(ls) + 1)) * angle
    angles_bottom = np.ones((2, len(cs))) * angle
    angles_bottom[:, ::2] = np.pi - angle
    # angles_bottom[:, :] += 0.1
    # angles_left[0, 0] += 0.5
    # angles_left[1, 0] += -0.3
    # angles_left[0, 2] += 0.2
    # angles_left[1, 2] += -0.4

    angles_bottom[0, 0] += 0.5
    angles_bottom[1, 0] -= 0.3
    # angles_bottom[0, 1] += 0.1
    # angles_bottom[1, 1] += 0.1
    # angles_bottom[0, 2] += 0.1
    # angles_bottom[1, 2] += 0.1
    # angles_bottom[0, 3] += 0.1
    # angles_bottom[1, 3] += 0.1

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    print(marching.alphas)
    print(marching.betas)
    dots, indexes = marching.create_dots(ls, cs)

    fig, ax = plt.subplots()
    ax: Axes = ax
    ax.scatter(dots[0, :], dots[1, :])
    plt.axis('scaled')
    plt.show()


def plot_zigzag():
    angle = 1
    ls = [1] * 8
    cs = np.ones(4)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[0, 0] += 0.5
    angles_bottom[1, 0] += -0.3

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    fig.savefig(os.path.join(FIGURES_PATH, 'perturbation_zigzag.png'))


def plot_same_perturbed_angles():
    angle = 1
    ls = np.ones(6)
    cs = np.ones(6)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, 0] += 0.3
    # angles_left[:, 0] += 0.05
    # angles_bottom[1, 0] += 0.2

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    fig.savefig(os.path.join(FIGURES_PATH, 'same_perturbed_angles.png'))


def plot_radial_creases():
    angle = 1
    ls = np.ones(8)
    cs = np.ones(12)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, :] += 0.2
    # angles_left[:, 0] += 0.05
    # angles_bottom[1, 0] += 0.2

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    fig.savefig(os.path.join(FIGURES_PATH, 'radial_creases.png'))


def plot_all_bottom_perturbed():
    angle = 1
    ls = np.ones(5) * 2
    cs = np.ones(6)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_bottom[:, :] += 0.1
    angles_left[:, 0] += 0.1
    # angles_left[:, :] += 0.1

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    # fig.savefig(os.path.join(FIGURES_PATH, 'all_bottom_perturbed.png'))


def single_angle_perturbation():
    angle = 1
    ls = np.ones(2)
    cs = np.ones(2)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    # angles_left[0, 0] += 0.4
    # angles_left[0, 0] += 0.2    # delta_11
    # angles_left[1, 1] += 0.2    # eta_21
    angles_bottom[1, 0] += 0.2  # eta_12

    # angles_left[0, 1] += 0.4    # delta_21
    # angles_bottom[0, 0] += 0.4    # delta_12
    # angles_left[1, 0] += 0.1    # eta_11

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    # fig.savefig(os.path.join(FIGURES_PATH, 'all_bottom_perturbed.png'))


def plot_parallel_G_const():
    G_0 = 0.05
    F = lambda x: 0
    G = lambda y: G_0
    angle = 1

    rows = 20
    cols = 20
    ls = np.ones(rows - 1) * 1
    cs = np.ones(cols - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQMOrigami.RFFQM(quads)

    fig, ax = origamiplots.plot_crease_pattern(ori, background_color='0.9')
    fig.savefig(os.path.join(FIGURES_PATH, 'parallel-G-const-flat.pdf'))

    # fig = plt.figure()
    # ax: Axes3D = fig.add_subplot(111, projection='3d', elev=-159, azim=103)
    #
    # ori.set_gamma(2.8)
    # ori.dots.plot(ax, alpha=0.4)
    #
    # ax.dist = 8.5
    # ax.set_aspect('equal')
    # # ax.set(xticks=[], yticks=[], zticks=[])
    # # ax.set(zticks=[-3, 0, 3])
    # plotutils.set_labels_off(ax)
    # # ax.set_axis_off()
    # fig.tight_layout()
    # fig.savefig(os.path.join(FIGURES_PATH, 'alternating-with-G-folded.svg'))


def main():
    # create_miura_ori()
    # plot_zigzag()
    # plot_radial_creases()
    # plot_same_perturbed_angles()
    # plot_all_bottom_perturbed()
    # single_angle_perturbation()
    # plot_parallel_G_const()

    plt.show()


if __name__ == '__main__':
    main()
