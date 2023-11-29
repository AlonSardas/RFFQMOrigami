import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc, FancyArrowPatch

import origami.plotsandcalcs
from origami import marchingalgorithm, origamiplots
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.quadranglearray import dots_to_quadrangles
from origami.RFFQMOrigami import RFFQM
from origami.utils import linalgutils

FIGURES_PATH = os.path.join(origami.plotsandcalcs.BASE_PATH, 'RFFQM', 'Figures')


def plot_panel_illustration():
    ls = [1.3, 0.8, 1.5]
    cs = [1, 1.3, 1.2]
    angle = 1.1
    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    # add some perturbations
    angles_left[0, :] += [-0.2, -0.1, 0.2, -0.1]
    angles_left[1, :] += [-0.1, 0.2, -0.15, -0.2]
    angles_bottom[0, :] += [0.1, 0.2, -0.1]
    angles_bottom[1, :] += [0.2, -0.1, -0.1]

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    # ori = RFFQM(quads)
    # origamiplots.plot_crease_pattern(ori)

    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)

    rot_angle = 0.2
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    quads.dots = rot @ quads.dots

    dots = quads.dots[:2, :]
    dots = np.array(dots, 'float64')
    indexes = quads.indexes

    arc_size = 0.2
    arc_pad = 0.08
    angle_color = 'orangered'
    plt.rcParams.update({'font.size': 30})

    def plot_line(i0, j0, i1, j1, color):
        ax.plot((dots[0, indexes[i0, j0]], dots[0, indexes[i1, j1]]),
                (dots[1, indexes[i0, j0]], dots[1, indexes[i1, j1]]),
                color)

    def plot_angle(ind1, ind0, ind2, name):
        v1 = dots[:, indexes[ind1]]
        v0 = dots[:, indexes[ind0]]
        v2 = dots[:, indexes[ind2]]
        theta1 = np.rad2deg(np.arctan2((v1-v0)[1], (v1-v0)[0]))
        theta2 = np.rad2deg(np.arctan2((v2-v0)[1], (v2-v0)[0]))
        arc = Arc(v0, arc_size, arc_size, theta1=theta1, theta2=theta2,
                  lw=1, zorder=-10, color=angle_color)
        ax.add_patch(arc)
        if 'alpha' in name:
            ax.text(v0[0]+arc_pad, v0[1]+arc_pad,
                    name, ha='left', va='bottom', color=angle_color)
        elif 'beta' in name:
            ax.text(v0[0]-arc_pad, v0[1] + arc_pad,
                    name, ha='right', va='bottom', color=angle_color)
        else:
            raise ValueError(f"Unknown angle type: {name}")

    plot_line(1, 1, 1, 2, 'r')
    plot_line(1, 1, 2, 1, 'r')
    plot_line(1, 2, 2, 2, 'b')
    plot_line(2, 1, 2, 2, 'b')
    plot_line(1, 1, 0, 1, '--b')
    plot_line(1, 1, 1, 0, '--r')
    plot_line(2, 1, 2, 0, '--b')
    plot_line(2, 1, 3, 1, '--b')
    plot_line(1, 2, 0, 2, '--r')
    plot_line(1, 2, 1, 3, '--r')
    plot_line(2, 2, 3, 2, '--r')
    plot_line(2, 2, 2, 3, '--b')

    def plot_alpha_beta(i0, j0, letter):
        plot_angle((i0, j0+1), (i0, j0), (i0+1, j0), rf'$ \alpha_{letter} $')
        plot_angle((i0+1, j0), (i0, j0), (i0, j0-1), rf'$ \beta_{letter} $')
    plot_alpha_beta(1, 1, 'A')
    plot_alpha_beta(1, 2, 'C')
    plot_alpha_beta(2, 1, 'B')
    plot_alpha_beta(2, 2, 'D')

    def plot_arrow(i0, j0, i1, j1, text, pad_x, pad_y):
        p0 = dots[:, indexes[i0, j0]]
        p1 = dots[:, indexes[i1, j1]]
        end = p0 + 0.3 * (p1-p0)
        arrow = FancyArrowPatch(p0, end, linestyle='-', arrowstyle='->', mutation_scale=30, lw=2.0, zorder=5)
        ax.add_patch(arrow)
        ax.text(end[0]+pad_x, end[1]+pad_y, text)

    plot_arrow(1, 1, 1, 2, r'$ t_2^A $', -0.10, -0.15)
    plot_arrow(1, 1, 0, 1, r'$ t_1^A $', -0.15, 0.0)
    plot_arrow(1, 2, 1, 1, r'$ t_4^C $', -0.05, -0.2)
    plot_arrow(1, 2, 2, 2, r'$ t_3^C $', +0.05, 0.0)
    plot_arrow(2, 2, 1, 2, r'$ t_1^D $', -0.15, 0.0)
    plot_arrow(2, 1, 1, 1, r'$ t_1^B $', -0.15, 0.0)
    plot_arrow(2, 1, 2, 2, r'$ t_2^B $', -0.0, 0.04)
    plot_arrow(2, 2, 2, 1, r'$ t_4^D $', -0.05, 0.05)
    plot_arrow(1, 1, 2, 1, r'$ t_3^A $', 0.03, 0.05)

    ax.set_xlim(0.0, 2.4)
    ax.set_ylim(0.8, 3.0)

    ax.set_aspect('equal')
    ax.set_axis_off()
    # ax.grid()
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_PATH, 'panel-compatibility-notation.pdf'))

    plt.show()


def plot_panel_illustration_old():
    cos, sin, tan = np.cos, np.sin, np.tan
    pi = np.pi

    def na(*args): return np.array(args)

    def vec(angle): return na(cos(angle), sin(angle))

    theta = 1.1
    alphaA = theta + 0.2
    betaA = theta + 0.01
    alphaB = theta - 0.2
    betaB = theta + 0.3
    alphaC = theta - 0.1
    betaC = theta - 0.2
    alphaD = theta
    betaD = theta + 0.1

    a0 = 0.15
    dA = na(0, 0)
    aC = a0
    dC = dA + 1.0 * vec(aC)
    aD = aC + betaC
    dD = dC + 1.3 * vec(aD)
    aB = a0 + alphaA
    dB = dA + 1.15 * vec(aB)

    outL = 0.5
    aA1 = a0 + alphaA + betaA
    dA1 = dA + outL * vec(aA1)
    aA2 = a0 - (pi - betaA)
    dA2 = dA + outL * vec(aA2)
    aB1 = aB + alphaB
    dB1 = dB + outL * vec(aB1)
    aB2 = aB1 + betaB
    dB2 = dB + outL * vec(aB1)

    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)

    def plot_line(a, b, color):
        return ax.plot((a[0], b[0]), (a[1], b[1]), color)[0]

    plot_line(dA, dC, 'r')
    plot_line(dC, dD, 'r')
    plot_line(dA, dB, 'r')
    plot_line(dB, dD, 'r')
    plot_line(dA, dA1, '--r')
    plot_line(dA, dA2, '--r')

    plt.show()


def main():
    plot_panel_illustration()


if __name__ == '__main__':
    main()
