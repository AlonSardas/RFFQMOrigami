import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arc
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D

from origami import origamiplots
from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import (create_angles_func_vertical_alternation,
                                       set_perturbations_by_func)
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles
from origami.origamiplots import plot_interactive
from origami.plotsandcalcs import articleillustrations
from origami.plotsandcalcs.alternating.utils import (create_F_from_list, create_MM_from_list,
                                                     create_perturbed_origami,
                                                     csc, sin)
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import linalgutils, plotutils

# FIGURES_PATH = os.path.join(
#     origami.plotsandcalcs.BASE_PATH,
#     'RFFQM', 'ContinuousMetric', 'AlternatingFigures', 'flat')
FIGURES_PATH = articleillustrations.FIGURES_PATH


def plot_simple_flat_old():
    # F = lambda x: -0.2 + 0.2 * (x >= 6) + 0.2 * (x >= 12)
    rows = 6
    cols = 6
    angle = 1

    L0 = 0.3
    C0 = 0.5

    def MM(y): return 0.0 * y

    def F(x): return -0.25 * (x >= 0)

    ori1 = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    def F(x): return 0.25 * (x >= 0)

    ori2 = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    ori1.dots.rotate_and_center()
    ori2.dots.rotate_and_center()

    fig: Figure = plt.figure(figsize=(10, 6))
    ax: Axes3D = fig.add_subplot(121, projection='3d', azim=-90, elev=90)
    ori1.dots.plot(ax)
    # plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1), zoom=1.7)
    ax.set_zlim(-1, 1)
    ax.set_title('F<0')

    ax: Axes3D = fig.add_subplot(122, projection='3d', azim=-90, elev=90)
    ori2.dots.plot(ax)
    ax.set_title('F>0')
    # ax.set_zlim(0, 0.01)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1), zoom=1.7)
    ax.set_zlim(-1, 1)
    # ax.autoscale(tight=True)
    # plotutils.set_axis_scaled(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'simple-flat-example.pdf'))
    fig.savefig(os.path.join(FIGURES_PATH, 'simple-flat-example.png'))

    plt.show()


def plot_F_M_pert_notation():
    mpl.rcParams['font.size'] = 14

    # F = lambda x: -0.2 + 0.2 * (x >= 6) + 0.2 * (x >= 12)
    rows = 6
    cols = 3
    angle = 1

    L0 = 0.3
    C0 = 0.5
    MM_list = np.append(0, np.cumsum(np.array([0.1, 0.05, -0.1])))
    MM = create_MM_from_list(MM_list)

    F_list = np.array([0.15, 0.2, -0.15, 0.2])
    F = create_F_from_list(F_list)
    ori1 = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)

    rot_angle = 0.15
    rot = linalgutils.create_XY_rotation_matrix(rot_angle)
    ori1.dots.dots = rot @ ori1.dots.dots
    dots = ori1.dots.dots
    indexes = ori1.dots.indexes
    fig, ax = origamiplots.plot_crease_pattern(ori1, initial_MVA=1)

    # plot_interactive(ori1)

    # Plot C_0 text
    for j in range(0, cols):
        center_of_line = 1 / 2 * (dots[:, indexes[0, j]] + dots[:, indexes[0, j + 1]])
        ax.text(center_of_line[0] - 0.00, center_of_line[1] - 0.015, '$C_0$', ha='center', va='top')

    # Plot L_0 text
    for i in range(0, rows, 2):
        center_of_line = 1 / 2 * (dots[:, indexes[i, 0]] + dots[:, indexes[i + 1, 0]])
        ax.text(center_of_line[0] - 0.01, center_of_line[1], '$L_0$', ha='right')

    for i in range(1, rows, 2):
        center_of_line = 1 / 2 * (dots[:, indexes[i, 0]] + dots[:, indexes[i + 1, 0]])
        ax.text(center_of_line[0], center_of_line[1], f'$L_0 + M({i // 2})$', ha='right')

    for i in range(0, rows):
        bottom_dot = dots[:, indexes[i, 1]]
        left_dot = dots[:, indexes[i, 0]]
        up_dot = dots[:, indexes[i + 1, 1]]
        right_dot = dots[:, indexes[i, 2]]

        if i % 2 == 0:
            left_style = {'linestyle': '--', 'color': 'C3'}
            right_style = {'linestyle': '-', 'color': 'C2'}
        else:
            left_style = {'linestyle': '-', 'color': 'C2'}
            right_style = {'linestyle': '--', 'color': 'C3'}

        v2 = left_dot - bottom_dot
        v1 = up_dot - bottom_dot
        theta1 = np.rad2deg(np.arctan2(v1[1], v1[0]))
        theta2 = np.rad2deg(np.arctan2(v2[1], v2[0]))
        arc = Arc(bottom_dot, 0.1, 0.1, theta1=theta1, theta2=theta2,
                  lw=2, **left_style, zorder=-10)
        ax.add_patch(arc)

        v2 = up_dot - bottom_dot
        v1 = right_dot - bottom_dot
        theta1 = np.rad2deg(np.arctan2(v1[1], v1[0]))
        theta2 = np.rad2deg(np.arctan2(v2[1], v2[0]))
        arc = Arc(bottom_dot, 0.1, 0.1, theta1=theta1, theta2=theta2,
                  lw=2, **right_style, zorder=-10)
        ax.add_patch(arc)

    pm = 1
    for i in range(0, 3):
        dot = dots[:, indexes[i, 1]]
        pm_text = '+' if pm == 1 else '-'
        color = 'C2' if pm == 1 else 'C3'
        # ax.text(dot[0]+0.1, dot[1]+0.05,
        #         '$ \delta_{'+str(i)+',0}=' + pm + 'F(0) $')
        angle_text = r'$ \pi-\vartheta' + pm_text + f'F(1/2) $'
        ax.text(dot[0] + 0.03, dot[1] - 0.06,
                angle_text, rotation=-30, ha='left', va='bottom', fontsize=10, color=color)
        pm *= -1
        pm_text = '+' if pm == 1 else '-'
        color = 'C2' if pm == 1 else 'C3'
        angle_text = r'$ \pi-\vartheta' + pm_text + f'F(1/2) $'
        ax.text(dot[0] - 0.03, dot[1] - 0.05,
                angle_text, ha='right', va='bottom', rotation=30, fontsize=10, color=color)
    # pm = 1
    # for i in range(0, 3):
    #     dot = dots[:, indexes[i, 2]]
    #     pm_text = '' if pm == 1 else '-'

    #     ax.text(dot[0]+0.01, dot[1]+0.07,
    #             '$ ' + pm_text + f'F(1) $', va='bottom', rotation=30)
    #     pm *= -1
    #     pm_text = '' if pm == 1 else '-'
    #     ax.text(dot[0] - 0.02, dot[1] + 0.05,
    #             '$ ' + pm_text + f'F(1) $', ha='right', va='bottom', rotation=-30)

    for i in range(0, rows):
        bottom_dot = dots[:, indexes[i, 2]]
        left_dot = dots[:, indexes[i, 1]]
        up_dot = dots[:, indexes[i + 1, 2]]
        right_dot = dots[:, indexes[i, 3]]

        # if i == 0:
        #     size = 0.5
        # else:
        #     size=0.2
        size = 0.2

        if i % 2 == 0:
            left_style = {'linestyle': ':', 'color': 'C0'}
            right_style = {'linestyle': (0, (3, 1, 1, 1, 1, 1)), 'color': 'C5'}
        else:
            left_style = {'linestyle': (0, (3, 1, 1, 1, 1, 1)), 'color': 'C5'}
            right_style = {'linestyle': ':', 'color': 'C0'}

        v2 = left_dot - bottom_dot
        v1 = up_dot - bottom_dot
        theta1 = np.rad2deg(np.arctan2(v1[1], v1[0]))
        theta2 = np.rad2deg(np.arctan2(v2[1], v2[0]))
        # print(theta1, theta2)
        arc = Arc(bottom_dot, size, size, theta1=theta1, theta2=theta2,
                  lw=2, **left_style, zorder=-10)
        ax.add_patch(arc)

        v2 = up_dot - bottom_dot
        v1 = right_dot - bottom_dot
        theta1 = np.rad2deg(np.arctan2(v1[1], v1[0]))
        theta2 = np.rad2deg(np.arctan2(v2[1], v2[0]))
        arc = Arc(bottom_dot, size, size, theta1=theta1, theta2=theta2,
                  lw=2, **right_style, zorder=-10)
        ax.add_patch(arc)

    dot = dots[:, indexes[0, 2]]
    ax.text(dot[0] + 0.06, dot[1] + 0.09,
            r'$ \vartheta+F(1) $', rotation=30, ha='left', va='bottom', fontsize=10, color='C5')
    ax.text(dot[0] - 0.06, dot[1] + 0.09,
            r'$ \vartheta-F(1) $', ha='right', va='bottom', rotation=-30, fontsize=10, color='C0')
    dot = dots[:, indexes[1, 2]]
    ax.text(dot[0] + 0.06, dot[1] + 0.09,
            r'$ \vartheta-F(1) $', rotation=30, ha='left', va='bottom', fontsize=10, color='C0')
    ax.text(dot[0] - 0.06, dot[1] + 0.09,
            r'$ \vartheta+F(1) $', ha='right', va='bottom', rotation=-30, fontsize=10, color='C5')

    # ax.set_axis_on()
    # ax.grid()

    fig.savefig(os.path.join(FIGURES_PATH, 'M-F-pert-notation.pdf'))

    plt.show()


def plot_F_pert_on_vertex():
    theta = 1
    origin = (0, 0)
    F = 0.4
    L = 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax: Axes = ax
    ax.plot([-L, 0, L], [np.tan(np.pi / 2 - theta) * L, 0, np.tan(np.pi / 2 - theta) * L], 'r')
    ax.plot([0, 0], [0, L / 2.4], '--', color='C7')
    ax.plot([0, -L * np.sin(F) / 2.1], [0, L * np.cos(F) / 2.1], '-', color='b')

    # print(np.rad2deg(theta))
    arc = Arc(origin, 0.4, 0.4, theta1=90 - np.rad2deg(theta), theta2=90, lw=2)
    ax.add_patch(arc)
    arc = Arc(origin, 0.5, 0.5, theta1=90, theta2=90 + np.rad2deg(theta), lw=2)
    ax.add_patch(arc)
    ax.text(origin[0] + 0.15, origin[1] + 0.2, r"$ \vartheta $", va='bottom', )

    arc = Arc(origin, .7, 0.7, theta1=90, theta2=90 + np.rad2deg(F), lw=2, linestyle='--', color="C9")
    ax.add_patch(arc)
    ax.text(origin[0] - 0.11, origin[1] + 0.37, r"$ F>0 $", va='bottom', color='C9')

    ax.set_xlim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.set_axis_off()
    # ax.grid()

    fig.savefig(os.path.join(FIGURES_PATH, 'F-pert-on-vertex.pdf'),
                bbox_inches=Bbox.from_extents(2, 0.5, 8, 3.2))

    plt.show()


def debug_lengths_perturbations():
    # F = lambda x: -0.02 * x
    # F = lambda x: -0.2 * np.sin(x / 40)
    def F(x): return -0.2 * np.sin(x ** 2 / 900)

    # F = lambda x: +0.00001*(x+1)**2

    def F1(x): return F(x)

    def F2(x): return -F(x)

    def FF(x): return F(x * 2)

    def dFF(x): return FF(x + 0.5) - FF(x - 0.5)

    def ddFF(x): return dFF(x + 0.5) - dFF(x - 0.5)

    def dddFF(x): return ddFF(x + 0.5) - ddFF(x - 0.5)

    angle = 1

    rows = 70
    cols = 100
    L0 = 0.2
    ls = np.ones(rows) * L0
    cs = np.ones(cols) * 1
    ls[1::2] += 0.2 * np.sin((np.arange(len(ls) // 2) - 13) / 5)
    # cs += -0.5+0.1*np.sin(np.arange(len(cs))/20)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom, 'delta+eta')

    print("pert test", pert_func(0, 0))

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    calced_cs, calced_ls = marching.get_cs(), marching.get_ls()

    xs = np.linspace(0, cols / 2, 200)
    cot_angle = 1 / np.tan(angle)
    csc_angle = 1 / np.sin(angle)
    fig, ax = plt.subplots()
    ax.plot(calced_ls[20, ::2], 'x', label='ls')
    ax.plot(xs,
            L0 * (1 + 1 / 4 * cot_angle * (dFF(xs) - dFF(0)) + 1 / (2 * np.sin(angle) ** 2) * (FF(xs) - FF(0)) ** 2),
            '--', label='Expansion 2nd order')

    # fig.savefig(os.path.join(FIGURES_PATH, 'ls-vs-xs.png'))

    fig, ax = plt.subplots()
    x = 20
    ax.plot(calced_cs[::2, 2 * x], 'x', label=f'cs[i,{x}] vs i')
    ls_diff_integral = np.append(0, -np.cumsum(ls[0::2]) + np.cumsum(ls[1::2]))
    # ls_diff_integral = -np.cumsum(calced_ls[0::2, 2*x]) + np.cumsum(calced_ls[1::2,2*x])
    # ls_diff_integral -= ls_diff_integral[0]
    # ls_integral = np.cumsum(calced_ls[0::2, 2*x]) + np.cumsum(calced_ls[1::2,2*x])
    ls_integral = np.append(0, np.cumsum(ls[0::2]) + np.cumsum(ls[1::2]))

    ls_sum1 = np.append(0, np.cumsum(ls[0::2]))
    ms_sum1 = np.append(0, np.cumsum(ls[1::2]))

    ls_sum2 = np.append(0, np.cumsum(calced_ls[0::2, 2 * x]))
    ms_sum2 = np.append(0, np.cumsum(calced_ls[1::2, 2 * x]))

    # ls_integral -= ls_integral[0]
    print(ls_diff_integral)
    print(ls_integral)
    ax.plot(cs[x * 2] + 1 / 2 * csc_angle * (ls_diff_integral * dFF(x)), '--', label='approx1')
    ax.plot(cs[x * 2] + 1 / 2 * csc_angle * ls_diff_integral * dFF(x)
            + cot_angle * csc_angle * (1 / 2 * (FF(x) * dFF(x))) * ls_integral, '--', label='approx2')
    ax.plot(cs[x * 2] + csc_angle * ls_diff_integral * (1 / 2 * dFF(x) + 1 / 8 * ddFF(x))
            + cot_angle * csc_angle * (1 / 2 * FF(x) * dFF(x)) * ls_integral, '--', label='approx3')

    ax.plot(cs[x * 2] + 1 / 2 * csc_angle * ls_diff_integral * dFF(x)
            + cot_angle * csc_angle * (
                    1 / 2 * FF(x) * dFF(x) + 1 / 4 * dFF(x) ** 2 + 1 / 8 * FF(x) * ddFF(x)) * ls_integral, '--',
            label='approx4')
    ax.plot(cs[x * 2] + csc_angle * ls_diff_integral * (
            1 / 2 * dFF(x) + 1 / 8 * ddFF(x) +
            1 / 4 * FF(x) ** 2 * dFF(x) + 1 / 2 * cot_angle ** 2 * FF(x) ** 2 * dFF(x) +
            1 / 48 * dddFF(x))
            + cot_angle * csc_angle * (
                    1 / 2 * FF(x) * dFF(x) + 1 / 4 * dFF(x) ** 2 + 1 / 8 * FF(x) * ddFF(x)) * ls_integral, '--',
            label='approx5')
    exact1 = cs[x * 2] + csc(angle + FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ls_sum1 - \
             csc(angle - FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ms_sum1
    exact2 = cs[x * 2] + csc(angle + FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ls_sum2 - \
             csc(angle - FF(x + 1 / 2)) * sin(FF(x) - FF(x + 1 / 2)) * ms_sum2
    ax.plot(exact1, label="exact1")
    ax.plot(exact2, '-+', label="exact2")
    ax.legend()

    # fig, ax = plt.subplots()
    # ax.plot(marching.alphas[:, 20])

    plt.show()

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)

    ori.set_gamma(2.9)

    plot_interactive(ori)


def test_jump_in_F():
    rows = 10
    cols = 40
    angle = np.pi / 2 - 0.1
    W0 = 2.0

    def F(x): return 0.1 * (x >= 10) + 0.1 * (x >= 20)

    def MM(y): return 0 * y

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    fig, _ = plot_flat_quadrangles(ori.dots)

    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    plot_interactive(ori)


def create_MARS_Barreto_using_alternating_angles():
    """
    We just show that MARS-Barreto is a special case of our alternating angles
    """
    angle_base = 0.65 * np.pi
    angle = angle_base

    rows, cols = 8, 8
    L0, C0 = 1, 1

    def F(x): return np.pi / 2 - angle_base

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, None)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=0, elev=90)
    ori.dots.rotate_and_center()
    ori.dots.plot(ax)
    plotutils.set_axis_scaled(ax)
    ax.set_axis_off()
    #     ax.set_box_aspect(None, zoom=1.5)
    # ax.set_title(r'F=$ \frac{\pi}{2}-\vartheta $')
    LIM = 2.8
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(-LIM, LIM)
    fig.savefig(os.path.join(FIGURES_PATH, 'MARS-barreto-by-alternating.pdf'))
    fig.savefig(os.path.join(FIGURES_PATH, 'MARS-barreto-by-alternating.png'))

    plot_interactive(ori)


def main():
    # create_MARS_Barreto_using_alternating_angles()
    plot_F_M_pert_notation()
    # plot_F_pert_on_vertex()


if __name__ == '__main__':
    main()
