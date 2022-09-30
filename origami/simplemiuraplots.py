import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import origami.quadranglearray
from formsanalyzer import SimpleMiuraOriFormAnalyzer
from miuraori import SimpleMiuraOri
from origami.utils import plotutils, logutils


def create_cylinder():
    ls_y = create_circular_ls()
    origami = SimpleMiuraOri(np.ones(30) * 0.1, ls_y)
    return origami


def create_cylinder_large_sector():
    ls_y = create_circular_ls(zigzag_angle=np.pi / 10, sector=4 / 5 * np.pi)
    origami = SimpleMiuraOri(np.ones(10) * 0.05, ls_y, np.pi / 10)
    return origami


def create_sphere_bad():
    ls = create_circular_ls()
    # ls_x = np.append(ls_y[1:], ls_y[0])
    ls_y = np.concatenate((ls, ls * 1.5, ls * 2, ls * 1.5, ls))
    origami = SimpleMiuraOri(np.array([0.1, 1, 0.1, 1, 0.1, 1]), ls_y)
    return origami


def create_cylinder_with_lag():
    ls_y = create_circular_ls()
    # ls_x = np.append(ls_y[1:], ls_y[0])
    origami = SimpleMiuraOri(np.ones(20) * 0.1, np.append(np.ones(20) * 0.1, ls_y))
    return origami


def create_planar():
    # origami = SimpleMiuraOri(np.ones(20) * 0.1, np.ones(20) * 0.1)
    origami = SimpleMiuraOri(np.ones(6) * 0.1, np.ones(6) * 0.1)
    # origami = SimpleMiuraOri(np.ones(6) * 0.1, np.ones(6) * 0.1, angle=0.01)
    return origami


def create_saddle():
    ls = create_circular_ls()
    origami = SimpleMiuraOri(ls, ls)
    return origami


def create_curved_cylinder():
    ls_y = create_circular_ls(zigzag_angle=np.pi / 10, sector=4 / 5 * np.pi)
    ls_x = ls_y
    origami = SimpleMiuraOri(ls_x, ls_y, np.pi / 10)
    return origami


def create_positive_K():
    origami = SimpleMiuraOri([14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7],
                             [14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7],
                             angle=4 / 5 * np.pi)
    return origami


def create_negative_K():
    origami = SimpleMiuraOri([14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7],
                             [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14],
                             angle=4 / 5 * np.pi)
    return origami


def create_changing_K():
    ls = [14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7] + \
         [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14]
    origami = SimpleMiuraOri([14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7],
                             ls,
                             angle=4 / 5 * np.pi)
    return origami


def create_circular_ls(num_of_angles=20, zigzag_angle=np.pi / 8, sector=np.pi / 2):
    tan = np.tan(zigzag_angle)

    angles = np.linspace(np.pi + sector / 2, 2 * np.pi - sector / 2, num_of_angles)
    xs = np.cos(angles)
    ys = np.sin(angles)

    ls = np.zeros((len(angles) - 1) * 2)
    for i in range(len(angles) - 1):
        a_x = xs[i]
        a_y = ys[i]
        b_x = xs[i + 1]
        b_y = ys[i + 1]

        middle_x = 1 / (2 * tan) * (b_y - a_y + (a_x + b_x) * tan)
        middle_y = 1 / 2 * (b_y + a_y + (b_x - a_x) * tan)

        d1 = np.sqrt((a_x - middle_x) ** 2 + (a_y - middle_y) ** 2)
        d2 = np.sqrt((b_x - middle_x) ** 2 + (b_y - middle_y) ** 2)
        ls[2 * i] = d1
        ls[2 * i + 1] = d2

    return ls


def create_test_origami():
    # origami = SimpleMiuraOri([1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], angle=np.pi / 10)
    origami = SimpleMiuraOri([1, 1, 1, 1, 1], [1, 1], angle=np.pi / 4)
    # origami = SimpleMiuraOri([1, 1, 1], [1], angle=np.pi / 4)
    # origami = SimpleMiuraOri(np.array([1, 1, 1]), np.array([1, 1, 1, 1, 1]), angle=np.pi / 4)
    # origami = SimpleMiuraOri([1, 1], [1, 1, 1, 1], angle=np.pi / 4)
    # origami = SimpleMiuraOri([1, 2, 3, 1], [1, 3, 1])
    # origami = SimpleMiuraOri([1, 1, 1, 1], [1])
    # origami = SimpleMiuraOri([1], [1, 1, 1, 1])
    # origami = SimpleMiuraOri([1], [1, 1])
    # origami = SimpleMiuraOri([1, 1, 1, 1], [1])
    # origami = SimpleMiuraOri([1, 1], [1, 1])
    # origami.set_omega(np.pi/40)
    # origami.set_omega(np.pi / 4)
    logutils.enable_logger()

    return origami


def plot_origami():
    # origami = SimpleMiuraOri([1, 3, 1, 3, 1, 1, 3, 1, 3], [1, 2, 1, 2, 1, 2], angle=np.pi / 10)
    # origami = SimpleMiuraOri([1, 3, 1, 3, 1, 1, 3, 1, 3], [1, 1, 1, 1, 1, 1], angle=np.pi / 10)
    # origami = SimpleMiuraOri([1, 3, 1, 1], [1, 2, 1, 1], angle=np.pi / 4)
    # origami = SimpleMiuraOri([1, 2, 1, 1] * 6, [1, 2, 1, 1] * 3 + [2, 1, 1, 1] * 3, angle=np.pi / 4)
    # origami = create_positive_K()
    # origami = create_changing_K()
    # origami = create_cylinder()
    # origami = create_saddle()
    # origami = create_curved_cylinder()
    # origami = create_cylinder_with_lag()
    # origami = create_sphere_bad()
    # origami = create_test_origami()
    ori = create_planar()
    # origami = create_cylinder_large_sector()

    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ori.plot(ax)

    ori.set_omega(1)
    valid, reason = origami.quadranglearray.is_valid(ori.initial_dots, ori.dots, ori.indexes)
    if not valid:
        raise RuntimeError(f'Not a valid folded configuration. Reason: {reason}')

    # noinspection PyUnusedLocal
    slider = add_slider(ax, ori, should_plot_normals=False)

    plt.show()


def analyze_forms():
    form_analyzer = SimpleMiuraOriFormAnalyzer([1, 5], [1, 4, 10, 2])

    form_analyzer.set_omega(2)
    form_analyzer.compare_to_theory()

    fig = plt.figure()

    ax: matplotlib.axes.Axes = fig.add_subplot(111, projection='3d')
    form_analyzer.plot(ax)
    # noinspection PyUnusedLocal
    slider = add_slider(ax, form_analyzer)

    form_analyzer.compare_to_theory()

    plt.show()


def add_slider(ax, origami, should_plot_normals=False):
    lim = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

    init_omega = 1

    # Make a horizontal slider to control the frequency.
    omega_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    omega_slider = matplotlib.widgets.Slider(
        ax=omega_slider_ax,
        label='Omega',
        valmin=0,
        valmax=np.pi,
        valinit=init_omega,
    )

    origami.plot(ax)

    def update_omega(omega):
        ax.clear()
        origami.set_omega(omega)
        if should_plot_normals:
            origami.plot_normals(ax)
            origami.plot(ax, alpha=0.3)
        else:
            origami.plot(ax, alpha=1)
        # ax.set_autoscale_on(False)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim / 2, lim / 2)

        plotutils.set_3D_labels(ax)

    omega_slider.on_changed(update_omega)
    # update_omega(np.pi/2)
    update_omega(init_omega)
    return omega_slider


def main():
    # analyze_forms()
    plot_origami()


if __name__ == '__main__':
    main()
