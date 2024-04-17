import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import origami.plotsandcalcs
from origami import zigzagmiuraori
from origami.plotsandcalcs import zigzagplots
from origami.utils import linalgutils, plotutils

FIGURES_PATH = os.path.join(
    origami.plotsandcalcs.BASE_PATH, "RFFQM", "Figures", "zigzag-origami"
)


def record_points():
    x_pts = []
    y_pts = []

    fig, ax = pyplot.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    (line,) = ax.plot(x_pts, y_pts, marker="o")

    def onpick(event):
        m_x, m_y = event.x, event.y
        x, y = ax.transData.inverted().transform([m_x, m_y])
        x_pts.append(x)
        y_pts.append(y)
        line.set_xdata(x_pts)
        line.set_ydata(y_pts)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onpick)

    pyplot.show()

    np.set_printoptions(precision=2)
    print(repr(np.array(x_pts)))
    print(repr(np.array(y_pts)))


def build_origami():
    xs = np.array([-5.69, -5.69, -4.69, -4.34, -3.28, -2.94, -2.06, -2.2, -1.11,
                   -0.91, 0.38, -0.01, 0.94, 1.34, 1.86, 1.95, 2.42, 3.59,
                   4.08, 5.46, 6.19, 6.15, 5.45, 5.45, 4.96, 4.33, 3.82,
                   3.85, 3.42, 3.01, 3.32, 2.24, 1.18, -1.09, -3.18, -5.16,
                   -6.43, -7.61, -8.98, -9.75, -9.12, -8.35, -7.3, -6.21, -5.59])
    ys = np.array([-2.81, -8.47, -8.71, -3.19, -2.81, -8.33, -8.36, -2.73, -2.73,
                   -7.3, -7.95, -2.75, -2.94, -7.95, -7.82, -2.78, -2.75, -2.,
                   -0.32, -0.05, 0.93, 2.72, 4.83, 7.37, 4.96, 4.88, 6.91,
                   4.8, 4.56, 3.34, 1.55, 2.47, 3.26, 3.47, 4.04, 2.5,
                   1.66, 0.74, -0.83, -2.43, -3.38, -1.86, -0.83, -0.97, -2.46])

    # xs = xs[:20]
    # ys = ys[:20]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, '-')
    ax.plot(xs, ys, "*")
    fig.savefig(os.path.join(FIGURES_PATH, "cat-curve.png"))
    # plt.show()

    num_of_dots = len(xs)
    ls = np.zeros(num_of_dots - 1)
    for i in range(num_of_dots - 1):
        ls[i] = np.sqrt((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2)
    print(ls)

    angles = np.zeros(num_of_dots - 2)
    angles_signs = np.zeros(num_of_dots - 2)
    for i in range(num_of_dots - 2):
        x0 = np.array([xs[i + 1], ys[i + 1]])
        x1 = np.array([xs[i], ys[i]])
        x2 = np.array([xs[i + 2], ys[i + 2]])
        v1 = x1 - x0
        v2 = x2 - x0
        angles[i] = linalgutils.calc_angle(v1, v2)
        angles_signs[i] = np.sign(np.cross(v1, v2))

    print(angles)
    print(angles_signs)

    origami_angles = np.zeros(num_of_dots)
    # These are arbitrary not 0 values, just to avoid a 0 zigzag angle
    origami_angles[0] = 0.1
    origami_angles[-1] = 0.1

    # We will use omega close to pi, so theta vs omega is approximately linear
    omega = 3.1
    omega_sign = 1  # This doesn't really matter, not entirely sure why
    for i in range(num_of_dots - 2):
        angle = angles[i]
        sign = angles_signs[i] * omega_sign
        if sign == 1:
            alpha = np.pi / 2 - angle / 2
        elif sign == -1:
            alpha = np.pi / 2 + angle / 2
        else:
            raise RuntimeError(f"Unexpected value for sign: {sign}")
        origami_angles[i + 1] = alpha
        omega_sign *= -1

    rows = num_of_dots
    cols = 3
    dots = zigzagmiuraori.create_zigzag_dots(origami_angles, cols, ls, 0.02)
    ori = zigzagmiuraori.ZigzagMiuraOri(dots, rows, cols)
    zigzagplots.plot_flat_configuration(ori)
    valid, reason = ori.is_valid()
    if not valid:
        # raise RuntimeError(f'Not a valid folded configuration. Reason: {reason}')
        print(f"Not a valid folded configuration. Reason: {reason}")
    else:
        print("The origami is valid")
    # plt.show()

    ori.set_omega(omega)
    valid, reason = ori.is_valid()
    if not valid:
        raise RuntimeError(
            f'Not a valid folded configuration. Reason: {reason}')
        # print(f"Not a valid folded configuration. Reason: {reason}")
    else:
        print("The origami is valid")
    # zigzagplots.plot_interactive(origami)

    fig: Figure = plt.figure(layout='constrained')
    ax: Axes3D = fig.add_subplot(111, projection="3d", elev=2, azim=173)
    quads = ori.get_quads()
    quads.plot(ax, edge_width=2)
    # ax.set_zlim(-2, 2)

    # ax.set_xlim(-0.0003, 0.0003)
    ax.set_xticks([-0.00025, 0.00025])
    ax.set_yticks([-5, 0, 5])
    ax.tick_params(axis='x', which='major', pad=28, labelrotation=25)
    # ax.xaxis.get_ticklabels().set_va("bottom")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")

    plotutils.save_fig_cropped(fig, os.path.join(FIGURES_PATH, 'origami-cat.pdf'),
                               1.17, 0.75, translate_x=-0.1, translate_y=-0.1)
    # fig.savefig(os.path.join(FIGURES_PATH, 'origami-cat.png'))

    plt.show()


def main():
    # record_points()
    build_origami()


if __name__ == "__main__":
    main()
