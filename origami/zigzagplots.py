import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from origami import miuraori, simplemiuraplots
from origami.utils import plotutils, logutils
from origami.zigzagmiuraori import ZigzagMiuraOri


def create_zigzag(n, dx, y, angle) -> np.ndarray:
    dots = np.zeros((2, n))
    dots[0, :] = np.arange(n) * dx
    dots[1, :] = y
    dots[1, 1::2] += -dx / np.tan(angle)
    return dots


def create_basic_crease1():
    n = 3
    dx = 1
    dy = 2

    rows = 3
    angles = 0.2 * np.ones(rows) * np.pi
    rows = len(angles)

    dots = np.zeros((2, len(angles) * n))

    y = 0
    for i, angle in enumerate(angles):
        dots[:, i * n:(i + 1) * n] = create_zigzag(n, dx, y, angle)

        y += dy

    MVs = np.ones(rows - 2)
    MVs[::2] = -1
    MVs *= -1
    print(MVs)

    return dots, len(angles), n, MVs


def create_basic_crease2():
    n = 7
    dx = 1
    dy = 2

    angles = np.array([0.2, 0.3, 0.7, 0.6, 0.7]) * np.pi

    dots = np.zeros((2, len(angles) * n))

    y = 0
    for i, angle in enumerate(angles):
        dots[:, i * n:(i + 1) * n] = create_zigzag(n, dx, y, angle)

        y += dy

    # MVs = np.ones(len(angles) - 2)
    MVs = [1, 1, -1]

    return dots, len(angles), n, MVs


def plot():
    logutils.enable_logger()

    dots, rows, cols, MVs = create_basic_crease2()
    origami = ZigzagMiuraOri(dots, MVs, rows, cols, 1)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    origami.plot(ax)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    origami.set_omega(0)
    origami.plot(ax)

    origami.set_omega(1)
    valid, reason = miuraori.is_valid(origami.initial_dots, origami.dots, origami.indexes)
    if not valid:
        # raise RuntimeError(f'Not a valid folded configuration. Reason: {reason}')
        print(f'Not a valid folded configuration. Reason: {reason}')

    # plotutils.set_3D_labels(ax)

    # We need to assign the return value to variable slider for the slider object
    # stay alive and keep functioning
    slider = simplemiuraplots.add_slider(ax, origami, should_plot_normals=False)

    plt.show()


def main():
    plot()


if __name__ == '__main__':
    main()
