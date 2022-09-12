import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from origami.marchingalgorithm import MarchingAlgorithm


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



def main():
    # create_miura_ori()
    plot_zigzag()


if __name__ == '__main__':
    main()
