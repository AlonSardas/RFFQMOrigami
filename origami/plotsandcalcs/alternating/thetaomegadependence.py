import fractions

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from origami.utils import plotutils


def plot_theta_omega_dependence():
    pi = np.pi
    sin, cos, tan, sqrt = np.sin, np.cos, np.tan, np.sqrt
    csc = lambda t: 1 / np.sin(t)
    sec = lambda t: 1 / np.cos(t)

    pad_theta = 0.01
    pad_omega = 0.1
    thetas = np.linspace(0.5, pi / 2 - pad_theta, 50)
    omegas = np.linspace(0, pi - pad_omega, 50)

    th, W = np.meshgrid(thetas, omegas)

    kx = csc(th) * tan(W / 2) * sec(W / 2) * sqrt(2) * sqrt(2 * csc(th) ** 2 - cos(W) - 1) / 4
    ky = tan(th) ** 2 * sin(W / 2) * sqrt(2) * sqrt(2 * csc(th) ** 2 - cos(W) - 1) / 8

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.plot_surface(th, W, kx)
    ax.plot_surface(th, W, ky)

    ax.set_xlabel(r'$\vartheta$')
    ax.set_ylabel(r'$\omega$')

    ax.set_xlim(0, pi / 2)
    ax.set_ylim(0, pi)
    plotutils.set_pi_ticks(ax, 'x', (0, fractions.Fraction(1, 2)), 2)
    plotutils.set_pi_ticks(ax, 'y', (0, 1), 4)
    # ax.set_zscale('log')

    plt.show()


def main():
    plot_theta_omega_dependence()


if __name__ == '__main__':
    main()
