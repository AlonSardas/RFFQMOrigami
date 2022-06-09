import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np

from miuraori import SimpleMiuraOri


def plot_simple_crease_pattern():
    origami = SimpleMiuraOri(np.ones(6), np.ones(6))
    fig = plt.figure()

    ax: matplotlib.axes.Axes = fig.add_subplot(111, projection='3d', azim=90, elev=-100)
    origami.plot(ax, should_rotate=False, should_center=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('../RFFQM/Figures/simple_pattern.png')

    plt.show()


def set_pi_ticks(ax, axis):
    if 'x' in axis:
        ax.set_xticks([0, np.pi/4, np.pi/2, 3/4*np.pi, np.pi])
        ax.set_xticklabels(['0', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$'])
    if 'y' in axis:
        ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 / 4 * np.pi, np.pi])
        ax.set_yticklabels(['0', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', r'$\pi$'])


def plot_gamma_vs_activation_angle():
    matplotlib.rc('font', size=22)

    fig, ax = plt.subplots()
    xs = np.linspace(0, np.pi, 200)
    ys = np.arccos((3 * np.cos(xs) - 1) / (3 - np.cos(xs)))

    ax.plot(xs, ys)

    set_pi_ticks(ax, 'xy')

    ax.set_xlabel(r'$ \omega $')
    ax.set_ylabel(r'$ \gamma\left(\omega;\beta=\pi/4\right) $')

    fig.savefig('../RFFQM/Figures/gamma_vs_activation_angle.png')

    plt.show()


def plot_phi_vs_activation_angle():
    matplotlib.rc('font', size=22)

    fig, ax = plt.subplots()
    ax: matplotlib.axes.Axes = ax
    xs = np.linspace(0, np.pi, 200)
    ys = 1 / 2 * np.arccos(-1 / 2 * np.cos(xs) + 1 / 2)

    ax.plot(xs, ys)
    ax.set_xlabel(r'$ \omega $')
    ax.set_ylabel(r'$ \phi\left(\omega;\beta=\pi/4\right) $')
    set_pi_ticks(ax, 'x')
    ax.set_yticks([0, np.pi / 8, np.pi / 4])
    ax.set_yticklabels(['0', r'$\frac{1}{8}\pi$', r'$\frac{1}{4}\pi$'])

    fig.savefig('../RFFQM/Figures/phi_vs_activation_angle.png')

    plt.show()


if __name__ == '__main__':
    plot_gamma_vs_activation_angle()
    plot_phi_vs_activation_angle()
