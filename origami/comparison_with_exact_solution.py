import numpy as np

FIGURES_PATH = '../../RFFQM/Figures'


def plot_smooth_surface():
    us = np.linspace(-1, 1, 200)
    vs = np.linspace(0, 2*np.pi, 200)

    Us, Vs = np.meshgrid(us, vs)

    xs = np.cos(Us) + Vs
    ys = np.sin(Vs)
    zs = np.sin(Us)

    fig = plt.figure()

    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=90, elev=-100)


def main():
    pass


if __name__ == '__main__':
    main()