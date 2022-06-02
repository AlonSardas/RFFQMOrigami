import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np

from miuraori import SimpleMiuraOri


def create_cylinder():
    ls_y = create_circular_ls()
    print(ls_y)
    origami = SimpleMiuraOri(np.ones(30) * 0.1, ls_y)
    return origami


def create_sphere():
    ls_y = create_circular_ls()
    # ls_x = np.append(ls_y[1:], ls_y[0])
    origami = SimpleMiuraOri(np.ones(20) * 0.1, ls_y)
    return origami


def create_sphere_with_lag():
    ls_y = create_circular_ls()
    # ls_x = np.append(ls_y[1:], ls_y[0])
    origami = SimpleMiuraOri(np.ones(20) * 0.1, np.append(np.ones(20)*0.1, ls_y))
    return origami


def create_saddle():
    ls = create_circular_ls()
    origami = SimpleMiuraOri(ls, ls)
    return origami


def create_circular_ls(num_of_angles=20):
    base_angle = np.pi / 8
    tan = np.tan(base_angle)

    angles = np.linspace(np.pi + np.pi / 4, np.pi + 3 * np.pi / 4, num_of_angles)
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


def main():
    # TODO: the angle parameter doesn't seem to work...

    # origami = SimpleMiuraOri([1, 2, 3, 1, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1, 1], angle=np.pi / 10)
    # origami = SimpleMiuraOri([1, 2, 3, 1], [1, 3, 1])
    # origami = SimpleMiuraOri([1, 1, 1, 1], [1])
    # origami = SimpleMiuraOri([1], [1, 1, 1, 1])
    # origami = SimpleMiuraOri([1], [1, 1])
    # origami = SimpleMiuraOri([1, 1, 1, 1], [1])
    # origami = SimpleMiuraOri([1, 1], [1, 1])
    # origami.set_omega(np.pi/40)
    # origami.set_omega(np.pi / 4)


    # origami = create_cylinder()
    # origami = create_saddle()
    origami = create_sphere_with_lag()
    # origami = create_sphere()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    origami.plot(ax)

    lim = np.max([sum(origami.ls_x), sum(origami.ls_y)])

    # Make a horizontal slider to control the frequency.
    omega_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    omega_slider = matplotlib.widgets.Slider(
        ax=omega_slider_ax,
        label='Omega',
        valmin=0,
        valmax=np.pi,
        valinit=0,
    )

    def update_omega(omega):
        ax.clear()
        origami.set_omega(omega)
        origami.plot(ax)
        # ax.set_autoscale_on(False)
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        ax.set_zlim([-lim / 2, lim / 2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    omega_slider.on_changed(update_omega)
    # update_omega(np.pi/2)
    update_omega(1)

    plt.show()


if __name__ == '__main__':
    main()
