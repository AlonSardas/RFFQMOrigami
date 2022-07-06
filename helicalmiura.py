"""
Plots helical origami based on the paper:
https://doi.org/10.1103/PhysRevE.101.033002

"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import linalgutils
from linalgutils import calc_angle

PI = np.pi
cos = np.cos
sin = np.sin
norm = np.linalg.norm
inv = np.linalg.inv


def create_flat_parallelogram(l, eta, lamda):
    x_1 = np.array([0, 0, 0])
    x_2 = np.array([1, 0, 0])
    x_4 = l * (np.array([1, 0, 0]) * cos(eta) + np.array([0, 1, 0]) * sin(eta))
    x_3 = x_2 + x_4 - x_1

    # Based on eq. 1,2 in paper
    if l > 1:
        f = 1 / 2 - np.sqrt(1 / l ** 2 * (lamda - 1 / 2) ** 2 + 1 / l ** 2 * (l ** 2 - 1) / 4)
        x_0 = x_1 + lamda * (x_2 - x_1) + f * (x_4 - x_1)
    elif l < 1:
        g = 1 / 2 - np.sqrt(l ** 2 * (lamda - 1 / 2) ** 2 + (1 - l ** 2) / 4)
        x_0 = x_1 + lamda * (x_4 - x_1) + g * (x_2 - x_1)
    else:
        raise RuntimeError("The case where l=0 is not implemented")

    dots = np.array([x_0, x_1, x_2, x_3, x_4])
    dots = dots.transpose()

    return dots


def fold_parallelogram(dots, omega, sigma) -> np.ndarray:
    x_0, x_1, x_2, x_3, x_4 = dots.transpose()

    alpha = linalgutils.calc_angle(x_1 - x_0, x_2 - x_0)
    beta = linalgutils.calc_angle(x_2 - x_0, x_3 - x_0)

    if np.isclose(alpha + beta, PI / 2):
        raise RuntimeError('alpha+beta is PI/2, this is not supported here')

    gamma_4 = omega
    sign = np.sign((cos(alpha) - sigma * cos(beta)) * omega)
    num = (sigma - cos(alpha) * cos(beta)) * cos(omega) + sin(alpha) * sin(beta)
    denom = (sigma - cos(alpha) * cos(beta)) + sin(alpha) * sin(beta) * cos(omega)
    gamma_3 = sign * np.arccos(num / denom)

    gamma_2 = sigma * omega
    gamma_1 = -sigma * gamma_3

    # print(-calc_angle(x_1 - x_0, [-1, 0, 0]))

    R_xy = linalgutils.create_XY_rotation_matrix(-calc_angle(x_1 - x_0, [-1, 0, 0]))
    R_yz = linalgutils.create_YZ_rotation_matrix(gamma_1)
    # y_translation = np.array([0, x_0[1], 0])
    x_4 = R_xy.transpose() @ R_yz @ R_xy @ (x_4 - x_0) + x_0

    R_xy = linalgutils.create_XY_rotation_matrix(calc_angle(x_2 - x_0, [1, 0, 0]))
    R_yz = linalgutils.create_YZ_rotation_matrix(gamma_2)
    x_3 = R_xy.transpose() @ R_yz @ R_xy @ (x_3 - x_0) + x_0

    dots = np.array([x_0, x_1, x_2, x_3, x_4])
    dots = dots.transpose()
    return dots


def create_single_parallelogram(l, eta, lamda, omega, sigma):
    dots = create_flat_parallelogram(l, eta, lamda)
    return fold_parallelogram(dots, omega, sigma)


def plot_single_parallelogram():
    l = 1.2
    eta = 7 / 8 * PI / 2
    lamda = 0.4
    omega = -np.pi / 2
    sigma = -1

    dots = create_single_parallelogram(l, eta, lamda, omega, sigma)

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    triangles = [[0, 1, 2], [0, 1, 4], [0, 3, 4], [0, 2, 3]]
    ax.plot_trisurf(dots[0, :],
                    dots[1, :],
                    dots[2, :], triangles=triangles)

    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.5)
    ax.set_zlim(0, 1.5)

    plt.show()


def plot_helical_miura():
    l = 2
    eta = 7 / 8 * PI / 2
    lamda = 0.4
    omega = -np.pi / 5
    sigma = -1

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    dots = create_single_parallelogram(l, eta, lamda, omega, sigma)

    g_1, g_2 = _create_isometries(dots, phi=1.0)

    p_max = 9
    q_max = 3

    for i in range(p_max):
        d_copy = dots.copy()
        for j in range(q_max):
            triangles = [[0, 1, 2], [0, 1, 4], [0, 3, 4], [0, 2, 3]]
            ax.plot_trisurf(d_copy[0, :],
                            d_copy[1, :],
                            d_copy[2, :], triangles=triangles)
            d_copy = g_2(d_copy)

        dots = g_1(dots)


    max_lim = max(ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1])
    ax.set_xlim(-max_lim, max_lim)
    ax.set_ylim(-max_lim, max_lim)
    ax.set_zlim(-max_lim, max_lim)
    # ax.set_aspect('equal', adjustable='box')

    plt.show()


def _test_folded_parallelogram():
    l = 1.2
    eta = 7 / 8 * PI / 2
    lamda = 0.4
    omega = -np.pi / 7
    sigma = -1

    dots = create_flat_parallelogram(l, eta, lamda)
    x_0, x_1, x_2, x_3, x_4 = dots.transpose()

    angle1 = linalgutils.calc_angle(x_1 - x_0, x_2 - x_0)
    angle2 = linalgutils.calc_angle(x_3 - x_0, x_4 - x_0)
    assert np.isclose(angle1 + angle2, np.pi), "Kawasaki condition is not satisfied"

    y_0, y_1, y_2, y_3, y_4 = fold_parallelogram(dots.copy(), omega, sigma).transpose()

    assert np.isclose(norm(y_2 - y_1), norm(x_2 - x_1))
    assert np.isclose(norm(y_3 - y_2), norm(x_3 - x_2))
    assert np.isclose(norm(y_1 - y_4), norm(x_1 - x_4))
    assert np.isclose(norm(y_3 - y_4), norm(x_3 - x_4))
    print('All tests passed!')


def _create_isometries(dots, phi=1.5):
    # Based on the solution of the compatibility equation,
    # see eq. 13

    # Note: phi seems like another DOF! I think that it can have discrete values
    # if we want to ensure that the parallelograms will stick together in the cylindrical form

    y_0, y_1, y_2, y_3, y_4 = dots.transpose()
    u_a = y_3 - y_4
    u_b = y_2 - y_1
    v_a = y_1 - y_4
    v_b = y_2 - y_3

    I = np.identity(3)

    # Define right-handed orthonormal basis
    f_1 = (u_a + u_b) / norm(u_a + u_b)
    f_2 = np.cross(u_a, u_b) / norm(np.cross(u_a, u_b))
    f_3 = (u_a - u_b) / norm(u_a - u_b)

    assert np.isclose(f_1.dot(f_2), 0), 'f_1, f_2 are not orthogonal'

    e: np.ndarray = cos(phi) * f_1 + sin(phi) * f_2

    assert np.isclose(norm(e), 1), 'e is not normalized'

    e_outer = np.outer(e, e)
    P_e = I - e_outer

    theta_1 = np.sign(e.dot(np.cross(u_a, u_b))) * np.arccos((u_a.dot(P_e @ u_b)) / (norm(P_e @ u_a) ** 2))
    theta_2 = np.sign(e.dot(np.cross(v_a, v_b))) * np.arccos((v_a.dot(P_e @ v_b)) / (norm(P_e @ v_a) ** 2))

    tau_1 = e.dot(v_a)
    tau_2 = e.dot(u_a)
    R_t1 = linalgutils.create_rotation_around_axis(e, theta_1)
    R_t2 = linalgutils.create_rotation_around_axis(e, theta_2)
    z = inv(I - R_t1 + e_outer) @ P_e @ (y_2 - R_t1 @ y_3)

    assert np.all(np.isclose(R_t1 @ e, e))

    # print(e_outer)

    g_1 = lambda v: R_t1 @ v + (tau_1 * e + (I - R_t1) @ z)[:, np.newaxis]
    g_2 = lambda v: R_t2 @ v + (tau_2 * e + (I - R_t2) @ z)[:, np.newaxis]

    return g_1, g_2


def main():
    # plot_single_parallelogram()
    plot_helical_miura()
    # _test_folded_parallelogram()


if __name__ == '__main__':
    main()
