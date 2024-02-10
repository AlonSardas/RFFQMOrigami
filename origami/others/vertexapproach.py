"""
Testing the vertex approach as discussed in:

Zhou Xiang, Wang Hai and You Zhong 2015
Design of three-dimensional origami structures based on a vertex approachProc. R. Soc. A.4712015040720150407
http://doi.org/10.1098/rspa.2015.0407
"""
from matplotlib import pyplot as plt
import numpy as np

from origami.quadranglearray import QuadrangleArray
from origami.utils import plotutils


def plot_vertices_example():
    V_x = np.array([[0, 0, 0],
                    [1, 0, 10],
                    [2, 0, 17],
                    [3, 0, 26],
                    [4, 0, 30],
                    [5, 0, 20]]).transpose()
    V_y = np.zeros((3, 14))
    V_y[1, :] = np.linspace(0, 10, V_y.shape[1])
    Ry = 6
    V_y[2, :] = np.sqrt(Ry ** 2-(V_y[1, :]-Ry)**2)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(V_x[0, :], V_x[2, :], '.')
    axes[1].plot(V_y[1, :], V_y[2, :], '.')
    # plt.show()

    Vs = generate_vertices(V_x, V_y)
    quads = vertices_to_quad_array(Vs)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    quads.plot_with_wireframe(ax, alpha=0.4)
    plt.show()


def vertices_to_quad_array(Vs: np.ndarray):
    cols, rows, _ = Vs.shape
    xs = Vs[:, :, 0].transpose().flatten()
    ys = Vs[:, :, 1].transpose().flatten()
    zs = Vs[:, :, 2].transpose().flatten()
    return QuadrangleArray(np.array([xs, ys, zs]), rows, cols)


def calc_theta(V_y, j):
    """According to eq. 2.3"""
    vec_diff = V_y[:, j+1]-V_y[:, j]
    vec = vec_diff / np.linalg.norm(vec_diff)
    assert np.isclose(vec[0], 0)
    theta = np.arctan2(vec[2], vec[1])
    assert np.isclose(np.cos(theta), vec[1])
    assert np.isclose(np.sin(theta), vec[2])
    return theta


def generate_vertices(V_x: np.ndarray, V_y: np.ndarray):
    m = V_x.shape[1]
    n = V_y.shape[1] - 2

    Vs = np.zeros((m, n+1, 3))

    thetas = np.zeros(n+2)
    thetas[0] = calc_theta(V_y, 0)
    cos, sin = np.cos, np.sin
    for j in range(1, n+1):
        thetas[j] = calc_theta(V_y, j)
        # Eq. 2.2
        A_j = np.array([[1, 0, 0],
                        [0, 0, (-1)**j*(cos(thetas[j-1])+cos(thetas[j])) /
                         (sin(thetas[j-1]-thetas[j]))],
                        [0, 0, (-1)**j*(sin(thetas[j-1])+sin(thetas[j])) /
                         (sin(thetas[j-1]-thetas[j]))]])
        # Eq. 2.1
        Vs[:, j, :] = (V_y[:, [j]] + A_j @ V_x).transpose()
    Vs = Vs[:, 1:, :]  # Trim the first zero vector - this fixes the expected shape
    return Vs


def test_miura_ori():
    """
    Design Miura-Ori by the vertices method, according to example 3.a
    """
    n, m = 4, 6
    T_x, h_x = 1, 0.2
    T_y, h_y = 3, 0.5
    i = np.arange(1, m+1)
    j = np.arange(0, n+2)
    V_x = np.array([(i-1)/2*T_x, 0*i, (1+(-1)**i)/2*h_x])
    V_y = np.array([0*j, (j-1)/2*T_y, (1+(-1)**j)/2*h_y])

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(V_x[0, :], V_x[2, :], '.')
    axes[1].plot(V_y[1, :], V_y[2, :], '.')
    # plt.show()

    Vs = generate_vertices(V_x, V_y)
    print(Vs[:, [0], 0])
    quads = vertices_to_quad_array(Vs)
    print(quads.dots[0, quads.indexes[0, :]])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    quads.plot_with_wireframe(ax, alpha=0.4)
    dots = quads.dots
    ax.scatter(
        dots[0, :],
        dots[1, :],
        dots[2, :],
        alpha=0.4)
    plotutils.set_axis_scaled(ax)
    plt.show()


def test_cylinder_design():
    """
    Following the example from sec. 3.b
    """
    n, m = 10, 16
    T_x, h_x = 1, 0.1
    r1, r2 = 3, 3.5
    i = np.arange(1, m+1)
    j = np.arange(0, n+2)
    d_j = (1-(-1)**j)/2*r1+(1+(-1)**j)/2*r2  # This is just alternating between r1, r2
    beta = 0.2

    V_x = np.array([(i-1)/2*T_x, 0*i, (1+(-1)**i)/2*h_x])
    # V_x [2, 4] = 1.2
    # V_x [2, 8] = -1.0
    V_y = np.array([0*j, d_j * np.sin((j-1)*beta), d_j * np.cos((j-1)*beta)])

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(V_x[0, :], V_x[2, :], '.')
    axes[1].plot(V_y[1, :], V_y[2, :], '.')

    # plt.show()

    Vs = generate_vertices(V_x, V_y)
    quads = vertices_to_quad_array(Vs)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    quads.plot_with_wireframe(ax, alpha=0.4)
    dots = quads.dots
    ax.scatter(dots[0, :], dots[1, :],        dots[2, :],        alpha=0.4)
    plotutils.set_axis_scaled(ax)
    plt.show()


def plot_with_intersection():
    n, m = 10, 16
    T_x, h_x = 1, 0.1
    r1, r2 = 3, 3.5
    i = np.arange(1, m+1)
    j = np.arange(0, n+2)
    d_j = (1-(-1)**j)/2*r1+(1+(-1)**j)/2*r2  # This is just alternating between r1, r2
    beta = 0.2

    V_x = np.array([(i-1)/2*T_x, 0*i, (1+(-1)**i)/2*h_x])
    # V_x [2, 4] = 1.2
    # V_x [2, 8] = -1.0
    V_y = np.array([0*j, d_j * np.sin((j-1)*beta), d_j * np.cos((j-1)*beta)])
    V_y[2, 4] = 2.2

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(V_x[0, :], V_x[2, :], '.')
    axes[1].plot(V_y[1, :], V_y[2, :], '.')
    axes[1].plot(V_y[1, 4], V_y[2, 4], '*')

    # plt.show()

    Vs = generate_vertices(V_x, V_y)
    quads = vertices_to_quad_array(Vs)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    quads.plot_with_wireframe(ax, alpha=0.4)
    dots = quads.dots
    ax.scatter(dots[0, :], dots[1, :],        dots[2, :],        alpha=0.4)
    plotutils.set_axis_scaled(ax)
    plt.show()


def main():
    # plot_vertices_example()
    # test_miura_ori()
    test_cylinder_design()


if __name__ == '__main__':
    main()
