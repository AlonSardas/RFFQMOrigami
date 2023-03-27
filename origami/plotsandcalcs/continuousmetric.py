import itertools

import numpy as np
from sympy import *

from origami import marchingalgorithm
from origami.RFFQMOrigami import RFFQM
from origami.interactiveplot import plot_interactive
from origami.quadranglearray import QuadrangleArray
from origami.utils import sympyutils

delta_11, delta_12, eta_12, eta_21, delta_22, eta_22 = symbols(
    r"\delta_11, \delta_12, \eta_12, \eta_21, \delta_22, \eta_22")
delta_24, eta_24 = symbols(r"\delta_24, \eta_24")
all_perturbation_angles = (delta_11, delta_12, eta_12, eta_21, delta_22, eta_22, delta_24, eta_24)
c_11, d_11, l_11, l_12 = symbols(r"c_11, d_11, \ell_11, \ell_12")
cth = symbols(r"\vartheta")
omega, gamma = symbols(r"\omega, \gamma")
I = eye(3)
e_3 = Matrix([0, 0, 1])


def make_angles_vanish(expr: Expr) -> Expr:
    subs_dict = dict(zip(all_perturbation_angles, itertools.repeat(0)))
    return expr.subs(subs_dict)


def _create_rotation_around_axis(t: Matrix, angle) -> Matrix:
    t_perp = Matrix([-t[1], t[0], 0])  # A vector in XY plane that is perp to t
    P = I - _outer(t, t)
    W = _outer(e_3, t_perp) - _outer(t_perp, e_3)
    R = _outer(t, t) + cos(angle) * P + sin(angle) * W
    return R


def _outer(v1: Matrix, v2: Matrix) -> Matrix:
    return v1 * v2.transpose()


def calc_AE() -> Expr:
    t = Matrix([sin(eta_12 + delta_11), cos(eta_12 + delta_11), 0])
    R = _create_rotation_around_axis(t, omega)
    R = R.transpose()  # Just for a convention of the sign of omega

    print("---------- Exact R=")
    print(latex(R))

    print("----------- linearized R=")
    R = sympyutils.linearize_multivariable(R, all_perturbation_angles)
    R = simplify(R)
    print(latex(R))

    a_sum = (delta_11 + eta_12 + delta_12)

    AC_l = c_11 * Matrix([sin(cth) + cos(cth) * delta_11, cos(cth) - sin(cth) * delta_11, 0])
    CE_l = d_11 * Matrix([sin(cth) - cos(cth) * a_sum, -cos(cth) - sin(cth) * a_sum, 0])

    AE_l = AC_l + R * CE_l

    AE_l = sympyutils.linearize_multivariable(AE_l, (delta_11, delta_12, eta_12))

    print("-----------linearized AE=")
    print(latex(simplify(AE_l)))

    print("------------Unperturbed AE=")
    AE_unperturbed = make_angles_vanish(AE_l)
    print(latex(simplify(AE_unperturbed)))

    return AE_l


def gamma2(s, a, b, o):
    nom = (-s + cos(a) * cos(b)) * cos(o) + sin(a) * sin(b)
    deno = -s + cos(a) * cos(b) + sin(a) * sin(b) * cos(o)
    calculated_gamma = acos(nom / deno)
    return calculated_gamma


def calc_gamma():
    s = +1
    a = cth - delta_22
    b = pi - cth + eta_22
    o = omega

    calculated_gamma = gamma2(s, a, b, o)

    print(latex(calculated_gamma))
    print("Ignoring perturbation: ")
    print(latex(simplify(make_angles_vanish(calculated_gamma))))

    inside_l = sympyutils.linearize_multivariable(nom / deno, (delta_22, eta_22))
    print("Linearized inside")
    print(latex(sympify(expand(inside_l))))

    print()
    print()
    print("Calculating linearized gamma. It takes a few seconds...")

    gamma_l = sympyutils.linearize_multivariable(calculated_gamma, (delta_22, eta_22))

    print("Linearized gamma: ")
    print(latex(gamma_l))


def calc_AJ() -> Expr:
    t = -Matrix([cos(pi / 2 - cth - eta_21), sin(pi / 2 - cth - eta_21), 0])
    t_perp = -Matrix([-sin(pi / 2 - cth - eta_21), cos(pi / 2 - cth - eta_21), 0])
    P = I - _outer(t, t)
    W = _outer(e_3, t_perp) - _outer(t_perp, e_3)

    # I'm not completely sure what should be the sign of gamma
    gamma_sign = -1
    R = _outer(t, t) + cos(gamma_sign * gamma) * P + sin(gamma_sign * gamma) * W

    R_l = sympyutils.linearize_multivariable(R, (eta_21,))

    print()
    print()
    print("----------- linearized R=")
    print(latex(simplify(R_l)))

    a_diff = eta_21 - delta_12

    AB = l_11 * Matrix([0, 1, 0])
    BJ = l_12 * Matrix([sin(a_diff), cos(a_diff), 0])

    AJ = AB + R * BJ
    AJ_l = sympyutils.linearize_multivariable(AJ, all_perturbation_angles)

    print("----------- linearized AJ=")
    print(latex(simplify(AJ_l)))

    AJ_unperturbed = make_angles_vanish(AJ_l)
    print()
    print("AJ unperturbed")
    print(latex(simplify(AJ_unperturbed)))
    print("AJ^2 unperturbed")
    print(latex(simplify(AJ_unperturbed.dot(AJ_unperturbed))))

    return AJ_l


def calc_metric() -> Expr:
    AE_l = calc_AE()
    AJ_l = calc_AJ()

    metric = Matrix([[AE_l.dot(AE_l), AE_l.dot(AJ_l)], [AE_l.dot(AJ_l), AJ_l.dot(AJ_l)]])
    metric_l = sympyutils.linearize_multivariable(metric, all_perturbation_angles)

    print()
    print("linearized metric")
    print(latex(simplify(trigsimp(metric_l))))
    return metric_l


def calc_g12_unperturbed():
    AE_l = calc_AE()
    AJ_l = calc_AJ()

    g_12 = AE_l.dot(AJ_l)
    g_12_unperturbed = make_angles_vanish(g_12)

    print()
    print("g_12 unperturbed:")
    print(latex(simplify(trigsimp(g_12_unperturbed))))


def calc_unperturbed_aligned():
    AE_l = calc_AE()
    AJ_l = calc_AJ()

    AE_unperturbed = make_angles_vanish(AE_l)
    AJ_unperturbed = make_angles_vanish(AJ_l)

    print("After aligning the vectors")
    R = _create_rotation_around_axis(Matrix([0, 1, 0]), omega / 2)
    AE = R * AE_unperturbed
    AJ = R * AJ_unperturbed

    print(latex(simplify(AJ)))

    AJ_same_l = AJ.subs(l_12, l_11)

    YZ_angle = AJ_same_l.dot(Matrix([0, 1, 0])) / sqrt(AJ_same_l.dot(AJ_same_l))
    YZ_angle = simplify(YZ_angle)
    print(latex(YZ_angle))


def test_g12():
    angle = 3 / 4 * np.pi
    # ls = np.ones(5) * 2
    ls = [1, 1]
    cs = [1, 2]

    angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)

    angles_left[0, 0] += 0.1
    angles_left[1, 0] -= 0.2
    angles_left[0, 1] -= 0.07
    angles_left[1, 1] -= 0.13
    angles_bottom[0, 0] -= 0.04
    angles_bottom[1, 0] += 0.2
    angles_bottom[0, 1] += 0.09
    angles_bottom[1, 1] += 0.1

    marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
    dots, indexes = marching.create_dots(ls, cs)
    rows, cols = indexes.shape
    quads = QuadrangleArray(dots, rows, cols)
    origami = RFFQM(quads)
    plot_interactive(origami)

    omegas = np.linspace(-np.pi + 0.01, np.pi - 0.01, 16)

    for o in omegas:
        origami.set_gamma(o, should_center=False)
        quads = origami.dots
        AE_vec = quads.dots[:, quads.indexes[0, 2]] - quads.dots[:, quads.indexes[0, 0]]
        AJ_vec = quads.dots[:, quads.indexes[2, 0]] - quads.dots[:, quads.indexes[0, 0]]
        print(f"omega={o}; inner product {AE_vec.transpose().dot(AJ_vec)}")


def calc_omega():
    omega_11 = Symbol(r"\omega_11")

    s = +1
    a = cth - eta_22
    b = pi - cth + delta_22
    o = omega_11
    calculated_gamma = gamma2(s, a, b, o)

    s = -1
    a = cth - delta_24
    b = cth - eta_24
    o = calculated_gamma
    omega_12 = gamma2(s, a, b, o)
    omega_12_unpert = make_angles_vanish(omega_12).simplify()
    print("Unperturbed:")
    print(latex(omega_12_unpert))

    omega_12 = omega_12.simplify()
    print("exact:")
    print(latex(omega_12))

    # TODO: Add perturbation!!!
    print("Computing linearized")
    omega_12_linearized = sympyutils.linearize_multivariable(omega_12, all_perturbation_angles)
    omega_12_linearized = omega_12_linearized.simplify()
    print("linearized omega12")
    print(latex(omega_12_linearized))


def main():
    # calc_AE()
    # calc_gamma()
    # calc_AJ()
    # calc_g12_unperturbed()
    # test_g12()
    # calc_unperturbed_aligned()
    # calc_metric()
    calc_omega()


if __name__ == '__main__':
    main()
