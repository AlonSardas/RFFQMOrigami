import itertools

from sympy import *

from origami.utils import sympyutils

delta_11, delta_12, eta_12, eta_21, delta_22, eta_22 = symbols(
    r"\delta_11, \delta_12, \eta_12, \eta_21, \delta_22, \eta_22")
all_perturbation_angles = (delta_11, delta_12, eta_12, eta_21)
c_11, d_11, l_11, l_12 = symbols(r"c_11, d_11, \ell_11, \ell_12")
cth = symbols(r"\vartheta")
omega, gamma = symbols(r"\omega, \gamma")
I = eye(3)
e_3 = Matrix([0, 0, 1])


def make_angles_vanish(expr: Expr) -> Expr:
    subs_dict = dict(zip(all_perturbation_angles, itertools.repeat(0)))
    return expr.subs(subs_dict)


def _outer(v1: Matrix, v2: Matrix) -> Matrix:
    return v1 * v2.transpose()


def calc_AE() -> Expr:
    t = Matrix([sin(eta_12 + delta_11), cos(eta_12 + delta_11), 0])
    t_perp = Matrix([-cos(eta_12 + delta_11), sin(eta_12 + delta_11), 0])
    P = I - _outer(t, t)
    W = _outer(e_3, t_perp) - _outer(t_perp, e_3)

    R = _outer(t, t) + cos(omega) * P + sin(omega) * W

    t_l = Matrix([eta_12 + delta_11, 1, 0])
    t_perp_l = Matrix([-1, eta_12 + delta_11, 0])
    P = I - _outer(t_l, t_l)
    W = _outer(e_3, t_perp_l) - _outer(t_perp_l, e_3)
    R = _outer(t_l, t_l) + cos(omega) * P + sin(omega) * W

    print("----------- linearized R=")
    print(latex(R))

    a_sum = (delta_11 + eta_12 + delta_12)

    AC_l = c_11 * Matrix([sin(cth) + cos(cth) * delta_11, cos(cth) - sin(cth) * delta_11, 0])
    CE_l = d_11 * Matrix([sin(cth) - cos(cth) * a_sum, -cos(cth) - sin(cth) * a_sum, 0])

    AE_l = AC_l + R * CE_l

    print("----------- linearized AE=")
    print(latex(AE_l))
    print(latex(sympify(AE_l)))

    AE_l = sympyutils.linearize_multivariable(AE_l, (delta_11, delta_12, eta_12))

    print("-----------Really linearized AE=")
    print(latex(sympify(AE_l)))

    print("------------Unperturbed AE=")
    AE_unperturbed = make_angles_vanish(AE_l)
    print(latex(simplify(AE_unperturbed)))

    return AE_l


def calc_gamma():
    s = +1
    a = cth - delta_22
    b = pi - cth + eta_22
    o = omega

    nom = (-s + cos(a) * cos(b)) * cos(o) + sin(a) * sin(b)
    deno = -s + cos(a) * cos(b) + sin(a) * sin(b) * cos(o)
    calculated_gamma = acos(nom / deno)

    print(latex(calculated_gamma))

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
    gamma_sign = 1
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
    print(latex(AJ_l))
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


def main():
    # calc_AE()
    # calc_gamma()
    # calc_AJ()
    calc_g12_unperturbed()
    # calc_metric()


if __name__ == '__main__':
    main()
