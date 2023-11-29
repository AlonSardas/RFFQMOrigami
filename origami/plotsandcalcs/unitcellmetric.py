import numpy as np
from sympy import *

from origami.RFFQMOrigami import RFFQM
from origami.quadranglearray import QuadrangleArray

delta_11, delta_12, eta_12, eta_21, delta_22, eta_22 = symbols(
    r"\delta_{1\,1}, \delta_{1\,2}, \eta_{1\,2}, \eta_{2\,1}, \delta_{2\,2}, \eta_{2\,2}")
delta_24, eta_24 = symbols(r"\delta_{2\,4}, \eta_{2\,4}")
c_11, d_11, l_11, l_12, m_11 = symbols(r"c_{1\,1}, d_{1\,1}, \ell_{1\,1}, \ell_{1\,2}, m_{1\,1}", positive=True)
cth = symbols(r"\vartheta")
omega11, gamma11 = symbols(r"\omega_{1\,1}, \gamma_{1\,1}")
omegat, gammat = symbols(r"\omega_t, \gamma_t")
I = eye(3)
e_3 = Matrix([0, 0, 1])


class SymClass(object):
    latex_symbols = ['omega', 'gamma', 'delta', 'eta', 'ell']

    def __getattr__(self, item) -> Expr:
        name, indexes = item.split('_')
        i, j = indexes
        if name in self.latex_symbols:
            name = '\\' + name
        return Symbol(name + '_{' + i + ',' + j + '}')


sym = SymClass()


def _rotate_XY(angle) -> Matrix:
    return Matrix([[cos(angle), -sin(angle), 0],
                   [sin(angle), cos(angle), 0],
                   [0, 0, 0]])


def _create_rotation_around_axis(t: Matrix, angle) -> Matrix:
    t_perp = Matrix([-t[1], t[0], 0])  # A vector in XY plane that is perp to t
    P = I - _outer(t, t)
    W = _outer(e_3, t_perp) - _outer(t_perp, e_3)
    R = _outer(t, t) + cos(angle) * P + sin(angle) * W
    return R


def _outer(v1: Matrix, v2: Matrix) -> Matrix:
    return v1 * v2.transpose()


def calc_vectors():
    pA0 = Matrix([0, 0, 0])
    pB0 = Matrix([0, l_11, 0])
    ABn = Matrix([0, 1, 0])

    BDn = _rotate_XY(pi - (cth + eta_21)) * (-1 * ABn)
    BDn.simplify()

    rot_gamma11 = _create_rotation_around_axis(-BDn, -gamma11)
    BJn = _rotate_XY(cth + sym.delta_21) * BDn
    BJn.simplify()
    BJ0 = m_11 * BJn
    BJ = rot_gamma11 * BJ0
    AJ = pB0 + BJ
    # AJ = simplify(AJ)
    print('---- AJ')
    print(latex(AJ))
    print()

    ACn = _rotate_XY(-cth - delta_11) * ABn
    pC0 = ACn * c_11
    CDn = _rotate_XY(-(pi - cth + eta_12)) * (-ACn)
    CDn = simplify(CDn)
    # print("CDn", CDn)
    R = _create_rotation_around_axis(-CDn, omega11)
    CEn = _rotate_XY(-(pi - cth + delta_12)) * CDn
    CE0 = CEn * d_11
    CE0 = simplify(CE0)
    pE0 = pC0 + CE0
    CE = R * CE0
    AE = pC0 + CE
    print('---- AE')
    print(latex(AE))
    print()

    JIn = _rotate_XY(pi - (cth + sym.eta_31)) * (-1 * BJn)
    JIn.simplify()

    def calc_AJ2():
        AC2n = JIn
        AB2n = _rotate_XY(cth + sym.delta_31) * AC2n
        # print(AB2n)
        AB2n.simplify()
        # print(AB2n)
        AB2_0 = AB2n * sym.ell_21
        BD2n = _rotate_XY(pi - cth - sym.eta_41) * (-1 * AB2n)
        BD2n.simplify()
        rot_gamma21 = _create_rotation_around_axis(-BD2n, -sym.gamma_21)
        BJ2n = _rotate_XY(cth + sym.delta_41) * BD2n
        BJ2n.simplify()
        BJ2_0 = sym.m_21 * BJ2n
        BJ2 = rot_gamma21 * BJ2_0
        AJ2 = AB2_0 + BJ2

        rot_gammat = _create_rotation_around_axis(-JIn, -gammat)
        AJ2 = rot_gammat * AJ2
        rot_gamma11 = _create_rotation_around_axis(-BDn, -gamma11)
        AJ2 = rot_gamma11 * AJ2

        print('---- AJ2')
        print(latex(AJ2))
        print()

    def calc_AEy2():
        print(JIn)

        ACy2n = JIn
        ACy2 = ACy2n * sym.c_21
        CDy2n = _rotate_XY(-(pi - cth + sym.eta_32)) * (-ACy2n)
        CDy2n = simplify(CDy2n)
        # print("CDy2n", CDy2n)

        # Note that here we rotate by -omega!!! it seems to be consistent
        Ry2 = _create_rotation_around_axis(-CDy2n, -sym.omega_21)
        CEy2n = _rotate_XY(-(pi - cth + sym.delta_32)) * CDy2n
        CEy2_0 = CEy2n * sym.d_21
        CEy2_0 = simplify(CEy2_0)
        CEy2 = Ry2 * CEy2_0
        AEy2 = ACy2 + CEy2

        rot_gamma11 = _create_rotation_around_axis(-BDn, -gamma11)
        AEy2 = rot_gamma11 * AEy2

        print('---- AEy2')
        print(latex(AEy2))
        print()

    def calc_AE2x():
        EFn = _rotate_XY(-(cth + sym.eta_13)) * (-CEn)
        print("trigsimp EFn")
        EFn = EFn.trigsimp()
        AB2n = EFn
        AC2n = _rotate_XY(-cth - sym.delta_13) * AB2n
        print("simplify AC2n")
        AC2n = simplify(AC2n)
        AC2 = sym.c_12 * AC2n
        CD2n = _rotate_XY(-(pi - cth + sym.eta_14)) * (-AC2n)
        print("Simplify CD2n")
        CD2n = simplify(CD2n)
        R = _create_rotation_around_axis(CD2n, sym.omega_12)
        R = R.transpose()  # Just for a convention of the sign of omega
        CE2n = _rotate_XY(-(pi - cth + sym.delta_14)) * CD2n
        # print(CE2n)
        # print()
        print("Simplify CE2n")
        CE2n = CE2n.simplify()
        # print(CE2n)
        CE2_0 = CE2n * sym.d_12
        CE2 = R * CE2_0
        # AE2 = pC2_0 + CE2 - pA2_0
        AE2 = AC2 + CE2

        R = _create_rotation_around_axis(EFn, omegat)
        R = R.transpose()  # Just for a convention of the sign of omega
        AE2 = R * AE2
        """ This is a bit more explicit calculations, but it is not necessary
        AE2 = AE2
        pE2 = pA2_0 + AE2
    
        R = _create_rotation_around_axis(CDn, omega11).transpose()
        pA2 = R * (pA2_0-pC0)+pC0
        pE2 = R * (pE2-pC0)+pC0
        """
        R = _create_rotation_around_axis(CDn, omega11).transpose()
        AE2 = R * AE2
        # print("Simplifying AE2")
        # trigsimp seems a bit faster than simplify
        # AE2 = AE2.trigsimp()

        print("----- AE2")
        print(latex(AE2))
        print()

    # calc_AEy2()
    # calc_AJ2()

    return AE, AJ


def calc_omega12():
    omega_11 = omega11

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
    omega_12 = simplify(omega_12)
    print("---- omega12")
    print(latex(omega_12))
    print()

    s = -1
    a = pi - (cth + sym.delta_23)
    b = pi - (cth + sym.eta_23)
    o = calculated_gamma
    omegat_v = gamma2(s, a, b, o)
    omegat_v = simplify(omegat_v)
    print('---- omegat')
    print(latex(omegat_v))
    print()


def calc_gamma21():
    s = -1
    a = cth - delta_22
    b = cth - eta_22
    o = gamma11
    calculated_gamma = gamma2(s, a, b, o)

    s = 1
    a = cth - sym.eta_42
    b = pi - cth + sym.delta_42
    o = calculated_gamma
    gamma_21 = gamma2(s, a, b, o)
    gamma_21 = simplify(gamma_21)
    print('---- gamma21')
    print(latex(gamma_21))
    print()


def gamma2(s, a, b, o):
    nom = (-s + cos(a) * cos(b)) * cos(o) + sin(a) * sin(b)
    deno = -s + cos(a) * cos(b) + sin(a) * sin(b) * cos(o)
    calculated_gamma = acos(nom / deno)
    return calculated_gamma


def test():
    quads = QuadrangleArray(np.array(
        [[sin(delta_11), 1 + sin(delta_22), 2 + sin(eta_12), sin(delta_11), 1 + sin(delta_22), 2 + sin(eta_12)],
         [cos(delta_11), cos(delta_12), cos(eta_21), 1 + cos(delta_11), 1 + cos(delta_12), 1 + cos(eta_21)],
         [0, 0, 0, 0, 0, 0]]), 2, 3)
    a = RFFQM(quads)
    a.set_gamma(gamma, should_center=False)
    print(a.dots.dots)


def test_rotate_around_axis():
    t = -Matrix([0, 1, 0])
    R = _create_rotation_around_axis(t, pi / 2)
    v0 = Matrix([1, 0, 0])
    print(R*v0)


def main():
    # test_rotate_around_axis()
    # calc_vectors()
    # calc_omega12()
    calc_gamma21()


if __name__ == '__main__':
    main()
