import numbers
from typing import Tuple, Callable, Union

import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from sympy import Expr

from origami.plotsandcalcs.alternating.utils import get_FF_dFF_dMM_ddMM
from origami.utils import sympyutils

# We initialize these values on demand since parsing latex takes long time
_kx, _ky = None, None

Ft = sympyutils.from_latex("F(t)")
Ftt = sympyutils.from_latex("F'(t)")
Mt = sympyutils.from_latex("M'(t)")
Mtt = sympyutils.from_latex("M''(t)")


def sub_values(expr: sympy.Expr, L0, C0, W0, theta) -> sympy.Expr:
    L0sym = sympy.Symbol("L_{0}")
    C0sym = sympy.Symbol("C_{0}")
    W0sym = sympy.Symbol("W_{0}")
    theta_sym = sympy.Symbol(r"vartheta")
    F0sym = sympy.Symbol("F_{0}")
    chi_sym = sympy.Symbol(r"chi")

    subs = [(L0sym, L0),
            (C0sym, C0),
            (W0sym, W0),
            (theta_sym, theta),
            (F0sym, 1), (chi_sym, 1)]
    return expr.subs(subs)


def get_ky_kx_for_values(L0, C0, W0, theta) -> Tuple[Expr, Expr]:
    global _kx, _ky
    if _ky is None:
        _ky = sympyutils.from_latex(
            r"-\frac{L_{0}\sin\left(\frac{W_{0}}{2}\right)\sin(\vartheta)\cos(\vartheta)M''(t)\sqrt{-2\cos\left(W_{"
            r"0}\right)\sin^{2}(\vartheta)+\cos(2\vartheta)+3}}{\left(4L_{0}\cos^{2}(\vartheta)M'(t)+4L_{0}^{2}\cos^{2}("
            r"\vartheta)+M'(t)^{2}\left(\sin^{2}\left(\frac{W_{0}}{2}\right)\sin^{2}(\vartheta)+\cos^{2}("
            r"\vartheta)\right)\right)^{3/2}}")
    if _kx is None:
        _kx = sympyutils.from_latex(
            r"-\frac{2F_{0}\chi\cos\left(\frac{W_{0}}{2}\right)\sin(\vartheta)\cos(\vartheta)F'(t)\left(\sin\left(W_{"
            r"0}\right)(\cos(2\vartheta)+3)-\sin\left(2W_{0}\right)\sin^{2}(\vartheta)\right)}{C_{0}\sqrt{\sin^{2}\left("
            r"W_{0}\right)\sin^{2}(2\vartheta)+16\cos^{2}\left(\frac{W_{0}}{2}\right)\cos^{4}(\vartheta)}\left(\cos^{"
            r"2}\left(\frac{W_{0}}{2}\right)\sin^{2}(\vartheta)\left(F_{0}^{2}F(t)^{2}\sin^{2}\left(\frac{W_{0}}{"
            r"2}\right)+2\right)^{2}+\frac{F_{0}^{2}F(t)^{2}\cos^{2}(\vartheta)\left(4F_{0}F(t)\sin\left(\frac{W_{0}}{"
            r"2}\right)\cos^{3}\left(\frac{W_{0}}{2}\right)\sin(\vartheta)\cos(\vartheta)+\sin\left(2W_{0}\right)\sin^{"
            r"2}(\vartheta)-\sin\left(W_{0}\right)(\cos(2\vartheta)+3)\right)^{2}}{\sin^{2}\left(W_{0}\right)\sin^{2}("
            r"2\vartheta)+16\cos^{2}\left(\frac{W_{0}}{2}\right)\cos^{4}(\vartheta)}\right)^{3/2}}")

    kx_sub = sub_values(_kx, L0, C0, W0, theta)
    ky_sub = sub_values(_ky, L0, C0, W0, theta)
    return kx_sub, ky_sub


def create_kx_ky_funcs(L0, C0, W0, theta) -> Tuple[Callable, Callable]:
    kx_sub, ky_sub = get_ky_kx_for_values(L0, C0, W0, theta)
    kx_func = sympy.lambdify([Ft, Ftt], kx_sub)
    ky_func = sympy.lambdify([Mt, Mtt], ky_sub)

    return kx_func, ky_func


def get_F_for_kx(L0, C0, W0, theta,
                 kx: Union[float, Callable[[float], float]],
                 F0: float, t0: int, tf: int) -> Tuple[np.ndarray, np.ndarray]:
    kx_sub, _ = get_ky_kx_for_values(L0, C0, W0, theta)
    kx_symbol = sympy.Symbol('k_x')
    eq = kx_sub - kx_symbol
    dfdt = sympy.solve(eq, Ftt)[0]
    dfdt_func = sympy.lambdify([Ft, kx_symbol], dfdt)

    if isinstance(kx, numbers.Number):
        kx_num = kx
        kx = lambda t: kx_num

    def d_func(t, ft):
        return dfdt_func(ft, kx(t))

    # We look at F every half step, since there are 2 perturbations for
    # each step in F
    ts = np.arange(t0, tf + 0.1, 0.5)
    sol = solve_ivp(d_func, (t0, tf), [F0], t_eval=ts)
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    return sol.t, sol.y[0, :]


def get_MM_for_ky(L0, C0, W0, theta,
                  ky: Union[float, Callable[[float], float]],
                  M0: float, t0: int, tf: int):
    _, ky_sub = get_ky_kx_for_values(L0, C0, W0, theta)
    ky_symbol = sympy.Symbol('k_y')
    eq = ky_sub - ky_symbol
    ddMdt = sympy.solve(eq, Mtt)[0]
    dMdt_func = sympy.lambdify([Mt, ky_symbol], ddMdt)

    if isinstance(ky, numbers.Number):
        ky_num = ky
        ky = lambda t: ky_num

    def d_func(t, mt):
        return dMdt_func(mt, ky(t))

    sol = solve_ivp(d_func, (t0, tf), [M0], t_eval=np.arange(t0, tf + 1))
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    ys = np.append(0, np.cumsum(sol.y[0, :]))
    return sol.t, ys


def get_MM_for_ky_by_recurrence(L0, C0, W0, theta,
                                ky: float,
                                M0: float, t0: int, tf: int):
    _, ky_sub = get_ky_kx_for_values(L0, C0, W0, theta)
    ky_symbol = sympy.Symbol('k_y')
    eq = ky_sub - ky_symbol
    ddMdt = sympy.solve(eq, Mtt)[0]
    dMdt_func = sympy.lambdify([Mt, ky_symbol], ddMdt)

    Ms = np.zeros(tf - t0)
    Ms[0] = M0
    for i in range(1, tf - t0):
        Ms[i] = Ms[i - 1] + dMdt_func(Ms[i - 1], ky)
    MMs = np.append(0, np.cumsum(Ms))
    return np.arange(t0, tf), MMs


def test_F_for_const():
    ky = 0.01
    W0 = 3.0
    theta = 0.8
    C0 = 0.5
    L0 = 0.2
    M0 = 0.5
    ts, fs = get_MM_for_ky(L0, C0, W0, theta, ky, M0, 0, 30)
    fig, ax = plt.subplots()
    ax.plot(ts, fs, '.')


def create_expected_K_func(L0, C0, W0, angle, F, MM) -> Callable:
    return create_expected_curvatures_func(L0, C0, W0, angle, F, MM)[0]


def create_expected_curvatures_func(L0, C0, W0, angle, F, MM) -> Tuple[Callable, Callable]:
    FF, dFF, dMM, ddMM = get_FF_dFF_dMM_ddMM(F, MM)
    kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, angle)
    expected_K_func = lambda x, y: kx_func(FF(x), dFF(x)) * ky_func(dMM(y), ddMM(y))
    expected_H_func = lambda x, y: 1 / 2 * (kx_func(FF(x), dFF(x)) + ky_func(dMM(y), ddMM(y)))
    return expected_K_func, expected_H_func
