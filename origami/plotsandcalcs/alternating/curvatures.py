import logging
import numbers
from typing import Tuple, Callable, Union

import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from sympy import Expr

from origami.plotsandcalcs.alternating.utils import get_FF_dFF_dMM_ddMM, csc, tan, sec, cos, sin
from origami.utils import sympyutils

# We initialize these values on demand since parsing latex takes long time
_kx, _ky = None, None

Ft = sympyutils.from_latex("F(x)")
Ftt = sympyutils.from_latex("F'(x)")
Mt = sympyutils.from_latex("M'(y)")
Mtt = sympyutils.from_latex("M''(y)")

logger = logging.getLogger('origami.alternating.curvatures')


def sub_values(expr: sympy.Expr, L0, C0, W0, theta) -> sympy.Expr:
    L0sym = sympy.Symbol("L_{0}")
    C0sym = sympy.Symbol("C_{0}")
    W0sym = sympy.Symbol("W_{0}")
    theta_sym = sympy.Symbol(r"vartheta")
    s_sym = sympy.Symbol("s")
    t_sym = sympy.Symbol("t")

    subs = [(L0sym, L0),
            (C0sym, C0),
            (W0sym, W0),
            (theta_sym, theta),
            (s_sym, 1), (t_sym, 1)]
    return expr.subs(subs)


def get_ky_kx_for_values(L0, C0, W0, theta) -> Tuple[Expr, Expr]:
    kx, ky = _get_kx_ky_expr()
    kx_sub = sub_values(kx, L0, C0, W0, theta)
    ky_sub = sub_values(ky, L0, C0, W0, theta)
    return kx_sub, ky_sub


def _get_kx_ky_expr():
    global _kx, _ky

    def replace_latex(txt):
        # txt.replace(r"\widetilde{\Delta L}", "M")
        return (txt.replace("M", "M'")
                .replace(r"\tilde{\delta}", "F")
                .replace(r"\omega_{0,0}", "W_{0}"))

    if _ky is None:
        _ky = sympyutils.from_latex(replace_latex(
            r"\frac{16L_{0}t\sin\left(\frac{W_{0}}{2}\right)\sin\left(2\vartheta\right)M'\left(y\right)\sqrt{"
            r"-\cos\left(W_{0}\right)\sin^{2}(\vartheta)+\cos^{2}(\vartheta)+1}}{\left(32L_{0}tM(y)\cos^{2}("
            r"\vartheta)+32L_{0}^{2}\cos^{2}(\vartheta)+t^{2}M\left(y\right)^{2}\left(\cos\left(2\vartheta-W_{"
            r"0}\right)+\cos\left(W_{0}+2\vartheta\right)-2\cos\left(W_{0}\right)+2\cos(2\vartheta)+6\right)\right)^{"
            r"3/2}}"))
    if _kx is None:
        _kx = sympyutils.from_latex(replace_latex(
            r"\frac{s\sin\left(\omega_{0,0}\right)\sin^{2}(\vartheta)\tilde{\delta}'(x)\sqrt{2}\sqrt{2\csc^{2}\left("
            r"\vartheta\right)-\cos\left(\omega_{0,0}\right)-1}}{C_{0} \cdot \left(2s^{2}\sin^{2}\left(\frac{"
            r"\omega_{0,"
            r"0}}{2}\right)\sin^{2}\left(\vartheta\right)\tilde{\delta}(x)^{2}\left(2\csc^{2}\left("
            r"\vartheta\right)-\cos\left(\omega_{0,0}\right)-1\right)+4\cos^{2}\left(\frac{\omega_{0,"
            r"0}}{2}\right)\sin^{2}(\vartheta)\right)^{3/2}}"))

    return _kx, _ky


def create_kx_ky_funcs(L0, C0, W0, theta) -> Tuple[Callable, Callable]:
    kx_sub, ky_sub = get_ky_kx_for_values(L0, C0, W0, theta)
    logger.debug(f"kx func: {kx_sub}")
    logger.debug(f"ky func: {ky_sub}")
    kx_func = sympy.lambdify([Ft, Ftt], kx_sub)
    ky_func = sympy.lambdify([Mt, Mtt], ky_sub)

    return kx_func, ky_func


def create_kx_ky_funcs_linearized(L0, C0, W0, theta) -> Tuple[Callable, Callable]:
    sqrt = np.sqrt
    th = theta
    kx_factor = csc(th) * tan(W0 / 2) * sec(W0 / 2) * sqrt(2) * sqrt(2 * csc(th) ** 2 - cos(W0) - 1) / 4
    ky_factor = tan(th) ** 2 * sin(W0 / 2) * sqrt(2) * sqrt(2 * csc(th) ** 2 - cos(W0) - 1) / 8

    kx_func = lambda ddelta: kx_factor * ddelta / C0
    ky_func = lambda dDeltaL: ky_factor * dDeltaL / L0 ** 2
    return kx_func, ky_func


def get_delta_for_kx(L0, C0, W0, theta,
                     kx: Union[float, Callable[[float], float]],
                     F0: float, chi: float) -> Tuple[np.ndarray, np.ndarray]:
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

    t0, tf = 0, 1
    ts = np.arange(t0, tf + 0.1 * chi, 0.5 * chi)
    sol = solve_ivp(d_func, (t0, tf), [F0], t_eval=ts)
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    return sol.t, sol.y[0, :]


def get_DeltaL_for_ky(L0, C0, W0, theta,
                      ky: Union[float, Callable[[float], float]],
                      DeltaL0: float, xi) -> Tuple[np.ndarray, np.ndarray]:
    _, ky_sub = get_ky_kx_for_values(L0, C0, W0, theta)
    ky_symbol = sympy.Symbol('k_y')
    eq = ky_sub - ky_symbol
    ddMdt = sympy.solve(eq, Mtt)[0]
    dMdt_func = sympy.lambdify([Mt, ky_symbol], ddMdt)

    logger.debug(f"Solving equation DeltaL'={ddMdt}, for initial condition {DeltaL0} on range [0,1]")

    if isinstance(ky, numbers.Number):
        ky_num = ky
        ky = lambda t: ky_num

    def d_func(t, mt):
        return dMdt_func(mt, ky(t))

    t0, tf = 0, 1
    ts = np.arange(t0, tf, xi)
    sol = solve_ivp(d_func, (t0, tf), [DeltaL0], t_eval=ts)
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    return sol.t, sol.y[0, :]


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
    ts, fs = get_DeltaL_for_ky(L0, C0, W0, theta, ky, M0, 0, 30)
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
