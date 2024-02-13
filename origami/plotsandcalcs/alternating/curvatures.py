import logging
import numbers
from typing import Tuple, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.integrate import solve_ivp
from sympy import Expr

from origami.plotsandcalcs.alternating.utils import csc, tan, sec, cos, sin, plot_perturbations
from origami.utils import sympyutils

# We initialize these values on demand since parsing latex takes long time
_kx, _ky = None, None

Ft = sympyutils.from_latex("F(x)")
Ftt = sympyutils.from_latex("F'(x)")
Mt = sympyutils.from_latex("M(y)")
Mtt = sympyutils.from_latex("M'(y)")

logger = logging.getLogger('origami.alternating.curvatures')


def sub_values(expr: sympy.Expr, L_tot, C_tot, W0, theta) -> sympy.Expr:
    L_tot_sym = sympy.Symbol("L_{t}")
    C_tot_sym = sympy.Symbol("C_{t}")
    W0_sym = sympy.Symbol("W_{0}")
    theta_sym = sympy.Symbol(r"vartheta")
    s_sym = sympy.Symbol("s")
    t_sym = sympy.Symbol("t")

    subs = [(L_tot_sym, L_tot),
            (C_tot_sym, C_tot),
            (W0_sym, W0),
            (theta_sym, theta),
            (s_sym, 1), (t_sym, 1)]
    return expr.subs(subs)


def get_ky_kx_for_values(L_tot, C_tot, W0, theta) -> Tuple[Expr, Expr]:
    kx, ky = _get_kx_ky_expr()
    kx_sub = sub_values(kx, L_tot, C_tot, W0, theta)
    ky_sub = sub_values(ky, L_tot, C_tot, W0, theta)
    return kx_sub, ky_sub


def _get_kx_ky_expr():
    global _kx, _ky

    def replace_latex(txt):
        return (txt.replace(r"\tilde{\Delta}", "M")
                .replace(r"\tilde{\delta}", "F")
                .replace(r"\omega_{0,0}", "W_{0}")
                .replace("L_{tot}", "L_t")
                .replace("C_{tot}", "C_t"))

    if _ky is None:
        _ky = sympyutils.from_latex(replace_latex(
            r"\frac{t\sin^{2}\left(\vartheta\right)\cos\left(\vartheta\right)\sin\left(\frac{\omega_{0,"
            r"0}}{2}\right)\tilde{\Delta}'\left(y\right)\sqrt{2}\sqrt{2\csc^{2}\left(\vartheta\right)-\cos\left("
            r"\omega_{0,0}\right)-1}}{8L_{tot}*\left(\cos^{2}\left(\vartheta\right)+t\cos^{2}\left("
            r"\vartheta\right)\tilde{\Delta}\left(y\right)+\frac{1}{8}t^{2}\tilde{\Delta}\left(y\right)^{2}\sin\left("
            r"\vartheta\right)\left(2\csc^{2}\left(\vartheta\right)-\cos\left(\omega_{0,0}\right)-1\right)\right)^{"
            r"3/2}}"))
    if _kx is None:
        _kx = sympyutils.from_latex(replace_latex(
            r"\frac{s\csc\left(\vartheta\right)\tan\left(\frac{\omega_{0,0}}{2}\right)\sec\left(\frac{\omega_{0,"
            r"0}}{2}\right)\tilde{\delta}'\left(x\right)\sqrt{2}\sqrt{2\csc^{2}\left(\vartheta\right)-\cos\left("
            r"\omega_{0,0}\right)-1}}{4C_{tot}*\left(1+\frac{1}{2}s^{2}\tan^{2}\left(\frac{\omega_{0,"
            r"0}}{2}\right)\tilde{\delta}\left(x\right)^{2}\left(2\csc^{2}\left(\vartheta\right)-\cos\left(\omega_{0,"
            r"0}\right)-1\right)\right)^{3/2}}"))

    return _kx, _ky


def create_kx_ky_funcs(L_tot, C_tot, W0, theta) -> Tuple[Callable, Callable]:
    kx_sub, ky_sub = get_ky_kx_for_values(L_tot, C_tot, W0, theta)
    logger.debug(f"kx func: {kx_sub}")
    logger.debug(f"ky func: {ky_sub}")
    kx_func = sympy.lambdify([Ft, Ftt], kx_sub)
    ky_func = sympy.lambdify([Mt, Mtt], ky_sub)

    return kx_func, ky_func


def create_kx_ky_funcs_linearized(L_tot, C_tot, W0, theta) -> Tuple[Callable, Callable]:
    sqrt = np.sqrt
    th = theta
    kx_factor = csc(th) * tan(W0 / 2) * sec(W0 / 2) * sqrt(2) * sqrt(2 * csc(th) ** 2 - cos(W0) - 1) / (4 * C_tot)
    ky_factor = tan(th) ** 2 * sin(W0 / 2) * sqrt(2) * sqrt(2 * csc(th) ** 2 - cos(W0) - 1) / (8 * L_tot)

    kx_func = lambda ddelta: kx_factor * ddelta
    ky_func = lambda dDelta: ky_factor * dDelta
    return kx_func, ky_func


def get_delta_func_for_kx(L_tot, C_tot, W0, theta,
                          kx: Union[float, Callable[[float], float]],
                          delta0: float) -> Callable[[float], float]:
    d_func = _create_ddelta_func(L_tot, C_tot, W0, theta, kx)
    logger.debug(f"Solving equation delta'={d_func}, for initial condition {delta0} on range [0,1]")
    t0, tf = 0, 1
    sol = solve_ivp(d_func, (t0, tf), [delta0], dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    return _get_sol_wrapper(sol.sol)


def _get_sol_wrapper(sol):
    def sol_wrapper(t):
        if isinstance(t, np.ndarray):
            if len(t.shape) == 2:
                vals = sol(t.flatten())[0]
                return vals.reshape(t.shape)
            elif len(t.shape) >= 3:
                raise RuntimeError("Not supported")

        return sol(t)[0]

    return sol_wrapper


def get_deltas_for_kx(L_tot, C_tot, W0, theta,
                      kx: Union[float, Callable[[float], float]],
                      delta0: float, chi: float) -> Tuple[np.ndarray, np.ndarray]:
    d_func = _create_ddelta_func(L_tot, C_tot, W0, theta, kx)
    logger.debug(f"Solving equation delta'={d_func}, for initial condition {delta0} on range [0,1]")
    t0, tf = 0, 1
    ts = np.arange(t0, tf + 0.1 * chi, 0.5 * chi)
    sol = solve_ivp(d_func, (t0, tf), [delta0], t_eval=ts)
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    return sol.t, sol.y[0, :]


def _create_ddelta_func(L_tot, C_tot, W0, theta, kx):
    kx_sub, _ = get_ky_kx_for_values(L_tot, C_tot, W0, theta)
    kx_symbol = sympy.Symbol('k_x')
    eq = kx_sub - kx_symbol
    dfdt = sympy.solve(eq, Ftt)[0]
    dfdt_func = sympy.lambdify([Ft, kx_symbol], dfdt)

    if isinstance(kx, numbers.Number):
        kx_num = kx
        kx = lambda t: kx_num

    def d_func(t, ft):
        return dfdt_func(ft, kx(t))

    return d_func


def get_Delta_func_for_ky(L_tot, C_tot, W0, theta,
                          ky: Union[float, Callable[[float], float]],
                          Delta0: float) -> Callable[[float], float]:
    d_func = _create_dDelta_eq(L_tot, C_tot, W0, theta, ky)
    sol = solve_ivp(d_func, (0, 1), [Delta0], dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    return _get_sol_wrapper(sol.sol)


def get_Deltas_for_ky(L_tot, C_tot, W0, theta,
                      ky: Union[float, Callable[[float], float]],
                      Delta0: float, xi) -> Tuple[np.ndarray, np.ndarray]:
    d_func = _create_dDelta_eq(L_tot, C_tot, W0, theta, ky)
    logger.debug(f"Solving equation Delta'={d_func}, for initial condition {Delta0} on range [0,1]")
    t0, tf = 0, 1
    ts = np.arange(t0, tf, xi)
    sol = solve_ivp(d_func, (t0, tf), [Delta0], t_eval=ts)
    if not sol.success:
        raise RuntimeError(f"Could not solve the ODE. Error: {sol.message}")

    return sol.t, sol.y[0, :]


def _create_dDelta_eq(L_tot, C_tot, W0, theta, ky):
    _, ky_sub = get_ky_kx_for_values(L_tot, C_tot, W0, theta)
    ky_symbol = sympy.Symbol('k_y')
    eq = ky_sub - ky_symbol
    ddMdt = sympy.solve(eq, Mtt)[0]
    dDelta_dt_func = sympy.lambdify([Mt, ky_symbol], ddMdt)

    if isinstance(ky, numbers.Number):
        ky_num = ky
        ky = lambda t: ky_num

    def d_func(t, mt):
        return dDelta_dt_func(mt, ky(t))

    return d_func


def get_Delta_for_ky_by_recurrence(L_tot, C_tot, W0, theta,
                                   ky: Union[float, Callable[[float], float]],
                                   Delta0: float, Ny: int):
    _, ky_sub = get_ky_kx_for_values(L_tot, C_tot, W0, theta)
    ky_symbol = sympy.Symbol('k_y')
    eq = ky_sub - ky_symbol
    dDelta_dt = sympy.solve(eq, Mtt)[0]
    dDelta_dt_func = sympy.lambdify([Mt, ky_symbol], dDelta_dt)

    if isinstance(ky, numbers.Number):
        ky_num = ky
        ky = lambda t: ky_num

    Deltas = np.zeros(Ny)
    Deltas[0] = Delta0
    ys = np.arange(Ny) / Ny
    for i in range(1, Ny - 1):
        Deltas[i] = Deltas[i - 1] + 1 / Ny * dDelta_dt_func(Deltas[i - 1], ky(i / Ny))
    return ys, Deltas


def create_expected_K_func(L_tot, C_tot, W0, theta, delta, Delta) -> Callable:
    return create_expected_curvatures_func(L_tot, C_tot, W0, theta, delta, Delta)[0]


def create_expected_curvatures_func(
        L_tot, C_tot, W0, theta, delta, Delta) -> Tuple[Callable, Callable]:
    eps = 0.02
    ddelta = lambda t: 1 / (2 * eps) * (delta(t + eps) - delta(t - eps))
    dDelta = lambda t: 1 / (2 * eps) * (Delta(t + eps) - Delta(t - eps))

    # def dDelta(t):
    #     val = 1 / (2 * eps) * (Delta(t + eps) - Delta(t - eps))
    #     print(f"evaluating {t}, {val}")
    #     return val

    kx_func, ky_func = create_kx_ky_funcs(L_tot, C_tot, W0, theta)
    expected_K_func = lambda x, y: kx_func(delta(x), ddelta(x)) * ky_func(Delta(y), dDelta(y))
    expected_H_func = lambda x, y: 1 / 2 * (kx_func(delta(x), ddelta(x)) + ky_func(Delta(y), dDelta(y)))
    return expected_K_func, expected_H_func
