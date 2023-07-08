"""
Some history:
At first, we used alternating angle perturbation to calculate the Gaussian curvature
after linearization of the perturbed angles and vertical lengths
We got results that seems to work well but only when the deflection of the folded
paper from a plane were small. In a sense, this is what we could expect.

Here we test the hypothesis that the Gaussian curvature is simply the multiplication
of the radii of curvature caused separately by the angle perturbation and by the lengths perturbation
Hopefully it will give us better predictions to the Gaussian curvature.
"""
from typing import Tuple, Callable

import numpy as np
import sympy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import fsolve

from origami import origamimetric
from origami.interactiveplot import plot_interactive
from origami.plotsandcalcs.continuousmetricalternating import cos, tan, sec, csc
from origami.plotsandcalcs.continuousmetricalternating import create_perturbed_origami, _get_FF_dFF_dMM_ddMM, \
    _compare_curvatures
from origami.utils import sympyutils


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


_kx, _ky = None, None


def create_kx_ky_funcs(L0, C0, W0, theta) -> Tuple[Callable, Callable]:
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

    Ft = sympyutils.from_latex("F(t)")
    Ftt = sympyutils.from_latex("F'(t)")
    Mt = sympyutils.from_latex("M'(t)")
    Mtt = sympyutils.from_latex("M''(t)")
    kx_func = sympy.lambdify([Ft, Ftt], kx_sub)
    ky_func = sympy.lambdify([Mt, Mtt], ky_sub)

    return kx_func, ky_func


def test_constant():
    rows = 40
    cols = 40
    theta = np.pi / 2 - 0.2
    W0 = 3

    F = lambda x: 0.01 * (x - cols / 2)
    MM = lambda y: 0.04 * ((y - rows / 4)) ** 2

    FF, dFF, dMM, ddMM = _get_FF_dFF_dMM_ddMM(F, MM)

    L0 = 1
    C0 = 0.5

    ori = create_perturbed_origami(theta, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))

    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

    kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, theta)
    expected_K_func = lambda x, y: kx_func(FF(x), dFF(x)) * ky_func(dMM(y), ddMM(y))

    fig, axes = plt.subplots(2)
    _compare_curvatures(fig, axes, Ks, expected_K_func)
    fig.tight_layout()

    angle = theta
    dF0 = dFF(1)
    ddMM0 = ddMM(1)
    expectedK = -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dF0 * \
                ddMM0 * (cos(W0) - 2 * csc(angle) ** 2 + 1)
    print(expectedK)

    plot_interactive(ori)

    plt.show()


def test_hills():
    rows, cols = 80, 80
    angle = np.pi / 2 - 0.2
    W0 = 3

    L0 = 1
    C0 = 0.5

    F0, MM0 = 0.1, 0.7  # This is large perturbations

    F = lambda x: F0 * np.sin(2 * np.pi * x / 39)
    MM = lambda y: MM0 * np.cos(2 * np.pi * y / 16)
    FF, dFF, dMM, ddMM = _get_FF_dFF_dMM_ddMM(F, MM)

    ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
    ori.set_gamma(ori.calc_gamma_by_omega(W0))
    Ks, g11, g12, g22 = origamimetric.calc_curvature_and_metric(ori.dots)

    expected_K_func_bad = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(x) * \
                                       ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)

    kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, angle)
    expected_K_func = lambda x, y: kx_func(FF(x), dFF(x)) * ky_func(dMM(y), ddMM(y))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    vmin, vmax = -4.1, 4.1
    len_ys, len_xs = Ks.shape
    xs, ys = np.arange(len_xs), np.arange(len_ys)
    Xs, Ys = np.meshgrid(xs, ys)

    im = axes[0].imshow(Ks, vmin=vmin, vmax=vmax, origin='lower')
    axes[1].imshow(expected_K_func(Xs, Ys), vmin=vmin, vmax=vmax, origin='lower')
    axes[2].imshow(expected_K_func_bad(Xs, Ys), vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title("Ks")
    axes[1].set_title("better\napproximation")
    axes[2].set_title("old\napproximation")
    print(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
    fig.suptitle(rf'$ \omega={W0:.3f} $, F0={F0:.3f}, MM0={MM0:.3f}')
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.08, shrink=1.0)


def test_constant_angle_factor():
    angles_factor = 1000

    L0 = 1
    C0 = 0.5
    rows, cols = 30, 40
    F = lambda x: 0.005 * (x - cols / 2)

    MM = lambda y: 0.02 * ((y - rows / 4)) ** 2
    FF, dFF, dMM, ddMM = _get_FF_dFF_dMM_ddMM(F, MM)

    def find_omega_for_angle(angle) -> float:
        eq_func = lambda w: -tan(w / 2) ** 2 * tan(angle) * sec(angle) * \
                            (cos(w) - 2 * csc(angle) ** 2 + 1) - angles_factor
        return fsolve(eq_func, 1, factor=0.1)[0]

    print(F(0))

    angles = [0.7, 1, 1.4, 1.5]
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.5))
    fig: Figure = fig
    for i, angle in enumerate(angles):
        W0 = find_omega_for_angle(angle)
        print('W0', W0)

        ori = create_perturbed_origami(angle, rows, cols, L0, C0, F, MM)
        expected_K_func_bad = lambda x, y: -1 / (16 * C0 * L0 ** 2) * tan(W0 / 2) ** 2 * tan(angle) * sec(angle) * dFF(
            x) * \
                                           ddMM(y) * (cos(W0) - 2 * csc(angle) ** 2 + 1)
        ori.set_gamma(ori.calc_gamma_by_omega(W0))
        Ks, _, _, _ = origamimetric.calc_curvature_and_metric(ori.dots)
        K0 = expected_K_func_bad(1, 1)
        print(K0)

        expected_K_func = create_expected_K_func(L0, C0, W0, angle, F, MM)

        len_ys, len_xs = Ks.shape
        xs, ys = np.arange(len_xs), np.arange(len_ys)
        Xs, Ys = np.meshgrid(xs, ys)

        im = axes[0, i].imshow(Ks, vmin=0.005, vmax=0.07, origin='lower')
        axes[0, i].set_title(fr'$ \vartheta={angle:.2f} $' + '\n' + f'W0={W0:.2f}')
        im2 = axes[1, i].imshow(expected_K_func(Xs, Ys), vmin=0.005, vmax=0.07, origin='lower')
        axes[1, i].set_title("predicted")

    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.08, shrink=1.0)

    plt.show()


def create_expected_K_func(L0, C0, W0, angle, F, MM) -> Callable:
    return create_expected_curvatures_func(L0, C0, W0, angle, F, MM)[0]


def create_expected_curvatures_func(L0, C0, W0, angle, F, MM) -> Tuple[Callable, Callable]:
    FF, dFF, dMM, ddMM = _get_FF_dFF_dMM_ddMM(F, MM)
    kx_func, ky_func = create_kx_ky_funcs(L0, C0, W0, angle)
    expected_K_func = lambda x, y: kx_func(FF(x), dFF(x)) * ky_func(dMM(y), ddMM(y))
    expected_H_func = lambda x, y: 1 / 2 * (kx_func(FF(x), dFF(x)) + ky_func(dMM(y), ddMM(y)))
    return expected_K_func, expected_H_func


def main():
    test_constant()
    # test_hills()
    # test_constant_angle_factor()
    plt.show()


if __name__ == "__main__":
    main()
