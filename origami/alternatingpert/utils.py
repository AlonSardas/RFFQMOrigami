from typing import Optional, Tuple

import numpy as np

from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import AnglesFuncType, create_angles_func_vertical_alternation, set_perturbations_by_func
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles
from origami.utils.plotutils import imshow_with_colorbar

sin, cos, tan = np.sin, np.cos, np.tan


def cot(x): return 1 / tan(x)


def csc(x): return 1 / sin(x)


def sec(x): return 1 / cos(x)


def compare_curvatures(fig, axes, Ks, expected_K_func):
    len_ys, len_xs = Ks.shape
    xs, ys = np.arange(len_xs), np.arange(len_ys)
    Xs, Ys = np.meshgrid(xs, ys)

    im = imshow_with_colorbar(fig, axes[0], Ks, "K")
    vmin, vmax = im.get_clim()
    im2 = imshow_with_colorbar(fig, axes[1], expected_K_func(Xs, Ys), "expected K")
    im2.set_clim(vmin, vmax)


def get_FF_dFF_dMM_ddMM(F, MM):
    # There is no good reason why the derivative of F is calculated differently
    def FF(x): return F(x * 2)

    def dFF(x): return (FF(x + 0.5) - FF(x)) / 0.5

    def dMM(y): return MM(y + 1) - MM(y)

    def ddMM(y): return dMM(y + 1) - dMM(y)

    return FF, dFF, dMM, ddMM


def create_perturbed_origami(theta, N_y, N_x, L_tot, C_tot,
                             delta: Optional[AnglesFuncType], Delta: Optional[AnglesFuncType]) -> RFFQM:
    if delta is None:
        def delta(x): return 0 * x
    if Delta is None:
        def Delta(y): return 0 * y

    chi = 1 / N_x
    xi = 1 / N_y

    def F1(n):
        return delta(n / 2 * chi)

    def F2(n):
        return -delta(n / 2 * chi)

    rows = 2 * N_y
    cols = 2 * N_x

    L0 = L_tot * xi
    C0 = C_tot * chi
    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0

    i = np.arange(N_y)
    ls[1::2] += L0 * Delta(i * xi)

    angles_left, angles_bottom = create_miura_angles(ls, cs, theta)
    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)
    return ori


def create_perturbed_origami_by_list(
        theta, L0, C0,
        deltas: np.ndarray, Deltas: np.ndarray) -> RFFQM:
    cols = len(deltas) - 1
    rows = len(Deltas) * 2

    # Nx = cols / 2
    # Ny = rows / 2

    def F1(n):
        return deltas[n]

    def F2(n):
        return -deltas[n]

    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0

    i = np.arange(rows // 2)
    ls[1::2] += L0 * Deltas[i]

    angles_left, angles_bottom = create_miura_angles(ls, cs, theta)
    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)
    return ori


def create_F_from_list(Fs: np.ndarray) -> AnglesFuncType:
    def F(x):
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, 'float64'):
                x = x.astype('int')
        return Fs[x]

    return F


def create_MM_from_list(MMs: np.ndarray) -> AnglesFuncType:
    def MM(y):
        return MMs[y]

    return MM


def get_pert_list_by_func(delta_func, Delta_func, Nx, Ny) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(0, 1 + 0.1 / Nx, 1 / (Nx * 2))
    assert len(xs) == 2 * Nx + 1
    ys = np.arange(0, 1, 1 / Ny)
    assert len(ys) == Ny

    return xs, delta_func(xs), ys, Delta_func(ys)


def plot_perturbations(axes, delta, Delta, Nx, Ny):
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)

    deltas = delta(xs)
    Deltas = Delta(ys)

    axes[0].plot(xs, deltas)
    axes[1].plot(ys, Deltas)
    xs, deltas, ys, Deltas = get_pert_list_by_func(delta, Delta, Nx, Ny)
    plot_perturbations_by_list(axes, xs, deltas, ys, Deltas)


def plot_perturbations_by_list(axes, xs, deltas, ys, Deltas):
    axes[0].plot(xs, deltas, '.')
    axes[0].set_xlabel(r'$ x $')
    axes[0].set_ylabel(r'$ \tilde{\delta}(x)$')
    axes[1].plot(ys, Deltas, '.')
    axes[1].set_xlabel(r'$ y $')
    axes[1].set_ylabel(r'$ \tilde{\Delta}(y)$')
