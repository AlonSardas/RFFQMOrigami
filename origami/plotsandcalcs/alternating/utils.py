from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

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


def create_perturbed_origami(angle, chi, xi, L0, C0,
                             delta: Optional[AnglesFuncType], DeltaL: Optional[AnglesFuncType]) -> RFFQM:
    if delta is None:
        def delta(x): return 0 * x
    if DeltaL is None:
        def DeltaL(y): return 0 * y

    def F1(n):
        return delta(n / 2 * chi)

    def F2(n):
        return -delta(n / 2 * chi)

    N_x = int(1 / chi)
    N_y = int(1 / xi)
    rows = 2 * N_y
    cols = 2 * N_x

    ls = np.ones(rows) * L0 * xi
    cs = np.ones(cols) * C0 * chi

    i = np.arange(N_y)
    ls[1::2] += xi * DeltaL(i * xi)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    pert_func = create_angles_func_vertical_alternation(F1, F2)
    set_perturbations_by_func(pert_func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    ori = RFFQM(quads)
    return ori


def create_perturbed_origami_by_list(
        angle, L0, C0,
        deltas: np.ndarray, DeltaLs: np.ndarray) -> RFFQM:
    cols = len(deltas) - 1
    rows = len(DeltaLs) * 2
    Nx = cols / 2
    Ny = rows / 2

    # if len(deltas) != cols + 1:
    #     raise ValueError(f'The length of deltas should be {cols + 1}, got {len(deltas)}')
    # if len(DeltaLs) != rows // 2:
    #     raise ValueError(f'The length of deltas should be {cols + 1}, got {len(deltas)}')

    def F1(n):
        return deltas[n]

    def F2(n):
        return -deltas[n]

    ls = np.ones(rows) * L0 / Ny
    cs = np.ones(cols) * C0 / Nx

    i = np.arange(rows // 2)
    ls[1::2] += DeltaLs[i] / Ny

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
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
