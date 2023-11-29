from typing import Optional

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


def create_perturbed_origami(angle, rows, cols, L0, C0, 
                             F: Optional[AnglesFuncType], MM: Optional[AnglesFuncType]) -> RFFQM:
    if F is None:
        def F(x): return 0 * x
    if MM is None:
        def MM(y): return 0 * y

    def dMM(y): return MM(y + 1) - MM(y)
    def ddMM(y): return dMM(y + 1) - dMM(y)

    def F1(x): return F(x)
    def F2(x): return -F(x)

    ls = np.ones(rows) * L0
    cs = np.ones(cols) * C0

    ys = np.arange(len(ls) // 2)
    # ls[1::2] = L0 + dMM(ys) + 1 / 2 * ddMM(ys)
    ls[1::2] = L0 + dMM(ys)

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
