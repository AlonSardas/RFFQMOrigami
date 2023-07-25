import numpy as np

from origami.RFFQMOrigami import RFFQM
from origami.angleperturbation import create_angles_func_vertical_alternation, set_perturbations_by_func
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles
from origami.utils.plotutils import imshow_with_colorbar

sin, cos, tan = np.sin, np.cos, np.tan
cot = lambda x: 1 / tan(x)
csc = lambda x: 1 / sin(x)
sec = lambda x: 1 / cos(x)


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
    FF = lambda x: F(x * 2)
    dFF = lambda x: FF(x + 0.5) - FF(x - 0.5)
    dMM = lambda y: MM(y + 1) - MM(y)
    ddMM = lambda y: dMM(y + 1) - dMM(y)
    return FF, dFF, dMM, ddMM


def create_perturbed_origami(angle, rows, cols, L0, C0, F, MM) -> RFFQM:
    if F is None:
        F = lambda x: 0 * x
    if MM is None:
        MM = lambda y: 0 * y

    dMM = lambda y: MM(y + 1) - MM(y)
    ddMM = lambda y: dMM(y + 1) - dMM(y)

    F1 = lambda x: F(x)
    F2 = lambda x: -F(x)

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
