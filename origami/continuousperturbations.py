"""
We performed linear approximation to the marching algorithm. The perturbation
is relative to the classic Miura-Ori
By the approximation we got a recurrence relation for the perturbed angles.
We assumed the angles vary slowly in neighboring panels. This enables us to
get a differential equation to the perturbation angles. The solution for the
equation is:
delta   = F(x)+G(y)
eta     = F(x)-G(y)+C
where F,G are general functions and C is constant

Here we try to test this result
"""
import os.path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import origami
from origami.RFFQMOrigami import RFFQM
from origami.RFFQMexamples import plot_interactive
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles, IncompatibleError
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles

FIGURES_PATH = os.path.join(origami.BASE_PATH, 'RFFQM/Compatibility/Figures/continuous-angles/')


def test_continuous_perturbations():
    # F = lambda x: np.sin(x / 50) / 100
    F = lambda x: 0
    G = lambda y: np.sin(y / 20) / 50
    C = 0.0

    angle = 1

    rows = 80
    # ls = np.ones(64)
    ls = np.linspace(1, 17, rows)
    cs = np.ones(20) * 10
    cs[1::2] *= 5

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func(F, G, C, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)
    plot_interactive(ori)


def set_perturbations_by_func(F: Callable, G: Callable, C: float,
                              angles_left, angles_bottom, ls, cs,
                              which_angle='delta+eta', which_boundary='left+bottom'):
    bottom_range = np.arange(len(cs))
    left_range = np.arange(len(ls) + 1)

    deltas_bottom = F(bottom_range) + G(0)
    etas_bottom = F(bottom_range) - G(0) + C
    deltas_left = F(0) + G(left_range)
    etas_left = F(0) - G(left_range) + C

    if 'delta' in which_angle:
        if 'bottom' in which_boundary:
            angles_bottom[0, :] += deltas_bottom
        if 'left' in which_boundary:
            angles_left[0, :] += deltas_left

    if 'eta' in which_angle:
        if 'bottom' in which_boundary:
            angles_bottom[1, :] += etas_bottom
        if 'left' in which_boundary:
            angles_left[1, :] += etas_left


def test_approximation_validity():
    """
    We see how big can the perturbation angles be to still be a valid input data
    """
    angle = 1
    ls = np.ones(200) * 1
    cs = np.ones(200) * 1

    scales = np.arange(1, 10) / 100
    for s in scales:
        F = lambda x: s * (np.cos(x / 20) + 1)
        G = lambda y: 0.7 * s * np.cos(y / 25)
        C = -3 * s

        angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

        set_perturbations_by_func(F, G, C, angles_left, angles_bottom, ls, cs)
        MarchingAlgorithm(angles_left, angles_bottom)
        print(f'Got a valid pattern for {s}')


def test_crease_size():
    """
    Setting delta at the bottom determines F
    setting delta at the left boundary determines G
    altogether, eta is by now is determined up to a constant.
    Here we check what if there is a mismatch for eta
    """
    F = lambda x: np.cos(x / 20) / 100 + 0.01
    G = lambda y: np.cos(y / 25) / 70
    C = -0.03

    angle = 1

    sizes = np.array([50, 60, 70, 80, 90, 100])
    for size in sizes:
        ls = np.ones(size) * 1
        cs = np.ones(size) * 1

        angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

        set_perturbations_by_func(F, G, C, angles_left, angles_bottom, ls, cs, 'delta')
        set_perturbations_by_func(F, G, C, angles_left, angles_bottom, ls, cs, 'eta', 'left')
        # Making the input data invalid by setting deltas to 0
        print(f'start marching algorithm for bad input data with size {size}X{size}')
        try:
            MarchingAlgorithm(angles_left, angles_bottom)
        except IncompatibleError as e:
            print('The input data is indeed invalid as expected')
            print(e.args)
        else:
            print('The invalid input data yielded a valid crease pattern!')

    print('-------------------')
    print("Checking validity, this takes a few seconds")
    ls = np.ones(1000) * 1
    cs = np.ones(1000) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func(F, G, C, angles_left, angles_bottom, ls, cs)
    print('start marching algorithm')
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    print('end marching algorithm')
    print('Valid angles!')


def compare_to_expected():
    """
    Here we compare the angles calculated by the marching algorithm to the
    angles we expect to have based on the functions F,G
    """
    F = lambda x: np.sin(x / 20) / 100
    G = lambda y: np.exp(-(y - 50) ** 2 / 100) * 0.02 + np.exp(-(y - 120) ** 2 / 300) * 0.03
    C = -0.03

    angle = 1

    ls = np.ones(200 - 1) * 1
    cs = np.ones(150 - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    miura = MarchingAlgorithm(angles_left, angles_bottom)

    set_perturbations_by_func(F, G, C, angles_left, angles_bottom, ls, cs)
    marching = MarchingAlgorithm(angles_left, angles_bottom)

    bottom_range = np.arange(miura.cols)
    left_range = np.arange(miura.rows)

    xs, ys = np.meshgrid(bottom_range, left_range)

    expected_deltas = F(xs) + G(ys)
    expected_etas = F(xs) - G(ys) + C

    calculated_deltas = marching.alphas - miura.alphas
    calculated_etas = marching.betas - miura.betas

    max_diff_deltas = np.max(np.abs(expected_deltas - calculated_deltas))
    max_diff_etas = np.max(np.abs(expected_etas - calculated_etas))

    deltas_relative_errors = np.abs((expected_deltas - calculated_deltas) / calculated_deltas)
    etas_relative_errors = np.abs((expected_etas - calculated_etas) / calculated_etas)

    print(f"Pattern size: {miura.rows}X{miura.cols}")
    print(f"Max delta: {np.abs(calculated_deltas).max()}, max diff: {max_diff_deltas}")
    print(f"Max eta: {np.abs(calculated_etas).max()}, max diff: {max_diff_etas}")
    print(f"delta relative error: {deltas_relative_errors.max()}")
    print(f"eta relative error: {etas_relative_errors.max()}")

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    fig: Figure = fig
    _imshow_with_colorbar(axes[0, 0], calculated_deltas, r'Calculated $ \delta $')
    _imshow_with_colorbar(axes[0, 1], expected_deltas, r'Expected $ \delta $')
    _imshow_with_colorbar(axes[1, 0], calculated_etas, r'Calculated $ \eta $')
    _imshow_with_colorbar(axes[1, 1], expected_etas, r'Expected $ \eta $')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'PDE-V1-calculated-and-expected.png'))

    fig, axes = plt.subplots(1, 2, sharey=True)
    _imshow_with_colorbar(axes[0], calculated_deltas + calculated_etas, r'calculated $ \delta+\eta $')
    _imshow_with_colorbar(axes[1], calculated_deltas - calculated_etas, r'calculated $ \delta-\eta $')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_PATH, 'PDE-V1-delta-eta-sum-diff.png'))

    # quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    # fig, _ = plot_flat_quadrangles(quads)


def _imshow_with_colorbar(ax: Axes, data: np.ndarray, ax_title):
    im = ax.imshow(data)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.15)
    plt.colorbar(im, cax=cax)

    ax.set_title(ax_title)
    ax.invert_yaxis()


def main():
    test_continuous_perturbations()
    # test_crease_size()
    # compare_to_expected()
    # test_approximation_validity()
    plt.show()


if __name__ == '__main__':
    main()
