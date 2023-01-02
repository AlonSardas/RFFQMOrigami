"""
We performed linear approximation to the marching algorithm. The perturbation
is relative to the classic Miura-Ori
By the approximation we got a recurrence relation for the perturbed angles.
We assumed the angles vary slowly in neighboring panels. This enables us to
get a differential equation to the perturbation angles. The solution for the
equation is:
delta   = F(x)+G(y)
eta     = F(x)-G(y)
where F,G are general functions

Here we try to test this result
"""
import os.path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import origami
from origami.RFFQMOrigami import RFFQM
from origami.RFFQMexamples import plot_interactive
from origami.marchingalgorithm import MarchingAlgorithm, create_miura_angles, IncompatibleError
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles
from origami.utils import plotutils

FIGURES_PATH = os.path.join(origami.BASE_PATH, 'RFFQM/Compatibility/Figures/continuous-angles/')


def test_continuous_perturbations():
    # F = lambda x: np.sin(x / 50) / 100
    F = lambda x: (np.sin(x / 12)) / 150
    G = lambda y: 0  # +(np.sin(y / 12)) / 150
    C = 0.0

    angle = 1

    rows = 40
    cols = 100
    ls = np.ones(rows)
    # ls = np.linspace(1, 17, rows)
    cs = np.ones(cols) * 1
    # cs[1::2] *= 5

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(F, G, C, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)
    plot_interactive(ori)


def test_continuous_perturbations_2steps():
    # F = lambda x: np.sin(x / 50) / 100
    F = lambda x: (np.sin(x / 12)+0.3) / 70 * (x % 2)
    G = lambda y: +(np.cos(y / 12)+0.2) / 50 * (y % 2)

    angle = 1

    rows = 30
    cols = 30
    ls = np.ones(rows)
    ls[::2] *=1.5
    # ls = np.linspace(1, 17, rows)
    cs = np.ones(cols) * 1
    # cs[1::2] *= 5

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)
    plot_interactive(ori)


def test_non_continuous_perturbation():
    angle = 1
    ls = np.ones(100) * 1
    cs = np.ones(100) * 1

    Fs = (4 * np.random.random(len(cs) + 1) - 2) ** 3 / 20
    Gs = (4 * np.random.random(len(ls) + 1) - 2) ** 3 / 20
    F = lambda x: Fs[x]
    G = lambda y: Gs[y]
    C = 0
    func_v1 = _create_func_v1(F, G, C)

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

    set_perturbations_by_func(func_v1, angles_left, angles_bottom)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    print(f'Got a valid pattern')

    fig1, axes1, fig2, axes2 = _compare_calculated_to_exact(
        ls, cs, angle, marching, func_v1)
    fig1.savefig(os.path.join(FIGURES_PATH, 'PDE-random-calculated-and-expected.png'))
    fig2.savefig(os.path.join(FIGURES_PATH, 'PDE-random-delta-eta-sum-diff.png'))


def set_perturbations_by_func_v1(F: Callable, G: Callable, C: float,
                                 angles_left, angles_bottom, ls, cs,
                                 which_angle='delta+eta', which_boundary='left+bottom'):
    func_v1 = _create_func_v1(F, G, C)
    set_perturbations_by_func(func_v1, angles_left, angles_bottom, which_angle, which_boundary)


def set_perturbations_by_func(func: Callable, angles_left: np.ndarray, angles_bottom: np.ndarray,
                              which_angle: str = 'delta+eta', which_boundary: str = 'left+bottom'):
    bottom_range = np.arange(angles_bottom.shape[1])
    left_range = np.arange(angles_left.shape[1])

    deltas_bottom, etas_bottom = func(bottom_range, 0)
    deltas_left, etas_left = func(0, left_range)

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

        set_perturbations_by_func_v1(F, G, C, angles_left, angles_bottom, ls, cs)

        print(angles_left)
        print(angles_bottom)
        print(f'F(0) should be constant {(angles_left[0, :] + angles_left[1, :]) / 2}')
        print(f'G(i) should be {(angles_left[0, :] - angles_left[1, :]) / 2}')
        print(f'F(j) should be {(angles_bottom[0, :] + angles_bottom[1, :]) / 2}')
        print(f'G(0) should be constant {(angles_bottom[0, :] - angles_bottom[1, :]) / 2}')

        MarchingAlgorithm(angles_left, angles_bottom)
        print(f'Got a valid pattern for {s}')


def debug_approximation_validity():
    angle = 1
    ls = np.ones(200) * 1
    cs = np.ones(200) * 1

    s = 0.06
    F = lambda x: s * (np.cos(x / 20) + 1)
    G = lambda y: 0.7 * s * np.cos(y / 25)
    C = -3 * s

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    angles_left = np.array(angles_left, dtype='float128')
    angles_bottom = np.array(angles_bottom, dtype='float128')

    set_perturbations_by_func_v1(F, G, C, angles_left, angles_bottom, ls, cs)

    print(angles_left)
    print(angles_bottom)
    print(f'F(0) should be constant {(angles_left[0, :] + angles_left[1, :]) / 2}')
    print(f'G(i) should be {(angles_left[0, :] - angles_left[1, :]) / 2}')
    print(f'F(j) should be {(angles_bottom[0, :] + angles_bottom[1, :]) / 2}')
    print(f'G(0) should be constant {(angles_bottom[0, :] - angles_bottom[1, :]) / 2}')

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    print(f'Got a valid pattern for {s}')

    print(marching.alphas.dtype)

    Fs = np.append(angles_left[0, 0] + angles_left[1, 0], angles_bottom[0, :] + angles_bottom[1, :]) / 2
    Gs = (angles_left[0, :] - angles_left[1, :]) / 2
    expected_alphas = Fs[np.newaxis, :] + Gs[:, np.newaxis]
    fig1, ax = plt.subplots()
    _imshow_with_colorbar(fig1, ax, expected_alphas - marching.alphas, 'alpha comparison')

    fig2, axes2 = plt.subplots(1, 2, sharey=True)
    _imshow_with_colorbar(fig2, axes2[0], marching.alphas + marching.betas, r'calculated $ \alpha+\beta $')
    _imshow_with_colorbar(fig2, axes2[1], marching.alphas - marching.betas, r'calculated $ \alpha-\beta $')


def test_no_approximation():
    """
    We see if the field
    alpha   = F(x) + G(y)
    beta    = F(x) - G(y)

    is a valid angles fields
    """
    angle = 1
    ls = np.ones(100) * 1
    cs = np.ones(100) * 1

    scales = np.arange(1, 10) / 50
    for s in scales:
        F = lambda x: s * (np.cos(x / 20) + 1)
        G = lambda y: s * (0.7 * np.cos(y / 25) - 2)

        def func(xs, ys):
            # sign = (2 * (xs % 2) - 1)
            sign = 1
            deltas = sign * (F(xs) + G(ys))
            etas = sign * (F(xs) - G(ys))

            return deltas, etas

        angles_left, angles_bottom = create_miura_angles(ls, cs, angle)

        set_perturbations_by_func(func, angles_left, angles_bottom)

        print(f'F(0) should be {(angles_left[0, :] + angles_left[1, :]) / 2}')
        print(f'G(i) should be {(angles_left[0, :] - angles_left[1, :]) / 2}')
        print(f'F(j) should be {(angles_bottom[0, :] + angles_bottom[1, :]) / 2}')
        print(f'G(0) should be {(angles_bottom[0, :] - angles_bottom[1, :]) / 2}')

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

        set_perturbations_by_func_v1(F, G, C, angles_left, angles_bottom, ls, cs, 'delta')
        set_perturbations_by_func_v1(F, G, C, angles_left, angles_bottom, ls, cs, 'eta', 'left')
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

    set_perturbations_by_func_v1(F, G, C, angles_left, angles_bottom, ls, cs)
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
    func_v1 = _create_func_v1(F, G, C)

    angle = 1

    ls = np.ones(200 - 1) * 1
    cs = np.ones(150 - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    set_perturbations_by_func_v1(F, G, C, angles_left, angles_bottom, ls, cs)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    fig1, axes1, fig2, axes2 = _compare_calculated_to_exact(
        ls, cs, angle, marching, func_v1)
    fig1.savefig(os.path.join(FIGURES_PATH, 'PDE-V1-calculated-and-expected.png'))
    fig2.savefig(os.path.join(FIGURES_PATH, 'PDE-V1-delta-eta-sum-diff.png'))


def _compare_calculated_to_exact(ls, cs, angle, marching: MarchingAlgorithm, func):
    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    miura = MarchingAlgorithm(angles_left, angles_bottom)
    bottom_range = np.arange(miura.cols)
    left_range = np.arange(miura.rows)

    xs, ys = np.meshgrid(bottom_range, left_range)

    expected_deltas, expected_etas = func(xs, ys)

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

    fig1, axes1 = plt.subplots(2, 2, sharex=True, sharey=True)
    fig1: Figure = fig1
    _imshow_with_colorbar(fig1, axes1[0, 0], calculated_deltas, r'Calculated $ \delta $')
    _imshow_with_colorbar(fig1, axes1[0, 1], expected_deltas, r'Expected $ \delta $')
    _imshow_with_colorbar(fig1, axes1[1, 0], calculated_etas, r'Calculated $ \eta $')
    _imshow_with_colorbar(fig1, axes1[1, 1], expected_etas, r'Expected $ \eta $')

    fig1.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, sharey=True)
    _imshow_with_colorbar(fig2, axes2[0], calculated_deltas + calculated_etas, r'calculated $ \delta+\eta $')
    _imshow_with_colorbar(fig2, axes2[1], calculated_deltas - calculated_etas, r'calculated $ \delta-\eta $')

    fig2.tight_layout()
    return fig1, axes1, fig2, axes2


def test_decay_term():
    def func(xs, ys):
        deltas = np.cos(ys / 20) / 1000
        etas = np.cos(ys / 20) / 1000
        return deltas, etas

    angle = 1

    ls = np.ones(50 - 1) * 1
    cs = np.ones(50 - 1) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    miura = MarchingAlgorithm(angles_left, angles_bottom)

    set_perturbations_by_func(func, angles_left, angles_bottom)
    marching = MarchingAlgorithm(angles_left, angles_bottom)

    calculated_deltas = marching.alphas - miura.alphas
    calculated_etas = marching.betas - miura.betas

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    _imshow_with_colorbar(fig, axes[0, 0], calculated_deltas, r'Calculated $ \delta $')
    _imshow_with_colorbar(fig, axes[1, 0], calculated_etas, r'Calculated $ \eta $')


def test_no_miura_general_func():
    F = lambda x: 1.2 + 0.3 * np.cos(x / 2)
    G = lambda y: 0.3 + 0.2 * np.sin(y / 0.6)

    rows, cols = 50, 70
    ls = np.ones(cols)
    cs = np.ones(rows) * 10
    left_range = np.arange(0, cols + 1)
    bottom_range = np.arange(1, rows + 1)

    angles_left = np.zeros((2, len(left_range)))
    angles_bottom = np.zeros((2, len(bottom_range)))
    angles_left[0, :] = F(0) + G(left_range)
    angles_left[1, :] = F(0) - G(left_range)
    angles_bottom[0, :] = F(bottom_range) + G(0)
    angles_bottom[1, :] = F(bottom_range) - G(0)
    marching = MarchingAlgorithm(angles_left, angles_bottom)
    print('Got a valid pattern!')
    # quads = dots_to_quadrangles(*marching.create_dots(ls, cs))
    # fig, _ = plot_flat_quadrangles(quads)
    fig2, axes2 = plt.subplots(1, 2, sharey=True)
    _imshow_with_colorbar(fig2, axes2[0], marching.alphas + marching.betas, r'calculated $ \alpha+\beta $')
    _imshow_with_colorbar(fig2, axes2[1], marching.alphas - marching.betas, r'calculated $ \alpha-\beta $')

    calculated_F = np.mean(marching.alphas + marching.betas, axis=0) / 2
    fig, axes = plt.subplots(1, 2)
    ax: Axes = axes[0]
    ax.plot(np.append(0, bottom_range), calculated_F, '.')
    smooth_xs = np.linspace(0, bottom_range[-1], 200)
    ax.plot(smooth_xs, F(smooth_xs))

    calculated_G = np.mean(marching.alphas - marching.betas, axis=1) / 2
    ax: Axes = axes[1]
    ax.plot(left_range, calculated_G, '.')
    smooth_xs = np.linspace(0, left_range[-1], 300)
    ax.plot(smooth_xs, G(smooth_xs))


def test_other_angles_field():
    """
    Test the hypothesis that if delta=eta=F(x,y)
    and delta, eta alternate signs for each row, then the pattern is compatible
    """
    angle = 1
    ls = np.ones(200) * 1
    cs = np.ones(200) * 1

    angles_left, angles_bottom = create_miura_angles(ls, cs, angle)
    # angles_left[:, :] = angle
    # angles_bottom[:, :] = angle
    angles_left = np.array(angles_left, dtype='float128')
    angles_bottom = np.array(angles_bottom, dtype='float128')

    def func(xs, ys):
        sign = (2 * (ys % 2) - 1)
        # sign = 1
        F = lambda x, y: 0.6 * np.exp(-(-x + 50 - y) ** 2 / 400) + 0.02 * np.exp(-(-x + 100 + 1 / 2 * y) ** 2 / 400)
        # F = lambda x, y: 0.02 * np.exp(-(-x+50+y)**2/20)
        deltas = sign * F(xs, ys)
        etas = -sign * F(xs, ys)

        return deltas, etas

    miura = MarchingAlgorithm(angles_left, angles_bottom)

    set_perturbations_by_func(func, angles_left, angles_bottom)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    print(f'Got a valid pattern!!!')

    calculated_deltas = marching.alphas - miura.alphas
    calculated_deltas[::2, :] *= -1

    calculated_etas = marching.betas - miura.alphas
    calculated_etas[::2, :] *= -1

    fig1, axes = plt.subplots(1, 2)
    _imshow_with_colorbar(fig1, axes[0], calculated_deltas, 'delta calculated')
    _imshow_with_colorbar(fig1, axes[1], calculated_deltas, 'eta calculated')

    fig2, ax = plt.subplots()
    xs = np.arange(marching.cols)
    ys = np.arange(marching.rows)
    _imshow_with_colorbar(fig2, ax, func(xs[np.newaxis, :], ys[:, np.newaxis])[0], 'Expected')


def _imshow_with_colorbar(fig: Figure, ax: Axes, data: np.ndarray, ax_title):
    im = ax.imshow(data)
    plotutils.create_colorbar(fig, ax, im)
    ax.set_title(ax_title)
    ax.invert_yaxis()


def _create_func_v1(F, G, C) -> Callable:
    def func_v1(xs, ys):
        deltas = F(xs) + G(ys)
        etas = F(xs) - G(ys) + C
        return deltas, etas

    return func_v1


def main():
    # test_non_continuous_perturbation()
    test_continuous_perturbations_2steps()
    # test_crease_size()
    # compare_to_expected()
    # debug_approximation_validity()
    # test_decay_term()
    # test_no_approximation()
    # test_no_miura_general_func()
    # test_other_angles_field()
    plt.show()


if __name__ == '__main__':
    main()
