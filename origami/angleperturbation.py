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
"""
from typing import Callable, Tuple

import numpy as np

AnglesFuncType = Callable[[np.ndarray | float | int, np.ndarray | float | int], Tuple[np.ndarray, np.ndarray]]


def set_perturbations_by_func(func: AnglesFuncType, angles_left: np.ndarray, angles_bottom: np.ndarray,
                              which_angle: str = 'delta+eta', which_boundary: str = 'left+bottom'):
    bottom_range = np.arange(angles_bottom.shape[1]) + 1
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


def set_perturbations_by_func_v1(F: Callable, G: Callable, C: float,
                                 angles_left, angles_bottom, ls, cs,
                                 which_angle='delta+eta', which_boundary='left+bottom'):
    func_v1 = _create_func_v1(F, G, C)
    set_perturbations_by_func(func_v1, angles_left, angles_bottom, which_angle, which_boundary)


def _create_func_v1(F, G, C) -> AnglesFuncType:
    def func_v1(xs, ys):
        deltas = F(xs) + G(ys)
        etas = F(xs) - G(ys) + C
        return deltas, etas

    return func_v1


def create_angles_func(F, G) -> AnglesFuncType:
    def func(xs, ys):
        deltas = F(xs) + G(ys)
        etas = F(xs) - G(ys)
        return deltas, etas

    return func


def create_angles_func_vertical_alternation(F1, F2) -> AnglesFuncType:
    def func(xs, ys):
        deltas = F1(xs) * (1 - ys % 2) + F2(xs) * (ys % 2)
        etas = F1(xs) * (ys % 2) + F2(xs) * (1 - ys % 2)

        return deltas, etas

    return func
