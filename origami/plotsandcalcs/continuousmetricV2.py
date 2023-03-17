"""
We found an expression that describes the curvature in the linear regime of the perturbations
Here we want to verify numerically these results.
"""
import numpy as np

from origami.RFFQMOrigami import RFFQM
from origami.interactiveplot import plot_interactive
from origami.angleperturbation import set_perturbations_by_func_v1
from origami.marchingalgorithm import create_miura_angles, MarchingAlgorithm
from origami.quadranglearray import dots_to_quadrangles, plot_flat_quadrangles


def test_F():
    """
    We expect to have constant curvature when F grows parabolically
    here we examine that
    """
    F = lambda x: x ** 2 / 10000
    G = lambda y: 0

    angle = np.pi-0.2

    rows = 50
    cols = 100
    ls = np.ones(rows)
    cs = np.ones(cols)

    angles_left, angles_bottom = create_miura_angles(ls, cs, np.pi - angle)

    set_perturbations_by_func_v1(F, G, 0, angles_left, angles_bottom, ls, cs)

    marching = MarchingAlgorithm(angles_left, angles_bottom)
    quads = dots_to_quadrangles(*marching.create_dots(ls, cs))

    fig, _ = plot_flat_quadrangles(quads)
    ori = RFFQM(quads)
    plot_interactive(ori)


def main():
    test_F()


if __name__ == '__main__':
    main()
