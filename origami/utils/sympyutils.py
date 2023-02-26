"""
Utils especially for parsing latex to sympy
"""
from functools import reduce
from typing import Iterable

from sympy import *
# Initializing session seems unnecessary, and it causes bugs
# sympy.init_session()
from sympy.parsing.latex import parse_latex

SHOULD_SUB_IMAGINARY = True


def from_latex(tex):
    if tex.isspace():
        print('Got blank input, assuming 0')
        return 0

    if tex.startswith(r'\begin{bmatrix}'):
        return parse_matrix(tex)

    expr = parse_latex(tex)
    pi_sym = Symbol('pi')
    e_sym = Symbol('e')
    expr = expr.subs({pi_sym: pi, e_sym: E})
    if SHOULD_SUB_IMAGINARY:
        expr = sub_imaginary(expr)
    return expr


def sub_imaginary(expr):
    i_sym = Symbol('i')
    return expr.subs(i_sym, I)


def parse_matrix(tex):
    r"""
    \begin{bmatrix}x+3 & -4 & 0 & 0\\
    9 & x-9 & 0 & 0\\
    0 & 0 & x+4 & 7\\
    0 & 0 & -7 & x-10
    \end{bmatrix}
    """
    tex = tex.replace('\n', '')
    tex = tex.replace(r'\begin{bmatrix}', '')
    tex = tex.replace(r'\end{bmatrix}', '')
    lines = tex.split(r'\\')
    cells = [list(map(from_latex, line.split('&'))) for line in lines]
    return Matrix(cells)


def read_latex():
    text = input('Write latex: ')
    return from_latex(text)


def np_matrix_to_latex(mat):
    import numpy as np
    precision = np.get_printoptions()['precision']
    s = "\\\\\n".join([" & ".join(
        map(lambda x: np.format_float_positional(x, precision=precision), line))
        for line in mat])
    s = r'\begin{bmatrix}' + s + '\n' + r'\end{bmatrix}'
    return s


def linearize_multivariable(expr: Expr | Matrix, variables: Iterable[Symbol]) -> Expr | Matrix:
    """
    Apparently, taking only linear term with several variables is hard.
    There is a mathematical trick, see
    https://community.wolfram.com/groups/-/m/t/572931
    """
    # For some reason, series doesn't work on matrices. This is a workaround
    if isinstance(expr, Matrix):
        for i, v in enumerate(expr):
            expr[i] = linearize_multivariable(expr[i], variables)
        return expr

    t = Symbol("temp_symbol")
    # Replace all variables: [x, y] -> [tx, ty]
    expr = reduce(lambda ex, s: ex.subs(s, t * s), variables, expr)
    linearized = series(expr, t, 0, n=2)
    expr = linearized.removeO().subs(t, 1)
    return expr
