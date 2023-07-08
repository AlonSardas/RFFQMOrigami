import inspect

import matplotlib.lines
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy


class FitParams(object):
    def __init__(self, fit_func, xs, ys):
        self.fit_func = fit_func
        self.xs = xs
        self.ys = ys
        self.bounds = (-np.inf, np.inf)

        func_signature = inspect.signature(self.fit_func)
        func_params = func_signature.parameters.keys()
        self.fit_params_names = list(func_params)[1:]  # Remove x param

        if len(xs) == 0:
            raise ValueError('xs data is empty')
        if self._is_ufloat(self.xs[0]) or self._is_ufloat(self.ys[0]):
            self._convert_data_to_ufloat()
            self.should_plot_errors = True
        else:
            self.should_plot_errors = False
        self.additional_params = {}

    def _convert_data_to_ufloat(self):
        # Convert each array to ufloat, with 0 std if none exists
        self.xs = unumpy.uarray(unumpy.nominal_values(self.xs), unumpy.std_devs(self.xs))
        self.ys = unumpy.uarray(unumpy.nominal_values(self.ys), unumpy.std_devs(self.ys))

    def _is_ufloat(self, a):
        return hasattr(a, 'nominal_value')


class FuncFit(object):

    def __init__(self, fit_params: FitParams):
        self.params = fit_params
        self.fit_results: unumpy.uarray = self._fit()

    def _fit(self) -> unumpy.uarray:
        xs = unumpy.nominal_values(self.params.xs)
        ys = unumpy.nominal_values(self.params.ys)
        vals, pcov = curve_fit(self.params.fit_func, xs, ys,
                               bounds=self.params.bounds, **self.params.additional_params)
        diag = np.diag(pcov)
        if np.any(diag < 0):
            print("The errors found are negative. This may happen if there is a redundant "
                  "parameter in the fit function")
        errs = np.sqrt(diag)
        return unumpy.uarray(vals, errs)

    def plot_data(self, ax, label=None) -> matplotlib.lines.Line2D:
        xs = unumpy.nominal_values(self.params.xs)
        ys = unumpy.nominal_values(self.params.ys)
        if self.params.should_plot_errors:
            x_errs = unumpy.std_devs(self.params.xs)
            y_errs = unumpy.std_devs(self.params.ys)
            lines = ax.errorbar(xs, ys, y_errs, x_errs, '.', label=label)
        else:
            lines = ax.plot(xs, ys, '.', label=label)

        return lines[0]

    def plot_fit(self, ax, label=None, x_points=200) -> matplotlib.lines.Line2D:
        x_lim = ax.get_xlim()
        fit_xs = np.linspace(x_lim[0], x_lim[1], x_points)

        fit_ys = self.params.fit_func(fit_xs, *unumpy.nominal_values(self.fit_results))
        lines = ax.plot(fit_xs, fit_ys, '-', label=label)
        return lines[0]

    def plot_residues(self, ax):
        xs = unumpy.nominal_values(self.params.xs)
        ys = unumpy.nominal_values(self.params.ys)

        fitted_values = self.params.fit_func(xs, *unumpy.nominal_values(self.fit_results))
        residues = ys - fitted_values
        lines = ax.plot(xs, residues, '.', label='residues')

        x_lim = ax.get_xlim()
        ax.plot(x_lim, (0, 0), '--')

        return lines[0]

    def print_results(self):
        for index, fit_param in enumerate(self.params.fit_params_names):
            print(f'fit {fit_param}={self.fit_results[index]}')

    def print_results_latex(self):
        for index, fit_param in enumerate(self.params.fit_params_names):
            print(f'fit {fit_param}={self.fit_results[index]:L}')

    def eval_func(self, x):
        return self.params.fit_func(x, *self.fit_results)
