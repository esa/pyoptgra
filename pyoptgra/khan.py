# Copyright 2008, 2021 European Space Agency
#
# This file is part of pyoptgra, a pygmo affiliated library.
#
# This Source Code Form is available under two different licenses.
# You may choose to license and use it under version 3 of the
# GNU General Public License or under the
# ESA Software Community Licence (ESCL) 2.4 Weak Copyleft.
# We explicitly reserve the right to release future versions of
# Pyoptgra and Optgra under different licenses.
# If copies of GPL3 and ESCL 2.4 were not distributed with this
# file, you can obtain them at https://www.gnu.org/licenses/gpl-3.0.txt
# and https://essr.esa.int/license/european-space-agency-community-license-v2-4-weak-copyleft

from typing import Callable, List, Optional

import numpy as np
from pygmo import estimate_gradient_h
from scipy.optimize import root_scalar


class base_khan_function:
    r"""Base class for a function to smothly enforce optimisation parameter bounds as Michal Khan
    used to do:

    .. math::

        x = \frac{x_{max} + x_{min}}{2} + \frac{x_{max} - x_{min}}{2} \cdot \f(x_{khan})

    Where :math:`x` is the pagmo decision vector and :math:`x_{khan}` is the decision vector
    passed to OPTGRA. In this way parameter bounds are guaranteed to be satisfied, but the gradients
    near the bounds approach zero.

    The child class needs to implement the methods `_eval`, `_eval_inv`, `_eval_grad` and
    `_eval_inv_grad`
    """  # noqa: W605

    def __init__(self, lb: List[float], ub: List[float]):
        """Constructor

        Parameters
        ----------
        lb : List[float]
            Lower pagmo parameter bounds
        ub : List[float]
            Upper pagmo parameter bounds
        """
        self._lb = np.asarray(lb)
        self._ub = np.asarray(ub)
        self._nx = len(lb)

        # determine finite lower/upper bounds\
        def _isfinite(a: np.ndarray):
            """Custom _ function"""
            almost_infinity = 1e300
            return np.logical_and(np.isfinite(a), np.abs(a) < almost_infinity)

        finite_lb = _isfinite(self._lb)
        finite_ub = _isfinite(self._ub)

        # we only support cases where both lower and upper bounds are finite if given
        check = np.where(finite_lb != finite_ub)[0]
        if any(check):
            raise ValueError(
                "When using Khan bounds, both lower and upper bound for bounded parameters "
                "must be finite."
                f"Detected mismatch at decision vector indices: {check}"
            )

        # also exclude parameters where lower and upper bounds are identical
        with np.errstate(invalid="ignore"):
            # we ignore RuntimeWarning: invalid value encountered in subtract
            nonzero_diff = abs(self._lb - self._ub) > 1e-9

        # store the mask of finite bounds
        self.mask = np.logical_and(finite_ub, nonzero_diff)
        self._lb_masked = self._lb[self.mask]
        self._ub_masked = self._ub[self.mask]

    def _apply_to_subset(
        self, x: np.ndarray, func: Callable, default_result: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply a given function only to a subset of x defined by self.mask."""
        # Create a copy to preserve the original structure
        result = default_result if default_result is not None else x.copy()
        # Apply the function only to the selected subset
        result[self.mask] = func(x[self.mask])
        return result

    def _eval(self, x_khan_masked: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _eval_inv(self, x_masked: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _eval_grad(self, x_khan_masked: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _eval_inv_grad(self, x_masked: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eval(self, x_khan: np.ndarray) -> np.ndarray:
        """Convert :math:`x_{optgra}` to :math:`x`.

        Parameters
        ----------
        x_khan : np.ndarray
            Decision vector passed to OPTGRA

        Returns
        -------
        np.ndarray
            Pagmo decision vector
        """
        return self._apply_to_subset(np.asarray(x_khan), self._eval)

    def eval_inv(self, x: np.ndarray) -> np.ndarray:
        """Convert :math:`x` to :math:`x_{optgra}`.

        Parameters
        ----------
        x : np.ndarray
            Pagmo decision vector

        Returns
        -------
        np.ndarray
            Decision vector passed to OPTGRA

        """
        return self._apply_to_subset(np.asarray(x), self._eval_inv)

    def eval_grad(self, x_khan: np.ndarray) -> np.ndarray:
        """Gradient of ``eval`` function.

        Parameters
        ----------
        x_khan : np.ndarray
            Decision vector passed to OPTGRA

        Returns
        -------
        np.ndarray
            Pagmo decision vector
        """
        return np.diag(
            self._apply_to_subset(np.asarray(x_khan), self._eval_grad, np.ones(self._nx))
        )

    def eval_inv_grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient of ``eval_inv`` method.

        Parameters
        ----------
        x : np.ndarray
            Pagmo decision vector

        Returns
        -------
        np.ndarray
            Decision vector passed to OPTGRA
        """
        return np.diag(self._apply_to_subset(np.asarray(x), self._eval_inv_grad, np.ones(self._nx)))


class khan_function_sin(base_khan_function):
    r"""Function to smothly enforce optimisation parameter bounds as Michal Khan used to do:

    .. math::

        x = \frac{x_{max} + x_{min}}{2} + \frac{x_{max} - x_{min}}{2} \cdot \sin(x_{khan})

    Where :math:`x` is the pagmo decision vector and :math:`x_{khan}` is the decision vector
    passed to OPTGRA. In this way parameter bounds are guaranteed to be satisfied, but the gradients
    near the bounds approach zero.
    """  # noqa: W605

    def __init__(self, lb: List[float], ub: List[float], unity_gradient: bool = True):
        """Constructor

        Parameters
        ----------
        lb : List[float]
            Lower pagmo parameter bounds
        ub : List[float]
            Upper pagmo parameter bounds
        unity_gradient : bool, optional
            Uses an internal scaling that ensures that the derivative of pagmo parameters w.r.t.
            Khan parameters are unity at (lb + ub)/2. By default True.
            Otherwise, the original Khan method is used that can result in strongly modified
            gradients
        """
        # call parent class constructor
        super().__init__(lb, ub)

        # determine coefficients inside the sin function
        self._a = 2 / (self._ub_masked - self._lb_masked) if unity_gradient else 1.0
        self._b = (
            -(self._ub_masked + self._lb_masked) / (self._ub_masked - self._lb_masked)
            if unity_gradient
            else 0.0
        )

    def _eval(self, x_khan_masked: np.ndarray) -> np.ndarray:
        return (self._ub_masked + self._lb_masked) / 2 + (
            self._ub_masked - self._lb_masked
        ) / 2 * np.sin(x_khan_masked * self._a + self._b)

    def _eval_inv(self, x_masked: np.ndarray) -> np.ndarray:
        arg = (2 * x_masked - self._ub_masked - self._lb_masked) / (
            self._ub_masked - self._lb_masked
        )

        clip_value = 1.0 - 1e-8  # avoid boundaries
        if np.any((arg < -clip_value) | (arg > clip_value)):
            print(
                "WARNING: Numerical inaccuracies encountered during khan_function inverse.",
                "Clipping parameters to valid range.",
            )
            arg = np.clip(arg, -clip_value, clip_value)
        return (np.arcsin(arg) - self._b) / self._a

    def _eval_grad(self, x_khan_masked: np.ndarray) -> np.ndarray:
        return (
            (self._ub_masked - self._lb_masked)
            / 2
            * np.cos(self._a * x_khan_masked + self._b)
            * self._a
        )

    def _eval_inv_grad(self, x_masked: np.ndarray) -> np.ndarray:
        return (
            -1
            / self._a
            / (
                (self._lb_masked - self._ub_masked)
                * np.sqrt(
                    ((self._lb_masked - x_masked) * (x_masked - self._ub_masked))
                    / (self._ub_masked - self._lb_masked) ** 2
                )
            )
        )


class khan_function_tanh(base_khan_function):
    r"""Function to smothly enforce optimisation parameter bounds using the hyperbolic tangent:

    .. math::

        x = \frac{x_{max} + x_{min}}{2} + \frac{x_{max} - x_{min}}{2} \cdot \tanh(x_{khan})

    Where :math:`x` is the pagmo decision vector and :math:`x_{khan}` is the decision vector
    passed to OPTGRA. In this way parameter bounds are guaranteed to be satisfied, but the gradients
    near the bounds approach zero.
    """  # noqa: W605

    def __init__(self, lb: List[float], ub: List[float], unity_gradient: bool = True):
        """Constructor

        Parameters
        ----------
        lb : List[float]
            Lower pagmo parameter bounds
        ub : List[float]
            Upper pagmo parameter bounds
        unity_gradient : bool, optional
            Uses an internal scaling that ensures that the derivative of pagmo parameters w.r.t.
            khan parameters are unity at (lb + ub)/2. By default True.
            Otherwise, the original Khan method is used that can result in strongly modified
            gradients
        """
        # call parent class constructor
        super().__init__(lb, ub)

        # define amplification factor to avoid bounds to be only reached at +/- infinity
        amp = 1.0 + 1e-3

        # define the clip value (we avoid the boundaries of the parameters by this much)
        self.clip_value = 1.0 - 1e-6

        # determine coefficients inside the tanh function
        self._diff_masked = amp * (self._ub_masked - self._lb_masked)
        self._sum_masked = self._ub_masked + self._lb_masked
        self._a = 2 / self._diff_masked if unity_gradient else 1.0
        self._b = -self._sum_masked / self._diff_masked if unity_gradient else 0.0

    def _eval(self, x_khan_masked: np.ndarray) -> np.ndarray:
        return self._sum_masked / 2 + self._diff_masked / 2 * np.tanh(
            x_khan_masked * self._a + self._b
        )

    def _eval_inv(self, x_masked: np.ndarray) -> np.ndarray:
        arg = (2 * x_masked - self._sum_masked) / (self._diff_masked)

        if np.any((arg < -self.clip_value) | (arg > self.clip_value)):
            print(
                "WARNING: Numerical inaccuracies encountered during khan_function inverse.",
                "Clipping parameters to valid range.",
            )
            arg = np.clip(arg, -self.clip_value, self.clip_value)
        return (np.arctanh(arg) - self._b) / self._a

    def _eval_grad(self, x_khan_masked: np.ndarray) -> np.ndarray:
        return self._diff_masked / 2 / np.cosh(self._a * x_khan_masked + self._b) ** 2 * self._a

    def _eval_inv_grad(self, x_masked: np.ndarray) -> np.ndarray:

        return (2 * self._diff_masked) / (
            self._a * (self._diff_masked**2 - (2 * x_masked - self._sum_masked) ** 2)
        )


def triangular_wave_fourier(N, x):
    """
    Compute the truncated Fourier series of a triangular wave with normalized amplitude.

    Parameters
    ----------
    N : int
        Number of terms in the Fourier series.
    x : float or np.array
        Input value(s) where the series is evaluated.

    Returns
    -------
    float or np.array
        Approximated value of the triangular wave at x, normalized to unit amplitude.
    """
    result = np.zeros_like(x, dtype=np.float64)
    max_val = 0.0  # for later normalization

    if N == 0:
        return result

    for n in range(N):
        coeff = (-1) ** n / (2 * n + 1) ** 2
        result += coeff * np.sin((2 * n + 1) * x)
        max_val += abs(coeff)

    return result / max_val if max_val > 0 else result


def inverse_triangular_wave(N, y):
    """
    Compute the inverse of the truncated Fourier series of a triangular wave using root finding.

    Parameters
    ----------
    N : int
        Number of terms in the Fourier series.
    y : float or np.ndarray
        Output value for which to find the corresponding x.

    Returns
    -------
    float or np.ndarray
        Value of x such that triangular_wave_fourier(N, x) ≈ y.
    """
    y = np.atleast_1d(y)  # Ensure y is an array
    results = np.zeros_like(y, dtype=np.float64)

    for i, yi in enumerate(y):
        x_guess = np.arcsin(yi)

        def func(x):
            return triangular_wave_fourier(N, x) - yi

        root = root_scalar(func, bracket=[-np.pi / 2, np.pi / 2], x0=x_guess)
        results[i] = root.root if root.converged else np.nan

    return results if y.shape else results.item()


def triangular_wave_fourier_grad(N, x):
    """
    Compute the truncated Fourier series of a triangular wave derivative with normalized amplitude.

    Parameters
    ----------
    N : int
        Number of terms in the Fourier series.
    x : float or np.array
        Input value(s) where the series is evaluated.

    Returns
    -------
    float or np.array
        Approximated value of the triangular wave derivative at x
    """
    result = np.zeros_like(x, dtype=np.float64)
    max_val = 0.0  # for later normalization

    if N == 0:
        return result

    for n in range(N):
        coeff = (-1) ** n / (2 * n + 1)
        result += coeff * np.cos((2 * n + 1) * x)
        max_val += abs(coeff / (2 * n + 1))

    return result / max_val if max_val > 0 else result


class khan_function_triangle(base_khan_function):
    r"""Function to smothly enforce optimisation parameter bounds using a Fourier series of a
    Triangle wave:

    .. math::

        x = \frac{x_{max} + x_{min}}{2} + \frac{x_{max} - x_{min}}{2} \cdot \T(x_{Khan})

    Where :math:`x` is the pagmo decision vector and :math:`x_{khan}` is the decision vector
    passed to OPTGRA. :math:`T(x_{khan})` is the truncated Fourier expansion

    .. math::

        T(x_{khan}) = \frac{8 A}{\pi^2} \sum_{n=0}^{N-1} \frac{(-1)^n}{(2 n + 1)^2} \sin((2 n + 1) x_{Khan})

    In this way parameter bounds are guaranteed to be satisfied, but the gradients
    near the bounds approach zero.
    """  # noqa: W605

    def __init__(
        self, lb: List[float], ub: List[float], order: int = 3, unity_gradient: bool = True
    ):
        """Constructor

        Parameters
        ----------
        lb : List[float]
            Lower pagmo parameter bounds
        ub : List[float]
            Upper pagmo parameter bounds
        order : int, optional
            The truncation order of the Fourier series. ``order=1`` corresponds to a simple sin
            function. By default 3
        unity_gradient : bool, optional
            Uses an internal scaling that ensures that the derivative of pagmo parameters w.r.t.
            Khan parameters are unity at (lb + ub)/2. By default True.
            Otherwise, the original Khan method is used that can result in strongly modified
            gradients
        """
        # call parent class constructor
        super().__init__(lb, ub)
        self.order = int(order)

        # determine coefficients inside the Fourier function
        self._sum_masked = self._ub_masked + self._lb_masked
        self._diff_masked = self._ub_masked - self._lb_masked
        t_grad0 = triangular_wave_fourier_grad(self.order, 0.0)
        self._a = 2 / (self._diff_masked) / t_grad0 if unity_gradient else 1.0
        self._b = -self._a / 2 * (self._sum_masked) if unity_gradient else 0.0

    def _eval(self, x_khan_masked: np.ndarray) -> np.ndarray:
        return (self._sum_masked) / 2 + (self._diff_masked) / 2 * triangular_wave_fourier(
            self.order, x_khan_masked * self._a + self._b
        )

    def _eval_inv(self, x_masked: np.ndarray) -> np.ndarray:
        arg = (2 * x_masked - self._sum_masked) / (self._diff_masked)

        clip_value = 1.0 - 1e-8  # avoid boundaries
        if np.any((arg < -clip_value) | (arg > clip_value)):
            print(
                "WARNING: Numerical inaccuracies encountered during khan_function inverse.",
                "Clipping parameters to valid range.",
            )
            arg = np.clip(arg, -clip_value, clip_value)
        return (inverse_triangular_wave(self.order, arg) - self._b) / self._a

    def _eval_grad(self, x_khan_masked: np.ndarray) -> np.ndarray:
        return (
            (self._diff_masked)
            / 2
            * triangular_wave_fourier_grad(self.order, self._a * x_khan_masked + self._b)
            * self._a
        )

    def _eval_inv_grad(self, x_masked: np.ndarray, dx: float = 1e-7) -> np.ndarray:
        n = len(x_masked)
        return np.diag(
            estimate_gradient_h(lambda x: self._eval_inv(x), x_masked, dx=dx).reshape(n, n)
        )
