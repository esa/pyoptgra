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

import multiprocessing as mp
from typing import Any, Callable, List, Tuple

__all__ = ["get_optimize_with_timeout_function"]


def _run_optimize(
    func: Callable[..., Tuple[List[float], List[float], int]],
    args: tuple,
    kwargs: dict,
    return_dict: Any,
) -> None:
    """Worker process that runs the C++ optimizer and stores its result."""
    try:
        result = func(*args, **kwargs)
        return_dict["result"] = result
    except Exception as e:
        return_dict["error"] = str(e)


def get_optimize_with_timeout_function(
    optimize_func: Callable[..., Tuple[List[float], List[float], int]],
    timeout_seconds: float,
    x_timeout: List[float],
    fitness_func: Callable,
) -> Callable[..., Tuple[List[float], List[float], int]]:
    """
    Wrap the Pybind11-based `optimize` function with a timeout safeguard.

    Parameters
    ----------
    optimize_func : callable
        The Pybind11-bound `optimize` function to execute.
        Must return a tuple `(x_opt, f_opt, status)`.
    timeout_seconds : float
        Maximum runtime in seconds before the optimizer process is terminated.
    x_timeout : List[float]
        Decision vector to return on timeout
    fitness_func : callable
        Fitness function to return on timeout

    Returns
    -------
    callable
        A wrapped version of `optimize_func` with the same signature.
        When called:
          * Returns `(x_opt, multipliers, status)` if the optimizer completes.
          * Returns `([], [], 5)` if the optimizer exceeds the timeout.

    Notes
    -----
    - The wrapped function runs the optimizer in a separate process using
      :mod:`multiprocessing` to allow safe termination if the Fortran backend hangs.
    - Status code `5` indicates a timeout occurred.
    - This approach ensures that the main Python process remains responsive
      and that no hanging Fortran thread blocks program exit.
    """

    def wrapped_optimize(*args, **kwargs) -> Tuple[List[float], List[float], int]:
        manager = mp.Manager()
        return_dict = manager.dict()

        process = mp.Process(target=_run_optimize, args=(optimize_func, args, kwargs, return_dict))
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            print(
                f"⚠️  Optimization timed out after {timeout_seconds} seconds — terminating process."
            )
            process.terminate()
            process.join()
            # Return timeout status code instead of raising
            return (x_timeout, fitness_func(x_timeout), 5)

        if "error" in return_dict:
            print(f"⚠️  Optimizer process failed: {return_dict['error']}")
            return (x_timeout, fitness_func(x_timeout), 5)

        return return_dict.get("result", (x_timeout, fitness_func(x_timeout), 5))

    return wrapped_optimize
