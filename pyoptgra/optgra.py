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

from collections import deque
from math import isfinite
from typing import List, Tuple, Union

from pygmo import s_policy, select_best

from .core import (
    get_sensitivity_matrices,
    optimize,
    prepare_sensitivity_state,
    sensitivity_update_constraint_delta,
    sensitivity_update_new_callable,
)


class optgra:
    """
    This class is a user defined algorithm (UDA) providing a wrapper around OPTGRA, which is written in Fortran.

    It is specifically designed for near-linear optimization problems with many constraints.
    When optimizing a problem, Optgra will first move towards satisfying the constraints,
    then move along the feasible region boundary to optimize the merit function,
    fixing constraint violations as they occur.

    For this, constraints and the merit function are linearized. Optgra will perform less well on
    very non-linear merit functions or constraints.

    Example:

    >>> import pygmo
    >>> import pyoptgra
    >>> prob = pygmo.problem(pygmo.schwefel(30))
    >>> pop = pygmo.population(prob, 1)
    >>> algo = pygmo.algorithm(pyoptgra.optgra())
    >>> pop = algo.evolve(pop)

    """

    @staticmethod
    def _constraint_types_from_box_bounds(problem):

        lb, ub = problem.get_bounds()
        # all box-derived constraints are positive
        finite_lb = sum(isfinite(elem) for elem in lb)
        finite_ub = sum(isfinite(elem) for elem in ub)
        resultTypes = [1] * (finite_lb + finite_ub)
        return resultTypes

    @staticmethod
    def _wrap_fitness_func(
        problem,
        bounds_to_constraints: bool = True,
        force_bounds: bool = False,
    ):
        def wrapped_fitness(x):

            fixed_x = x
            lb, ub = problem.get_bounds()

            if force_bounds:
                for i in range(problem.get_nx()):
                    if x[i] < lb[i]:
                        fixed_x[i] = lb[i]
                    if x[i] > ub[i]:
                        fixed_x[i] = ub[i]

            result = deque(problem.fitness(fixed_x))

            # add constraints derived from box bounds
            if bounds_to_constraints:
                for i in range(len(lb)):
                    if isfinite(lb[i]):
                        result.append(x[i] - lb[i])

                for i in range(len(ub)):
                    if isfinite(ub[i]):
                        result.append(ub[i] - x[i])

            # optgra expects the fitness last, pagmo has the fitness first
            result.rotate(-1)
            return list(result)

        return wrapped_fitness

    @staticmethod
    def _wrap_gradient_func(
        problem, bounds_to_constraints: bool = True, force_bounds=False
    ):

        sparsity_pattern = problem.gradient_sparsity()

        shape = (problem.get_nf(), problem.get_nx())

        def wrapped_gradient(x):
            lb, ub = problem.get_bounds()
            fixed_x = x

            if force_bounds:
                for i in range(problem.get_nx()):
                    if x[i] < lb[i]:
                        fixed_x[i] = lb[i]
                    if x[i] > ub[i]:
                        fixed_x[i] = ub[i]

            sparse_values = problem.gradient(fixed_x)

            nnz = len(sparse_values)

            result = [[0 for j in range(shape[1])] for i in range(shape[0])]

            # expand gradient to dense representation
            for i in range(nnz):
                fIndex, xIndex = sparsity_pattern[i]

                result[fIndex][xIndex] = sparse_values[i]

            # add box-derived constraints
            if bounds_to_constraints:
                lb, ub = problem.get_bounds()
                for i in range(problem.get_nx()):
                    if isfinite(lb[i]):
                        box_bound_grad = [0 for j in range(problem.get_nx())]
                        box_bound_grad[i] = 1
                        result.append(box_bound_grad)

                for i in range(len(ub)):
                    if isfinite(ub[i]):
                        box_bound_grad = [0 for j in range(problem.get_nx())]
                        box_bound_grad[i] = -1
                        result.append(box_bound_grad)

            # reorder constraint order, optgra expects the merit function last, pagmo has it first
            # equivalent to rotating in a dequeue
            gradient_of_merit = result.pop(0)
            result.append(gradient_of_merit)

            return result

        return wrapped_gradient

    def __init__(
        self,
        max_iterations: int = 150,
        max_correction_iterations: int = 90,
        max_distance_per_iteration: int = 10,
        perturbation_for_snd_order_derivatives: int = 1,
        variable_scaling_factors: List[float] = [],  # x_dim
        variable_types: List[int] = [],  # x_dim
        constraint_priorities: List[int] = [],  # f_dim
        bounds_to_constraints: bool = True,
        bound_constraints_tolerance: float = 1e-6,
        merit_function_threshold: float = 1e-6,
        # bound_constraints_scalar: float = 1,
        force_bounds: bool = False,
        optimization_method: int = 2,
        log_level: int = 0,
    ) -> None:
        """
        Initialize a wrapper instance for the OPTGRA algorithm.

        Some of the construction arguments, for example the scaling factors, depend on the dimension of the problem.
        Passing a problem with a different dimension to the instance's evolve function will result in an error.

        Some problem-specific options are deduced from the problem in the population given to the evolve function.

        Args:

            max_iterations: maximum number of total iterations
            max_correction_iterations: number of constraint correction iterations in the beginning
                If no feasible solution is found within that many iterations, Optgra aborts
            max_distance_per_iteration: maximum scaled distance traveled in each iteration
            perturbation_for_snd_order_derivatives: Used as delta for numerically computing second order errors
                of the constraints in the optimization step
            variable_scaling_factors: optional - Scaling factors for the input variables.
                If passed, must be positive and as many as there are variables
            variable_types: optional - Flags to set variables to either free (0) or fixed (1). Fixed variables
                are also called parameters in sensitivity analysis.
                If passed, must be as many flags as there are variables
            constraint_priorities: optional - lower constraint priorities are fulfilled earlier.
                During the initial constraint correction phase, only constraints with a priority at most k
                are considered in iteration k. Defaults to zero, so that all constraints are considered
                from the beginning.
            bounds_to_constraints: optional - if true (default), translate box bounds of the given problems into
                inequality constraints for optgra. Note that when also passing constraint priorities, the original
                constraints of the problem come first, followed by those derived from the lower box bounds, then those
                from the upper box bounds. Infinite bounds are ignored and not counted.
            bound_constraints_tolerance: optional - constraint tolerance for the constraints derived from bounds
            merit_function_threshold: optional - convergence threshold for merit function
            force_bounds: optional - whether to force the bounds given by the problem. If false (default), the
                fitness function might also be called with values of x that are outside of the bounds. Set to true
                if the fitness function cannot handle that.
            optimization_method: select 0 for steepest descent, 1 for modified spectral conjugate gradient method,
                2 for spectral conjugate gradient method and 3 for conjugate gradient method
            log_level: Control the original screen output of OPTGRA. 0 has no output,
                4 and higher have maximum output`. Set this to 0 if you want to use the pygmo
                logging system based on `set_verbosity()`.

        Raises:

            ValueError: if optimization_method is not one of 0, 1, 2, or 3
            ValueError: if any of max_iterations, max_correction_iterations, max_distance_per_iteration,
                or perturbation_for_snd_order_derivatives are negative

        """
        self.max_iterations = max_iterations
        self.max_correction_iterations = max_correction_iterations
        self.max_distance_per_iteration = max_distance_per_iteration
        self.perturbation_for_snd_order_derivatives = (
            perturbation_for_snd_order_derivatives
        )
        self.variable_scaling_factors = variable_scaling_factors
        self.variable_types = variable_types
        self.constraint_priorities = constraint_priorities
        self.optimization_method = optimization_method
        self.selection = s_policy(select_best(rate=1))

        self.bounds_to_constraints = bounds_to_constraints
        self.bound_constraints_tolerance = bound_constraints_tolerance
        self.merit_function_threshold = merit_function_threshold

        self.force_bounds = force_bounds
        # self.bound_violation_penalty = bound_violation_penalty

        self.log_level = log_level
        self.verbosity = 0  # by default no pygmo-style output
        self._sens_state = None
        self._sens_constraint_types: Union[List[int], None] = None

        if optimization_method not in [
            0,
            1,
            2,
            3,
        ]:  # TODO: use strings as arguments instead
            raise ValueError(
                "Passed optimization method "
                + str(optimization_method)
                + " is invalid, choose one of 0,1,2 or 3."
            )

        if not max_iterations >= 0:
            raise ValueError(
                "Passed value of "
                + str(max_iterations)
                + " is invalid for max_iterations, must be non-negative."
            )

        if not max_correction_iterations >= 0:
            raise ValueError(
                "Passed value of "
                + str(max_correction_iterations)
                + " is invalid for max_correction_iterations, must be non-negative."
            )

        if not max_distance_per_iteration >= 0:
            raise ValueError(
                "Passed value of "
                + str(max_distance_per_iteration)
                + " is invalid for max_distance_per_iteration, must be non-negative."
            )

        if not perturbation_for_snd_order_derivatives >= 0:
            raise ValueError(
                "Passed value of "
                + str(perturbation_for_snd_order_derivatives)
                + " is invalid for perturbation_for_snd_order_derivatives, must be non-negative."
            )

    def set_verbosity(self, verbosity: int) -> None:
        """
        Sets pygmo verbosity of optgra wrapper.

        Args:
            verbosity: Useful values go from 0 to 4
        """
        if self.log_level and verbosity:
            raise ValueError(
                "Cannot set verbosity to >0 value if OPTGRA log_level is choosen "
                "not to be zero upon construction."
            )
        self.verbosity = verbosity

    def evolve(self, population):
        """
        Call OPTGRA with the best-fitness member of the population as start value.

        Args:

            population: The population containing the problem and a set of initial solutions.

        Returns:

            The changed population.

        Raises:

            ValueError: If the population is empty
            ValueError: If the problem contains multiple objectives
            ValueError: If the problem is stochastic
            ValueError: If the problem dimensions don't fit to constraint_priorities
                or variable_scaling_factors that were passed to the wrapper constructor
            ValueError: If the problem has finite box bounds and bounds_to_constraints was
                set to True in the wrapper constructor (default), constraint_priorities
                were also passed but don't cover the additional bound-derived constraints
        """

        problem = population.problem

        if len(population) == 0:
            raise ValueError(
                "Population needs to have at least one member for use as initial guess."
            )

        if problem.get_nobj() > 1:
            raise ValueError(
                "Multiple objectives detected in "
                + problem.get_name()
                + " instance. Optgra cannot deal with them"
            )

        if problem.is_stochastic():
            raise ValueError(
                problem.get_name()
                + " appears to be stochastic, optgra cannot deal with it"
            )

        scaling_len = len(self.variable_scaling_factors)
        if scaling_len > 0 and scaling_len != problem.get_nx():
            raise ValueError(
                str(scaling_len)
                + " variable scaling factors passed for problem"
                + " with "
                + str(problem.get_nx())
                + " parameters."
            )

        if (
            len(self.variable_types) > 0
            and len(self.variable_types) != problem.get_nx()
        ):
            raise ValueError(
                str(len(self.variable_types))
                + " variable types passed for problem"
                + " with "
                + str(problem.get_nx())
                + " parameters."
            )

        bound_types = []
        if self.bounds_to_constraints:
            bound_types = optgra._constraint_types_from_box_bounds(problem)

        num_function_output = (
            1 + problem.get_nec() + problem.get_nic() + len(bound_types)
        )
        prio_len = len(self.constraint_priorities)
        if prio_len > 0 and prio_len != num_function_output:
            raise ValueError(
                str(prio_len)
                + " constraint priorities passed for problem"
                + " with "
                + str(num_function_output)
                + " function outputs."
            )

        selected = self.selection.select(
            (population.get_ID(), population.get_x(), population.get_f()),
            problem.get_nx(),
            problem.get_nix(),
            problem.get_nobj(),
            problem.get_nec(),
            problem.get_nic(),
            problem.c_tol,
        )

        idx = list(population.get_ID()).index(selected[0][0])

        fitness_func = optgra._wrap_fitness_func(
            problem, self.bounds_to_constraints, self.force_bounds
        )
        grad_func = None
        derivatives_computation = 2
        if problem.has_gradient():
            grad_func = optgra._wrap_gradient_func(
                problem, self.bounds_to_constraints, self.force_bounds
            )
            derivatives_computation = 1

        # 0 for equality constraints, -1 for inequality constraints,
        # 1 for box-derived constraints, -1 for fitness
        constraint_types = (
            [0] * problem.get_nec() + [-1] * problem.get_nic() + bound_types + [-1]
        )

        # optgra has merit function last, that threshold can be ignored
        convergence_thresholds = []
        if any(elem > 0 for elem in problem.c_tol) or len(bound_types) > 0:
            c_tol_list = [1e-6 for _ in problem.c_tol]
            if any(elem > 0 for elem in problem.c_tol):
                c_tol_list = list(problem.c_tol)
            convergence_thresholds = (
                c_tol_list
                + [self.bound_constraints_tolerance] * len(bound_types)
                + [self.merit_function_threshold]
            )

        # adjust constraint priorities if adding constraints from box bound
        constraint_priorities = self.constraint_priorities
        if self.bounds_to_constraints:
            constraint_priorities = constraint_priorities + [0] * len(bound_types)

        # still to set: variable_names, constraint_names, autodiff_deltas
        variable_names: List[str] = []
        constraint_names: List[str] = []
        autodiff_deltas: List[float] = []

        variable_types: List[int] = self.variable_types
        if len(variable_types) == 0:
            variable_types = [0 for _ in range(problem.get_nx())]

        result = optimize(
            initial_x=population.get_x()[idx],
            constraint_types=constraint_types,
            fitness_callback=fitness_func,
            gradient_callback=grad_func,
            has_gradient=problem.has_gradient(),
            max_iterations=self.max_iterations,
            max_correction_iterations=self.max_correction_iterations,
            max_distance_per_iteration=self.max_distance_per_iteration,
            perturbation_for_snd_order_derivatives=self.perturbation_for_snd_order_derivatives,
            convergence_thresholds=convergence_thresholds,
            variable_scaling_factors=self.variable_scaling_factors,
            constraint_priorities=self.constraint_priorities,
            variable_names=variable_names,
            constraint_names=constraint_names,
            optimization_method=self.optimization_method,
            derivatives_computation=derivatives_computation,
            autodiff_deltas=autodiff_deltas,
            variable_types=variable_types,
            log_level=self.log_level,
            verbosity=self.verbosity,
        )

        best_x, best_f, finopt = result

        violated = False
        if self.force_bounds:
            lb, ub = problem.get_bounds()
            for i in range(problem.get_nx()):
                if best_x[i] < lb[i]:
                    best_x[i] = lb[i]
                    violated = True
                if best_x[i] > ub[i]:
                    best_x[i] = ub[i]
                    violated = True
        if violated:
            population.set_x(idx, best_x)
        else:
            # merit function is last, constraints are from 0 to problem.get_nc(),
            # we ignore bound-derived constraints
            # pagmo_fitness = [best_f[-1]] + best_f[0 : problem.get_nc()]
            population.set_x(idx, best_x)  # , list(pagmo_fitness))

        return population

    def prepare_sensitivity(self, problem, x: List[float]) -> None:
        """
        Prepare OPTGRA for sensitivity analysis at x. This is independant from previous and later calls to evolve,
        but enables calls to sensitivity_matrices, linear_update_new_callable and linear_update_delta on this instance.

        This works by creating a linearization of the problem's fitness function around x.

        Args:

            problem: The problem being analyzed.
            x: The value of x around which linearization is performed

        Raises:

            ValueError: If the problem contains multiple objectives
            ValueError: If the problem is stochastic
            ValueError: If the problem dimensions don't fit to constraint_priorities
                or variable_scaling_factors that were passed to the wrapper constructor
            ValueError: If the problem has finite box bounds and bounds_to_constraints was
                set to True in the wrapper constructor (default), constraint_priorities
                were also passed but don't cover the additional bound-derived constraints
        """

        if problem.get_nobj() > 1:
            raise ValueError(
                "Multiple objectives detected in "
                + problem.get_name()
                + " instance. Optgra cannot deal with them"
            )

        if problem.is_stochastic():
            raise ValueError(
                problem.get_name()
                + " appears to be stochastic, optgra cannot deal with it"
            )

        scaling_len = len(self.variable_scaling_factors)
        if scaling_len > 0 and scaling_len != problem.get_nx():
            raise ValueError(
                str(scaling_len)
                + " variable scaling factors passed for problem"
                + " with "
                + str(problem.get_nx())
                + " parameters."
            )

        bound_types = []
        if self.bounds_to_constraints:
            bound_types = optgra._constraint_types_from_box_bounds(problem)

        num_function_output = (
            1 + problem.get_nec() + problem.get_nic() + len(bound_types)
        )
        prio_len = len(self.constraint_priorities)
        if prio_len > 0 and prio_len != num_function_output:
            raise ValueError(
                str(prio_len)
                + " constraint priorities passed for problem"
                + " with "
                + str(num_function_output)
                + " function outputs."
            )

        fitness_func = optgra._wrap_fitness_func(problem, self.bounds_to_constraints)
        grad_func = None
        derivatives_computation = 2
        if problem.has_gradient():
            grad_func = optgra._wrap_gradient_func(problem, self.bounds_to_constraints)
            derivatives_computation = 1

        # 0 for equality constraints, -1 for inequality constraints,
        # 1 for box-derived constraints, -1 for fitness
        constraint_types = (
            [0] * problem.get_nec() + [-1] * problem.get_nic() + bound_types + [-1]
        )

        # adjust constraint priorities if adding constraints from box bound
        constraint_priorities = self.constraint_priorities
        if self.bounds_to_constraints:
            constraint_priorities = constraint_priorities + [0] * len(bound_types)

        # still to set: variable_names, constraint_names, autodiff_deltas
        autodiff_deltas: List[float] = []
        variable_types: List[int] = self.variable_types
        if len(variable_types) == 0:
            variable_types = [0 for _ in x]

        state, new_x = prepare_sensitivity_state(
            x=x,
            constraint_types=constraint_types,
            fitness_callback=fitness_func,
            gradient_callback=grad_func,
            has_gradient=problem.has_gradient(),
            max_distance_per_iteration=self.max_distance_per_iteration,
            perturbation_for_snd_order_derivatives=self.perturbation_for_snd_order_derivatives,
            variable_scaling_factors=self.variable_scaling_factors,
            derivatives_computation=derivatives_computation,
            autodiff_deltas=autodiff_deltas,
            variable_types=variable_types,
            log_level=self.log_level,
            verbosity=self.verbosity,
        )

        self._sens_state = state
        self._sens_constraint_types = constraint_types
        self._sens_variable_types = variable_types

    def sensitivity_matrices(self):
        """
        Get stored sensitivity matrices prepared by earlier call to prepare_sensivitity.
        Note that active constraints are constraints that are currently fulfilled,
        but could be violated in the next iteration.
        Parameters refer to variables whose variable type was declared as fixed.

        Returns:

            A tuple of one list and four matrices: a boolean list of whether each constraint is active,
            the sensitivity of constraints + merit function with respect to active constraints,
            the sensitivity of constraints + merit function with respect to parameters,
            the sensitivity of variables with respect to active constraints,
            and the sensitivity of variables with respect to parameters.

        Raises:

            RuntimeError: If prepare_sensitivity has not been called on this instance

        """

        if self._sens_state is None or len(self._sens_state) == 0:
            raise RuntimeError("Please call prepare_sensitivity first")

        return get_sensitivity_matrices(
            self._sens_state,
            self._sens_variable_types,
            self._sens_constraint_types,
            self.max_distance_per_iteration,
        )

    def linear_update_new_callable(self, problem) -> Tuple[List[float], List[float]]:
        """
        Perform a single optimization step on the stored value of x, but with a new callable

        Args:
            problem: A problem containing the new callable.
                     Has to have same dimensions and types as the problem passed to prepare_sensitivity

        Returns:

            tuple of new_x, new_y

        Raises:

            RuntimeError: If prepare_sensitivity has not been called on this instance
            ValueError: If number or type of constraints of the new problem are different from
                those of the problem passed to prepare_sensitivity

        """

        if self._sens_state is None or len(self._sens_state) == 0:
            raise RuntimeError("Please call prepare_sensitivity first")

        bound_types = []
        if self.bounds_to_constraints:
            bound_types = optgra._constraint_types_from_box_bounds(problem)

        fitness_func = optgra._wrap_fitness_func(problem, self.bounds_to_constraints)
        grad_func = None
        derivatives_computation = 2
        if problem.has_gradient():
            grad_func = optgra._wrap_gradient_func(problem, self.bounds_to_constraints)
            derivatives_computation = 1

        # 0 for equality constraints, -1 for inequality constraints,
        # 1 for box-derived constraints, -1 for fitness
        constraint_types = (
            [0] * problem.get_nec() + [-1] * problem.get_nic() + bound_types + [-1]
        )

        if constraint_types != self._sens_constraint_types:
            raise ValueError(
                "Derived constraint types from new problem are, "
                + str(constraint_types)
                + ", but stored types for analysis are "
                + str(constraint_types)
            )  # TOOD: maybe report exact index of difference

        autodiff_deltas: List[float] = []

        return sensitivity_update_new_callable(
            self._sens_state,
            self._sens_variable_types,
            constraint_types,
            fitness_func,
            grad_func,
            problem.has_gradient(),
            self.max_distance_per_iteration,
            self.perturbation_for_snd_order_derivatives,
            self.variable_scaling_factors,
            derivatives_computation,
            autodiff_deltas,
            self.log_level,
            self.verbosity,
        )

    def linear_update_delta(
        self, constraint_delta: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Perform a single optimization step on the linear approximation prepared with prepare_sensitivity.
        For this, no new function calls to the problem callable are performed, making this potentially very fast.

        Args:
            constraint_delta: A list of deltas against the constraints. They are subtracted from the stored values.

        Returns:

            tuple of new_x, new_y

        Raises:

            RuntimeError: If prepare_sensitivity has not been called on this instance
            ValueError: If number of deltas does not fit number of constraints.

        """

        if self._sens_state is None or len(self._sens_state) == 0:
            raise RuntimeError("Please call prepare_sensitivity first")

        return sensitivity_update_constraint_delta(
            self._sens_state,
            self._sens_variable_types,
            self._sens_constraint_types,
            constraint_delta,
            self.max_distance_per_iteration,
            self.perturbation_for_snd_order_derivatives,
            self.variable_scaling_factors,
            self.log_level,
            self.verbosity,
        )

    def get_name(self) -> str:
        """
        Returns the name of this instance
        """
        return "Optgra"

    def get_extra_info(self) -> str:
        """
        Returns the parameters used for construction
        """
        return (
            "max_iterations = {max_iterations}, "
            + "max_correction_iterations = {max_correction_iterations}, "
            + "max_distance_per_iteration = {max_distance_per_iteration}, "
            + "perturbation_for_snd_order_derivatives = {perturbation_for_snd_order_derivatives}, "
            + "variable_scaling_factors = {variable_scaling_factors}, "
            + "variable_types = {variable_types}, "
            + "constraint_priorities = {constraint_priorities}, "
            + "bounds_to_constraints = {bounds_to_constraints}, "
            + "bound_constraints_tolerance = {bound_constraints_tolerance}, "
            + "merit_function_threshold = {merit_function_threshold}, "
            + "force_bounds = {force_bounds}, "
            + "optimization_method = {optimization_method}, "
            + "log_level = {log_level}"
            + "verbosity = {verbosity}"
        ).format(
            max_iterations=self.max_iterations,
            max_correction_iterations=self.max_correction_iterations,
            max_distance_per_iteration=self.max_distance_per_iteration,
            perturbation_for_snd_order_derivatives=self.perturbation_for_snd_order_derivatives,
            variable_scaling_factors=self.variable_scaling_factors,
            variable_types=self.variable_types,
            constraint_priorities=self.constraint_priorities,
            bounds_to_constraints=self.bounds_to_constraints,
            bound_constraints_tolerance=self.bound_constraints_tolerance,
            merit_function_threshold=self.merit_function_threshold,
            force_bounds=self.force_bounds,
            optimization_method=self.optimization_method,
            log_level=self.log_level,
            verbosity=self.verbosity,
        )
