from collections import deque
from typing import List

from pygmo import s_policy, select_best

from .core import optimize


class optgra:
    """
    This class is a user defined algorithm (UDA) providing a wrapper around OPTGRA, which is written in Fortran.

    It is specifically designed for near-linear optimization problems with many constraints.
    When optimizing a problem, Optgra will first move towards satisfying the constraints,
    then move along the feasible region boundary to optimize the merit function,
    fixing constraint violations as they occur.

    For this, constraints and the merit function are linearized. Optgra will perform less well on
    very non-linear merit functions or constraints.

    """

    @staticmethod
    def _wrap_fitness_func(problem):
        """"""

        def wrapped_fitness(x):
            result = deque(problem.fitness(x))

            # optgra expects the fitness last, pagmo has the fitness first
            result.rotate(-1)
            return list(result)

        return wrapped_fitness

    @staticmethod
    def _wrap_gradient_func(problem):

        sparsity_pattern = problem.gradient_sparsity()

        shape = (problem.get_nf(), problem.get_nx())

        def wrapped_gradient(x):
            sparse_values = problem.gradient(x)

            nnz = len(sparse_values)

            result = [[0 for j in range(shape[1])] for i in range(shape[0])]

            for i in range(nnz):
                fIndex, xIndex = sparsity_pattern[i]

                # reorder constraint order, optgra expects the fitness last, pagmo has the fitness first
                if fIndex == 0:
                    fIndex = problem.get_nf() - 1
                else:
                    fIndex = int(fIndex - 1)

                result[fIndex][xIndex] = sparse_values[i]

            return result

        return wrapped_gradient

    def __init__(
        self,
        max_iterations: int = 10,
        max_correction_iterations: int = 10,
        max_distance_per_iteration: int = 10,
        perturbation_for_snd_order_derivatives: int = 10,
        # TODO the following might be replaced with c_tol, check with Johannes
        convergence_thresholds: List[float] = [],  # f_dim
        variable_scaling_factors: List[float] = [],  # x_dim
        constraint_priorities: List[int] = [],  # f_dim
        optimization_method: int = 2,
        verbosity: int = 0,
    ) -> None:
        """
        Initialize a wrapper instance for the OPTGRA algorithm.

        Some of the construction arguments, for example the scaling factors, depend on the dimension of the problem.
        Passing a problem with a different dimension to the instance's evolve function will result in an error.

        Some problem-specific options are deduced from the problem in the population given to the evolve function.

        Args:

            max_iterations: number of optimization iterations
            max_correction_iterations: number of constraint correction iterations within each optimization iteration
            max_distance_per_iteration: maximum distance traveled in each optimization iteration
            perturbation_for_snd_order_derivatives: Used as delta for numerically computing second order errors
                of the constraints in the optimization step
            convergence_thresholds: optional - Scaling factors for the constraints.
                If passed, must be positive and one more than there are constraints.
            variable_scaling_factors: optional - Scaling factors for the input variables.
                If passed, must be positive and as many as there are variables
            constraint_priorities: optional - How to prioritize constraint fulfillment in the initial phase
            optimization_method: select 0 for steepest descent, 1 for modified spectral conjugate gradient method,
                2 for spectral conjugate gradient method and 3 for conjugate gradient method
            verbosity: 0 has no output, 4 and higher have maximum output

        Raises:

            ValueError: if both convergence thresholds and constraint priorities are passed and their lengths differ
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
        self.convergence_thresholds = convergence_thresholds
        self.variable_scaling_factors = variable_scaling_factors
        self.constraint_priorities = constraint_priorities
        self.optimization_method = optimization_method
        self.selection = s_policy(select_best(rate=1))

        self.log_level = verbosity

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

        conv_len = len(convergence_thresholds)
        prio_len = len(constraint_priorities)
        if conv_len > 0 and prio_len > 0 and conv_len != prio_len:
            raise ValueError(
                str(conv_len)
                + " constraint scaling factors passed,"
                + " but "
                + str(prio_len)
                + " constraint priorities."
            )

    def set_verbosity(self, level: int) -> None:
        """
        Sets verbosity of optgra.

        Args:
            verbosity: Useful values go from 0 to 4
        """
        self.log_level = level

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
            ValueError: If during the construction of the wrapper, convergence_thresholds,
                constraint_priorities or variable_scaling_factors were passed that don't fit to the given problem.
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

        num_function_output = 1 + problem.get_nec() + problem.get_nic()
        conv_len = len(self.convergence_thresholds)
        if conv_len > 0 and conv_len != num_function_output:
            raise ValueError(
                str(conv_len)
                + " constraint scaling factors passed for problem"
                + " with "
                + str(num_function_output)
                + " function outputs."
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

        fitness_func = optgra._wrap_fitness_func(problem)
        grad_func = None
        derivatives_computation = 2
        if problem.has_gradient():
            grad_func = optgra._wrap_gradient_func(problem)
            derivatives_computation = 1

        # 0 for equality constraints, -1 for inequality constraints, -1 for fitness
        constraint_types = [0] * problem.get_nec() + [-1] * problem.get_nic() + [-1]

        # still to set: variable_names, constraint_names, autodiff_deltas
        variable_names: List[str] = []
        constraint_names: List[str] = []
        autodiff_deltas: List[float] = []

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
            convergence_thresholds=self.convergence_thresholds,
            variable_scaling_factors=self.variable_scaling_factors,
            constraint_priorities=self.constraint_priorities,
            variable_names=variable_names,
            constraint_names=constraint_names,
            optimization_method=self.optimization_method,
            derivatives_computation=derivatives_computation,
            autodiff_deltas=autodiff_deltas,
            log_level=self.log_level,
        )

        best_x, best_f, finopt = result

        best_f = deque(best_f)
        best_f.rotate(+1)
        population.set_xf(idx, best_x, list(best_f))

        return population
