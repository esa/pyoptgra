from collections import deque

from .core.core import optimize

from typing import List

from pygmo import s_policy, select_best


class optgra:
    """
    This class is a user defined algorithm (UDA) providing a wrapper around OPTGRA, which is written in Fortran.

    """

    @staticmethod
    def _wrap_fitness_func(problem):
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
        convergence_thresholds: List[float] = [],  # this should be replaced with c_tol
        variable_scaling_factors: List[float] = [],
        constraint_priorities: List[int] = [],
        optimization_method: int = 2,
        derivatives_computation: int = 1,
        selection: s_policy = s_policy(select_best(rate=1)),
        verbosity: int = 0,
    ):

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
        self.derivatives_computation = derivatives_computation
        self.selection = selection

        self.log_level = verbosity

    def set_verbosity(self, level: int) -> None:

        self.log_level = level

    def evolve(self, population):

        problem = population.problem

        selected = self.selection.select(
            (population.get_ID(), population.get_x(), population.get_f()),
            problem.get_nx(),
            problem.get_nix(),
            problem.get_nobj(),
            problem.get_nec(),
            problem.get_nic(),
            problem.c_tol,
        )

        if len(selected[0]) != 1:
            raise ValueError(
                "Selection policy returned "
                + str(len(selected[0]))
                + " elements, but 1 was needed."
            )

        idx = list(population.get_ID()).index(selected[0][0])

        fitness_func = optgra._wrap_fitness_func(problem)
        grad_func = None
        if problem.has_gradient():
            grad_func = optgra._wrap_gradient_func(problem)

        # 0 for equality constraints, -1 for inequality constraints, -1 for fitness
        constraint_types = [0] * problem.get_nec() + [-1] * problem.get_nic() + [-1]

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
            derivatives_computation=self.derivatives_computation,
            autodiff_deltas=autodiff_deltas,
            log_level=self.log_level,
        )

        # still to set: variable_names, constraint_names, autodiff_deltas
        best_x, best_f, finopt = result

        best_f = deque(best_f)
        best_f.rotate(+1)
        population.set_xf(idx, best_x, list(best_f))

        return population
