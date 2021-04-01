from collections import deque

from pyoptgra.core import optimize

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

    def __init__(self, max_iterations: int = 10, max_correction_iterations: int = 10,
    	max_distance_per_iteration: int = 10, perturbation_for_snd_order_derivatives: int = 10,
    	convergence_thresholds: List[float] = [], variable_scaling_factors: List[float] = [],
    	constraint_priorities: List[int] = [], optimization_method: int = 2,
    	derivatives_computation: int = 1,
    	selection: s_policy = s_policy(select_best(rate=1))):

    	self.max_iterations = max_iterations
		self.max_correction_iterations = max_correction_iterations
		self.max_distance_per_iteration = max_distance_per_iteration
		self.perturbation_for_snd_order_derivatives = perturbation_for_snd_order_derivatives
		self.convergence_thresholds = convergence_thresholds
		self.variable_scaling_factors = variable_scaling_factors
		self.constraint_priorities = constraint_priorities
		self.optimization_method = optimization_method
		self.derivatives_computation = derivatives_computation
		self.selection = selection

		self.log_level = 0

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




