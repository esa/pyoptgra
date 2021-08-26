Sensitivity Analysis
====================

Optgra supports sensitivity analysis with respect to the active constraints. An inequality constraint is considered active if it is fulfilled but close to being violated. Equality constraints are always active.

As an example, we take a problem with one variable (x_0) one minimization objective 2x_0 and one inequality constraint (x_0) >= 10. To make the problem bounded, we add bounds of \[-20, 20\]

.. testcode::

	import pygmo
	import pyoptgra
	
	class _prob(object):
	    
	    def __init__(self, silent=False):
	        self.silent = silent
	    
	    def get_bounds(self):
	        return ([-20], [20])
	
	    def fitness(self, x):
	        result = [2*x[0], -x[0] + 10]
	        if not self.silent:
	            print("f called with", x)
	        return result
	    
	    def get_nic(self):
	        return 1

The optimal solution is x_0 = 10, with a fitness of 20.

When using optgra to optimize this problem, the function will be called a few times, before settling at 10:

.. doctest::

	>>> import pygmo
	>>> import pyoptgra
	>>> prob = pygmo.problem(_prob())
	>>> prob.c_tol = 1e-6
	>>> pop = pygmo.population(prob, 1, seed=1)
	f called with [19.88739233]

.. testcode::

	algo = pygmo.algorithm(pyoptgra.optgra())
	pop = algo.evolve(pop)

.. testoutput::

    f called with [19.88739233]
    f called with [19.89739233]
    f called with [19.87739233]
    f called with [19.88739233]
    f called with [18.88739233]
    f called with [18.88739233]
    f called with [9.88739233]
    f called with [9.89739233]
    f called with [9.87739233]
    f called with [10.]
    f called with [10.]
    f called with [10.]
    f called with [10.01]
    f called with [9.99]
    f called with [10.]

.. doctest::

	>>> print(pop.champion_x, pop.champion_f)
	[10.] [2.00000000e+01 1.77635684e-15]

Functions for Sensitivity Analysis
==================================

The sensitivity analysis available by optgra consists of four functions:
1) prepare_sensitivity(problem, x) - prepares sensitivity analysis of *problem* at *x*. 
2) sensitivity_matrices() - returns one list and four matrices, giving the sensitivities of *problem* at *x*
3) linear_update_new_callable(new_problem) - evaluates *new_problem* at *x* and performs one optimization step
4) linear_update_delta(constraint_delta) - adds *constraint_delta* to the constraints of *problem* at *x* and performs one optimization step.

The first output of the function sensitivity_matrices is whether each constraint is *active*. Constraints are considered active if they are fulfilled, but not more than *max_distance_per_iteration* away from being unfulfilled, meaning they could be violated during the next optimization step.

For example, for *max_distance_per_iteration* = 1 the inequality constraint of x_0 >= 10 is *active* for x_0 in \[10, 11\] and inactive for x_0 < 10 or x_0 > 11.

Examples:

x_0 = 10, constraint x_0 >= 10 is just fulfilled, thus marked as active:

.. doctest::

	>>> prob = pygmo.problem(_prob(silent=True))
	>>> opt = pyoptgra.optgra(bounds_to_constraints=False)
	>>> opt.prepare_sensitivity(prob, [10])
	>>> opt.sensitivity_matrices()[0]
	[1]

x_0 = 9, constraint x_0 >= 10 is violated, thus inactive:

.. doctest::

	>>> import pygmo
	>>> prob = pygmo.problem(_prob(silent=True))
	>>> opt = pyoptgra.optgra(bounds_to_constraints=False)
	>>> opt.prepare_sensitivity(prob, [9])
	>>> opt.sensitivity_matrices()[0]
	[0]

x_0 = 11, constraint x_0 >= 10 is fulfilled and distance of one away from being violated, active:

.. doctest::

	>>> import pygmo
	>>> prob = pygmo.problem(_prob(silent=True))
	>>> opt = pyoptgra.optgra(bounds_to_constraints=False)
	>>> opt.prepare_sensitivity(prob, [11])
	>>> opt.sensitivity_matrices()[0]
	[1]

x_0 = 12, constraint x_0 >= 10 is fulfilled and distance of two away from being violated, reported as inactive:

.. doctest::

	>>> import pygmo
	>>> prob = pygmo.problem(_prob(silent=True))
	>>> opt = pyoptgra.optgra(bounds_to_constraints=False)
	>>> opt.prepare_sensitivity(prob, [12])
	>>> opt.sensitivity_matrices()[0]
	[0]