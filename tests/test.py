import pyoptgra
import pygmo
import math

import unittest

# problem class with numerical gradient, equality and inequality constraints from
# https://esa.github.io/pygmo2/tutorials/coding_udp_constrained.html
class luksan_vlcek:
    def fitness(self, x):
        obj = 0
        for i in range(3):
            obj += (x[2*i-2]-3)**2 / 1000. - (x[2*i-2]-x[2*i-1]) # + math.exp(20.*(x[2*i - 2]-x[2*i-1]))
        ce1 = 4*(x[0]-x[1])**2+x[1]-x[2]**2+x[2]-x[3]**2
        ce2 = 8*x[1]*(x[1]**2-x[0])-2*(1-x[1])+4*(x[1]-x[2])**2+x[0]**2+x[2]-x[3]**2+x[3]-x[4]**2
        ce3 = 8*x[2]*(x[2]**2-x[1])-2*(1-x[2])+4*(x[2]-x[3])**2+x[1]**2-x[0]+x[3]-x[4]**2+x[0]**2+x[4]-x[5]**2
        ce4 = 8*x[3]*(x[3]**2-x[2])-2*(1-x[3])+4*(x[3]-x[4])**2+x[2]**2-x[1]+x[4]-x[5]**2+x[1]**2+x[5]-x[0]
        ci1 = 8*x[4]*(x[4]**2-x[3])-2*(1-x[4])+4*(x[4]-x[5])**2+x[3]**2-x[2]+x[5]+x[2]**2-x[1]
        ci2 = -(8*x[5] * (x[5]**2-x[4])-2*(1-x[5]) +x[4]**2-x[3]+x[3]**2 - x[4])
        return [obj, ce1,ce2,ce3,ce4,ci1,ci2]

    def get_bounds(self):
        return ([-5]*6,[5]*6)
    def get_nic(self):
        return 2
    def get_nec(self):
        return 4
    def gradient(self, x):
        return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)

		

class pygmo_test(unittest.TestCase):

	def runTest(self):
		self.basic_no_gradient_test()
		self.gradient_no_constraints_test()
		self.gradient_with_constraints_test()

	def basic_no_gradient_test(self):
		# Basic test that the call works and the result changes. No constraints, not gradients.

		algo = pygmo.algorithm(pyoptgra.optgra())
		prob = pygmo.problem(pygmo.schwefel(30))

		# Check that empty population is rejected
		empty_pop = pygmo.population(prob, 0)
		with self.assertRaises(ValueError):
			pop = algo.evolve(pop)

		# Prepare normal population
		pop = pygmo.population(prob, 1)
		previous_best = pop.champion_f

		# Calling optgra
		pop = algo.evolve(pop)
		new_best = pop.champion_f

		self.assertLess(new_best, previous_best)

	def gradient_no_constraints_test(self):

		algo = pygmo.algorithm(pyoptgra.optgra())
		prob = pygmo.problem(pygmo.rosenbrock(30))
		pop = pygmo.population(prob, 1)
		previous_best = pop.champion_f

		# Calling optgra
		pop = algo.evolve(pop)
		new_best = pop.champion_f

		self.assertLess(new_best, previous_best)

	def gradient_with_constraints_test(self):
		prob = pygmo.problem(luksan_vlcek())
		og = pyoptgra.optgra(optimization_method=1,max_iterations=100,max_correction_iterations=100,derivatives_computation=1,convergence_thresholds=[1e-6]*prob.get_nf(), max_distance_per_iteration=10)
		og.set_verbosity(1)
		algo = pygmo.algorithm(og)
		pop = pygmo.population(prob, size=0, seed=1)  # empty population
		pop.push_back(  [ 0.5,  0.5, -0.5,  0.4,  0.3,  0.7] )             # add initial guess
		pop.problem.c_tol = [1E-6] * 6

		# Calling optgra
		pop = algo.evolve(pop)    # run the optimisation

		# objective function
		self.assertLess(pop.champion_f[0], 2.26)

		# equality constraints
		for i in [1, 2, 3, 4]:
			self.assertAlmostEqual(pop.champion_f[i], 0.0, 6)

		# inequality constraints
		for i in [5, 6]:
			self.assertLess(pop.champion_f[i], 1e-6)


if __name__ == '__main__':
    unittest.main()