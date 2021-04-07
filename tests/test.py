import pyoptgra
import pygmo

import unittest

class pygmo_test(unittest.TestCase):

	def runTest(self):
		self.basic_no_gradient_test()
		self.gradient_no_constraints_test()

	def basic_no_gradient_test(self):
		# Basic test that the call works and the result changes. No constraints, not gradients.

		algo = pygmo.algorithm(pyoptgra.optgra())
		prob = pygmo.problem(pygmo.schwefel(30))
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

if __name__ == '__main__':
    unittest.main()