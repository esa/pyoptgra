import pyoptgra
import pygmo

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
        return ([-5] * 6, [5] * 6)

    def get_nic(self):
        return 2

    def get_nec(self):
        return 4

    def gradient(self, x):
        return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)


class pygmo_test(unittest.TestCase):
    def runTest(self):
        self.constructor_test()
        self.evolve_input_check_test()
        self.basic_no_gradient_test()
        self.gradient_no_constraints_test()
        self.gradient_with_constraints_test()

    def constructor_test(self):
        # Check that invalid optimization method is rejected
        with self.assertRaises(ValueError):
            _ = pygmo.algorithm(pyoptgra.optgra(optimization_method=5))

        # Check that negative iteration count is rejected
        with self.assertRaises(ValueError):
            _ = pygmo.algorithm(pyoptgra.optgra(max_iterations=-1))

        # Check that negative correction iteration count is rejected
        with self.assertRaises(ValueError):
            _ = pygmo.algorithm(pyoptgra.optgra(max_correction_iterations=-1))

        # Check that negative distance per iteration is rejected
        with self.assertRaises(ValueError):
            _ = pygmo.algorithm(pyoptgra.optgra(max_distance_per_iteration=-1))

        # Check that negative perturbation is rejected
        with self.assertRaises(ValueError):
            _ = pygmo.algorithm(
                pyoptgra.optgra(perturbation_for_snd_order_derivatives=-1)
            )

        # Check that conflicting sizes of constraint scalings and priorities are rejected
        with self.assertRaises(ValueError):
            _ = pygmo.algorithm(
                pyoptgra.optgra(
                    convergence_thresholds=[1], constraint_priorities=[1, 2]
                )
            )

        # Valid constructor
        pygmo.algorithm(pyoptgra.optgra())

    def evolve_input_check_test(self):
        algo = pygmo.algorithm(pyoptgra.optgra())
        prob = pygmo.problem(pygmo.schwefel(30))

        # Check that empty population is rejected
        empty_pop = pygmo.population(prob, 0)
        with self.assertRaises(ValueError):
            empty_pop = algo.evolve(empty_pop)

        class toy_multi_problem(object):
            def __init__(self):
                pass

            def fitness(self, x):
                return (sum(x), 1)

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def get_nobj(self):
                return 2

        # Check that multi-objective problems are rejected
        mprob = pygmo.problem(toy_multi_problem())
        mpop = pygmo.population(mprob, 1)
        with self.assertRaises(ValueError):
            algo.evolve(mpop)

        class toy_stochastic_problem(object):
            def __init__(self):
                self.seed = 0

            def fitness(self, x):
                import random

                return [random.random() + x[0]]

            def get_bounds(self):
                return ([0], [1])

            def set_seed(self, seed):
                self.seed = seed

        # Check that stochastic problems are rejected
        sprob = pygmo.problem(toy_stochastic_problem())
        spop = pygmo.population(sprob, 1)
        with self.assertRaises(ValueError):
            algo.evolve(spop)

        # Check that scaling factors of wrong size are rejected
        algo = pygmo.algorithm(pyoptgra.optgra(variable_scaling_factors=[1] * 29))
        prob = pygmo.problem(pygmo.schwefel(30))
        pop = pygmo.population(prob, 1)
        with self.assertRaises(ValueError):
            algo.evolve(pop)

        # Correct size
        algo = pygmo.algorithm(pyoptgra.optgra(variable_scaling_factors=[1] * 30))
        algo.evolve(pop)

        # Check that convergence thresholds of wrong size are rejected
        algo = pygmo.algorithm(pyoptgra.optgra(convergence_thresholds=[1] * 2))
        prob = pygmo.problem(pygmo.schwefel(30))
        pop = pygmo.population(prob, 1)
        with self.assertRaises(ValueError):
            algo.evolve(pop)

        # Correct size
        algo = pygmo.algorithm(pyoptgra.optgra(convergence_thresholds=[1]))
        algo.evolve(pop)

        # Check that constraint priorities of wrong size are rejected
        algo = pygmo.algorithm(pyoptgra.optgra(constraint_priorities=[1] * 2))
        prob = pygmo.problem(pygmo.schwefel(30))
        pop = pygmo.population(prob, 1)
        with self.assertRaises(ValueError):
            algo.evolve(pop)

        # Correct size
        algo = pygmo.algorithm(pyoptgra.optgra(constraint_priorities=[1]))
        algo.evolve(pop)

    def basic_no_gradient_test(self):
        # Basic test that the call works and the result changes. No constraints, not gradients.

        algo = pygmo.algorithm(pyoptgra.optgra())
        prob = pygmo.problem(pygmo.schwefel(30))

        # Check that empty population is rejected
        empty_pop = pygmo.population(prob, 0)
        with self.assertRaises(ValueError):
            empty_pop = algo.evolve(empty_pop)

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
        og = pyoptgra.optgra(
            optimization_method=1,
            max_iterations=100,
            max_correction_iterations=100,
            convergence_thresholds=[1e-6] * prob.get_nf(),
            max_distance_per_iteration=10,
        )
        og.set_verbosity(1)
        algo = pygmo.algorithm(og)
        pop = pygmo.population(prob, size=0, seed=1)  # empty population
        pop.push_back([0.5, 0.5, -0.5, 0.4, 0.3, 0.7])  # add initial guess
        pop.problem.c_tol = [1e-6] * 6

        # Calling optgra
        pop = algo.evolve(pop)  # run the optimisation

        # objective function
        self.assertLess(pop.champion_f[0], 2.26)

        # equality constraints
        for i in [1, 2, 3, 4]:
            self.assertAlmostEqual(pop.champion_f[i], 0.0, 6)

        # inequality constraints
        for i in [5, 6]:
            self.assertLess(pop.champion_f[i], 1e-6)


if __name__ == "__main__":
    unittest.main()
