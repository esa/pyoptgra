import unittest

import pygmo

import pyoptgra


# problem class with numerical gradient, equality and inequality constraints from
# https://esa.github.io/pygmo2/tutorials/coding_udp_constrained.html
class luksan_vlcek:
    def fitness(self, x):
        obj = 0
        for i in range(3):
            obj += (x[2 * i - 2] - 3) ** 2 / 1000.0 - (
                x[2 * i - 2] - x[2 * i - 1]
            )  # + math.exp(20.*(x[2*i - 2]-x[2*i-1]))
        ce1 = 4 * (x[0] - x[1]) ** 2 + x[1] - x[2] ** 2 + x[2] - x[3] ** 2
        ce2 = (
            8 * x[1] * (x[1] ** 2 - x[0])
            - 2 * (1 - x[1])
            + 4 * (x[1] - x[2]) ** 2
            + x[0] ** 2
            + x[2]
            - x[3] ** 2
            + x[3]
            - x[4] ** 2
        )
        ce3 = (
            8 * x[2] * (x[2] ** 2 - x[1])
            - 2 * (1 - x[2])
            + 4 * (x[2] - x[3]) ** 2
            + x[1] ** 2
            - x[0]
            + x[3]
            - x[4] ** 2
            + x[0] ** 2
            + x[4]
            - x[5] ** 2
        )
        ce4 = (
            8 * x[3] * (x[3] ** 2 - x[2])
            - 2 * (1 - x[3])
            + 4 * (x[3] - x[4]) ** 2
            + x[2] ** 2
            - x[1]
            + x[4]
            - x[5] ** 2
            + x[1] ** 2
            + x[5]
            - x[0]
        )
        ci1 = (
            8 * x[4] * (x[4] ** 2 - x[3])
            - 2 * (1 - x[4])
            + 4 * (x[4] - x[5]) ** 2
            + x[3] ** 2
            - x[2]
            + x[5]
            + x[2] ** 2
            - x[1]
        )
        ci2 = -(
            8 * x[5] * (x[5] ** 2 - x[4])
            - 2 * (1 - x[5])
            + x[4] ** 2
            - x[3]
            + x[3] ** 2
            - x[4]
        )
        return [obj, ce1, ce2, ce3, ce4, ci1, ci2]

    def get_bounds(self):
        return ([-5] * 6, [5] * 6)

    def get_nic(self):
        return 2

    def get_nec(self):
        return 4

    def gradient(self, x):
        return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)

class _prob(object):

    def get_bounds(self):
        return ([0, 0], [1, 1])

    def fitness(self, a):
        return [42]


class optgra_test(unittest.TestCase):
    def runTest(self):
        self.constructor_test()
        self.evolve_input_check_test()
        self.basic_no_gradient_test()
        self.gradient_no_constraints_test()
        self.gradient_with_constraints_test()
        self.box_constraints_test()
        self.archipelago_evolve_test()
        self.archipelago_pickle_tests()

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

        # Check that constraint priorities of wrong size are rejected
        algo = pygmo.algorithm(pyoptgra.optgra(constraint_priorities=[1] * 2))
        prob = pygmo.problem(pygmo.schwefel(30))
        pop = pygmo.population(prob, 1)
        with self.assertRaises(ValueError):
            algo.evolve(pop)

        # Correct size
        algo = pygmo.algorithm(pyoptgra.optgra(constraint_priorities=[1] * 61))
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
        prob.c_tol = 1e-6
        og = pyoptgra.optgra(
            optimization_method=1,
            max_iterations=100,
            max_correction_iterations=100,
            max_distance_per_iteration=10,
        )
        og.set_verbosity(1)
        algo = pygmo.algorithm(og)
        pop = pygmo.population(prob, size=0, seed=1)  # empty population
        pop.push_back([0.5, 0.5, -0.5, 0.4, 0.3, 0.7])  # add initial guess

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

    def constraints_with_default_tolerances_test(self):
        prob = pygmo.problem(luksan_vlcek())
        pop = pygmo.population(prob, size=0, seed=1)  # empty population
        pop.push_back([0.5, 0.5, -0.5, 0.4, 0.3, 0.7])  # add initial guess
        algo = pygmo.algorithm(pyoptgra.optgra())
        pop = algo.evolve(pop)  # run the optimisation

    def box_constraints_test(self):
        class toy_box_bound_problem(object):
            def fitness(self, x):
                return [x[0] - x[1]]

            def get_bounds(self):
                return ([0, -float("inf")], [float("inf"), 1])

        # check that infinite bounds are not counted
        algo = pygmo.algorithm(pyoptgra.optgra(constraint_priorities=[1] * 5))
        prob = pygmo.problem(toy_box_bound_problem())
        pop = pygmo.population(prob, size=0)  # empty population
        pop.push_back([-0.5, 1.5])  # add initial guess, violating the constraints
        with self.assertRaises(ValueError):
            algo.evolve(pop)

        # Correct size
        algo = pygmo.algorithm(pyoptgra.optgra(constraint_priorities=[1] * 3))
        pop = algo.evolve(pop)

        # Check that box bounds are respected
        eps = 1e-6
        x = pop.get_x()[pop.best_idx()]
        self.assertGreaterEqual(x[0], 0 - eps)
        self.assertLessEqual(x[1], 1 + eps)

        # Check that bound constraint tolerances are respected
        tight_eps = 1e-20
        algo = pygmo.algorithm(pyoptgra.optgra(bound_constraints_tolerance=tight_eps))
        prob = pygmo.problem(toy_box_bound_problem())
        pop = pygmo.population(prob, size=0)  # empty population
        pop.push_back([-0.5, 1.5])  # add initial guess
        pop = algo.evolve(pop)
        x = pop.get_x()[pop.best_idx()]
        self.assertGreaterEqual(x[0], 0 - tight_eps)
        self.assertLessEqual(x[1], 1 + tight_eps)

    def archipelago_evolve_test(self):
        from pygmo import archipelago, rosenbrock, mp_island, evolve_status
        from copy import deepcopy
        a = archipelago()
        self.assertTrue(a.status == evolve_status.idle)
        a = archipelago(5, algo=pyoptgra.optgra(), prob=rosenbrock(), pop_size=10)
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait()
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait_check()
        # Copy while evolving.
        a.evolve(10)
        a.evolve(10)
        a2 = deepcopy(a)
        a.wait_check()
        a = archipelago(5, udi=mp_island(), algo=pyoptgra.optgra(),
                        prob=rosenbrock(), pop_size=10)
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait()
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait_check()
        # Copy while evolving.
        a.evolve(10)
        a.evolve(10)
        a2 = deepcopy(a)
        a.wait_check()
        # Throws on wait_check().
        a = archipelago(5, algo=pyoptgra.optgra(variable_scaling_factors=[0.5]*3), prob=rosenbrock(), pop_size=3)
        a.evolve()
        self.assertRaises(RuntimeError, lambda: a.wait_check())


    def archipelago_pickle_tests(self):
        from pygmo import archipelago, rosenbrock, mp_island, ring, migration_type, migrant_handling
        from pickle import dumps, loads
        a = archipelago(5, algo=pyoptgra.optgra(), prob=rosenbrock(), pop_size=10)
        self.assertEqual(repr(a), repr(loads(dumps(a))))
        a = archipelago(5, algo=pyoptgra.optgra(), prob=_prob(),
                        pop_size=10, udi=mp_island())
        self.assertEqual(repr(a), repr(loads(dumps(a))))


if __name__ == "__main__":
    unittest.main()
