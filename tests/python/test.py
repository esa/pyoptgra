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

import unittest

import numpy as np

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
            8 * x[5] * (x[5] ** 2 - x[4]) - 2 * (1 - x[5]) + x[4] ** 2 - x[3] + x[3] ** 2 - x[4]
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


class _prob_bound_test(object):
    def __init__(self):
        self._bounds_violated = False

    def get_bounds(self):
        return ([-10, -10, -10, -10, -10], [10, 10, 10, 10, 10])

    def fitness(self, x):
        result = [sum(x), sum([(x[i] - i) ** 2 for i in range(len(x))])]
        lb, ub = self.get_bounds()
        for i in range(len(lb)):
            if x[i] < lb[i]:
                self._bounds_violated = True
            if x[i] > ub[i]:
                self._bounds_violated = True
        return result

    def gradient(self, x):
        grad = []
        x_dim = len(self.get_bounds()[0])

        lb, ub = self.get_bounds()
        for i in range(len(lb)):
            if x[i] < lb[i]:
                self._bounds_violated = True
            if x[i] > ub[i]:
                self._bounds_violated = True

        for i in range(x_dim):  # merit gradient
            grad.append(1)

        for i in range(x_dim):  # constraint gradient
            grad.append(2 * (x[i] - 1))

        return grad

    def has_gradient(self):
        return True

    def get_nic(self):
        return 1


class optgra_test(unittest.TestCase):
    def runTest(self):
        self.constructor_test()
        self.evolve_input_check_test()
        self.basic_no_gradient_test()
        self.gradient_no_constraints_test()
        self.gradient_with_constraints_test()
        self.box_constraints_test()
        self.archipelago_evolve_test()
        self.archipelago_pickle_test()
        self.prepare_sensitivity_input_check_test()
        self.prepare_sensitivity_test()
        self.sensitivity_matrices_test()
        self.sensitivity_new_callable_test()
        self.sensitivity_constraint_delta_test()
        self.sensitivity_active_constraints_test()
        self.force_bounds_test()
        self.khan_bounds_test()
        self.khan_function_test()
        self.force_bounds_fitness_test()
        self.force_bounds_gradient_test()
        self.get_name_test()
        self.get_extra_info_test()
        self.verbosity_test()
        self.triangle_test()

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
            _ = pygmo.algorithm(pyoptgra.optgra(perturbation_for_snd_order_derivatives=-1))

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

        # Check that types of wrong size are rejected
        algo = pygmo.algorithm(pyoptgra.optgra(variable_types=[0] * 29))
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

        # check than nan in the decision vector is caught
        x = pop.get_x()[0]
        x[3] = np.nan
        pop.set_x(0, x)
        with self.assertRaises(ValueError):
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
        # 1. Run Luksan-Vlcek problem with optgra
        prob = pygmo.problem(luksan_vlcek())
        prob.c_tol = 1e-7
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
        # checking exact value as regression test
        self.assertEqual(pop.champion_f[0], 0.82929210248477)

        # equality constraints
        for i in [1, 2, 3, 4]:
            self.assertAlmostEqual(pop.champion_f[i], 0.0, 6)

        # inequality constraints
        for i in [5, 6]:
            self.assertLess(pop.champion_f[i], 1e-6)

        # get_extra_info
        info = algo.get_extra_info()
        self.assertTrue("Optimal Solution Found" in info)

        # 2. Run the same test with khan_bounds
        og = pyoptgra.optgra(
            optimization_method=1,
            max_iterations=100,
            max_correction_iterations=100,
            max_distance_per_iteration=10,
            khan_bounds=True,
        )
        algo = pygmo.algorithm(og)
        pop2 = pygmo.population(prob, size=0, seed=1)  # empty population
        pop2.push_back([0.5, 0.5, -0.5, 0.4, 0.3, 0.7])  # add initial guess

        # Calling optgra
        pop2 = algo.evolve(pop2)  # run the optimisation

        # objective function
        self.assertLess(pop2.champion_f[0], 2.26)
        # checking exact value as regression test
        self.assertEqual(pop2.champion_f[0], 0.8292921025820391)

        # equality constraints
        for i in [1, 2, 3, 4]:
            self.assertAlmostEqual(pop2.champion_f[i], 0.0, 6)

        # inequality constraints
        for i in [5, 6]:
            self.assertLess(pop2.champion_f[i], 1e-6)

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
        a = archipelago(5, udi=mp_island(), algo=pyoptgra.optgra(), prob=rosenbrock(), pop_size=10)
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
        a = archipelago(
            5,
            algo=pyoptgra.optgra(variable_scaling_factors=[0.5] * 3),
            prob=rosenbrock(),
            pop_size=3,
        )
        a.evolve()
        self.assertRaises(RuntimeError, lambda: a.wait_check())

    def archipelago_pickle_test(self):
        from pygmo import (
            archipelago,
            rosenbrock,
            mp_island,
        )
        from pickle import dumps, loads

        a = archipelago(5, algo=pyoptgra.optgra(), prob=rosenbrock(), pop_size=10)
        self.assertEqual(repr(a), repr(loads(dumps(a))))
        a = archipelago(5, algo=pyoptgra.optgra(), prob=_prob(), pop_size=10, udi=mp_island())
        self.assertEqual(repr(a), repr(loads(dumps(a))))

    def prepare_sensitivity_input_check_test(self):
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
        opt = pyoptgra.optgra()
        mprob = pygmo.problem(toy_multi_problem())
        with self.assertRaises(ValueError):
            opt.prepare_sensitivity(mprob, [0, 0])

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
        with self.assertRaises(ValueError):
            opt.prepare_sensitivity(sprob, [0])

        # Check that scaling factors of wrong size are rejected
        opt = pyoptgra.optgra(variable_scaling_factors=[1] * 29)
        prob = pygmo.problem(pygmo.schwefel(30))
        with self.assertRaises(ValueError):
            opt.prepare_sensitivity(prob, [0] * 30)

        # Correct size
        opt = pyoptgra.optgra(
            variable_scaling_factors=[1 for _ in range(30)], bounds_to_constraints=False
        )
        opt.prepare_sensitivity(prob, [0] * 30)

        # Check that constraint priorities of wrong size are rejected
        opt = pyoptgra.optgra(constraint_priorities=[1] * 2)
        prob = pygmo.problem(pygmo.schwefel(30))
        with self.assertRaises(ValueError):
            opt.prepare_sensitivity(prob, [0] * 30)

        # Correct size
        opt = pyoptgra.optgra(constraint_priorities=[1] * 61)
        opt.prepare_sensitivity(prob, [0] * 30)

    def prepare_sensitivity_test(self):
        opt = pyoptgra.optgra()
        prob = pygmo.problem(luksan_vlcek())
        opt.prepare_sensitivity(prob, [0] * 6)

    def sensitivity_matrices_test(self):
        opt = pyoptgra.optgra()

        # test illegal state
        with self.assertRaises(RuntimeError):
            opt.sensitivity_matrices()

        prob = pygmo.problem(luksan_vlcek())
        opt.prepare_sensitivity(prob, [0] * 6)

        matrices = opt.sensitivity_matrices()
        self.assertEqual(len(matrices), 5)
        self.assertEqual(len(matrices[0]), 6 + 12)  # luksan_vlcek has 6 constraints and 12 bounds
        self.assertLessEqual(max(matrices[0]), 1)
        self.assertGreaterEqual(min(matrices[0]), 0)

        # TODO: test matrix sizes and content

    def sensitivity_new_callable_test(self):
        opt = pyoptgra.optgra()

        prob = pygmo.problem(luksan_vlcek())

        # test illegal state
        with self.assertRaises(RuntimeError):
            opt.linear_update_new_callable(prob)

        opt.prepare_sensitivity(prob, [0] * 6)

        # test problem of different type
        prob2 = pygmo.problem(pygmo.schwefel(30))
        with self.assertRaises(ValueError):
            opt.linear_update_new_callable(prob2)

        # test correct problem
        opt.linear_update_new_callable(prob)

    def sensitivity_constraint_delta_test(self):
        opt = pyoptgra.optgra()

        # test illegal state
        with self.assertRaises(RuntimeError):
            opt.linear_update_delta([1] * 16)

        prob = pygmo.problem(luksan_vlcek())
        opt.prepare_sensitivity(prob, [0] * 6)

        opt.linear_update_delta([1] * 18)

    def sensitivity_active_constraints_test(self):
        class _prob(object):
            def get_bounds(self):
                return ([-10], [10])

            def fitness(self, x):
                result = [2 * x[0], 10 - x[0]]
                return result

            def get_nic(self):
                return 1

        prob = pygmo.problem(_prob())
        opt = pyoptgra.optgra(
            bounds_to_constraints=False, log_level=4, max_distance_per_iteration=1
        )
        x = [10]
        opt.prepare_sensitivity(prob, x)
        self.assertEqual(opt.sensitivity_matrices()[0], [1])

    def force_bounds_fitness_test(self):
        prob = pygmo.problem(_prob_bound_test())
        f = pyoptgra.optgra._wrap_fitness_func(prob, False, False)
        f([-20, 20, 0, 0, 0])
        extracted = prob.extract(_prob_bound_test)
        self.assertTrue(extracted._bounds_violated)

        prob2 = pygmo.problem(_prob_bound_test())
        f = pyoptgra.optgra._wrap_fitness_func(prob2, False, True)
        f([-20, 20, 0, 0, 0])
        extracted = prob2.extract(_prob_bound_test)
        self.assertFalse(extracted._bounds_violated)

    def force_bounds_gradient_test(self):
        prob = pygmo.problem(_prob_bound_test())
        g = pyoptgra.optgra._wrap_gradient_func(prob, False, False)
        g([-20, 20, 0, 0, 0])
        extracted = prob.extract(_prob_bound_test)
        self.assertTrue(extracted._bounds_violated)

        prob2 = pygmo.problem(_prob_bound_test())
        g = pyoptgra.optgra._wrap_gradient_func(prob2, False, True)
        g([-20, 20, 0, 0, 0])
        extracted = prob2.extract(_prob_bound_test)
        self.assertFalse(extracted._bounds_violated)

    def force_bounds_test(self):
        class _prob_bound_test_no_gradient(object):
            def __init__(self):
                self._bounds_violated = False

            def get_bounds(self):
                return ([-10, -10, -10, -10, -10], [10, 10, 10, 10, 10])

            def fitness(self, x):
                result = [
                    sum(x),
                    sum([(x[i] - 5 * i + 10) ** 2 for i in range(len(x))]),
                ]
                lb, ub = self.get_bounds()
                for i in range(len(lb)):
                    if x[i] < lb[i]:
                        self._bounds_violated = True
                    if x[i] > ub[i]:
                        self._bounds_violated = True
                return result

            def get_nic(self):
                return 1

        # check bounds violation with normal optgra
        algo = pygmo.algorithm(pyoptgra.optgra(force_bounds=False))
        prob = pygmo.problem(_prob_bound_test_no_gradient())
        pop = pygmo.population(prob, size=1)
        extracted = pop.problem.extract(_prob_bound_test_no_gradient)
        self.assertFalse(extracted._bounds_violated)
        pop = algo.evolve(pop)
        extracted = pop.problem.extract(_prob_bound_test_no_gradient)
        self.assertTrue(extracted._bounds_violated)

        # check bounds are forced when setting the argument
        algo = pygmo.algorithm(pyoptgra.optgra(force_bounds=True))
        prob = pygmo.problem(_prob_bound_test())
        pop = pygmo.population(prob, size=1)
        extracted = pop.problem.extract(_prob_bound_test)
        self.assertFalse(extracted._bounds_violated)
        pop = algo.evolve(pop)
        extracted = pop.problem.extract(_prob_bound_test)
        self.assertFalse(extracted._bounds_violated)

        # check that population has valid members
        algo = pygmo.algorithm(pyoptgra.optgra(force_bounds=True, bounds_to_constraints=False))
        prob = pygmo.problem(_prob_bound_test_no_gradient())
        pop = pygmo.population(prob, size=0)
        pop.push_back([2.47192039, -1.45880516, -9.03600606, -9.33306356, 3.85509973])
        extracted = pop.problem.extract(_prob_bound_test_no_gradient)
        self.assertFalse(extracted._bounds_violated)
        pop = algo.evolve(pop)
        extracted = pop.problem.extract(_prob_bound_test_no_gradient)
        self.assertFalse(extracted._bounds_violated)
        lb, ub = prob.get_bounds()
        for i in range(prob.get_nx()):
            self.assertTrue(pop.champion_x[i] >= lb[i])
            self.assertTrue(pop.champion_x[i] <= ub[i])

    def khan_bounds_test(self):
        class _prob_bound_test_no_gradient(object):
            def __init__(self):
                self._bounds_violated = False

            def get_bounds(self):
                return ([-10, -10, -10, -10, -10], [10, 10, 10, 10, 10])

            def fitness(self, x):
                result = [
                    sum(x),
                    sum([(x[i] - 5 * i + 10) ** 2 for i in range(len(x))]),
                ]
                lb, ub = self.get_bounds()
                for i in range(len(lb)):
                    if x[i] < lb[i]:
                        self._bounds_violated = True
                    if x[i] > ub[i]:
                        self._bounds_violated = True
                return result

            def get_nic(self):
                return 1

        # check bounds violation with normal optgra
        algo = pygmo.algorithm(pyoptgra.optgra(khan_bounds=False))
        prob = pygmo.problem(_prob_bound_test_no_gradient())
        pop = pygmo.population(prob, size=1)
        extracted = pop.problem.extract(_prob_bound_test_no_gradient)
        self.assertFalse(extracted._bounds_violated)
        pop = algo.evolve(pop)
        extracted = pop.problem.extract(_prob_bound_test_no_gradient)
        self.assertTrue(extracted._bounds_violated)

        # check bounds are satisfied when setting the argument
        algo = pygmo.algorithm(pyoptgra.optgra(khan_bounds=True))  # equivalent to 'sin'
        for khan_bounds in ["sin", "tanh", "triangle4"]:
            algo = pygmo.algorithm(pyoptgra.optgra(khan_bounds=khan_bounds))
            prob = pygmo.problem(_prob_bound_test())
            pop = pygmo.population(prob, size=1)
            extracted = pop.problem.extract(_prob_bound_test)
            self.assertFalse(extracted._bounds_violated)
            pop = algo.evolve(pop)
            extracted = pop.problem.extract(_prob_bound_test)
            self.assertFalse(extracted._bounds_violated)

            # check that population has valid members
            algo = pygmo.algorithm(
                pyoptgra.optgra(khan_bounds=khan_bounds, bounds_to_constraints=False)
            )
            prob = pygmo.problem(_prob_bound_test_no_gradient())
            pop = pygmo.population(prob, size=0)
            pop.push_back([2.47192039, -1.45880516, -9.03600606, -9.33306356, 3.85509973])
            extracted = pop.problem.extract(_prob_bound_test_no_gradient)
            self.assertFalse(extracted._bounds_violated)
            pop = algo.evolve(pop)
            extracted = pop.problem.extract(_prob_bound_test_no_gradient)
            self.assertFalse(extracted._bounds_violated)
            lb, ub = prob.get_bounds()
            for i in range(prob.get_nx()):
                self.assertTrue(pop.champion_x[i] >= lb[i])
                self.assertTrue(pop.champion_x[i] <= ub[i])

    def khan_function_test(self):
        # test base_khan_function
        lb = [-10, 0, -np.inf, -np.inf, -20]
        ub = [10, 30, np.inf, -np.inf, -10]
        x = np.asarray([-1, 2, 4, 4, -15.0])
        kfun = pyoptgra.base_khan_function(lb, ub)
        with self.assertRaises(NotImplementedError):
            kfun.eval(x)
        with self.assertRaises(NotImplementedError):
            kfun.eval_inv(x)
        with self.assertRaises(NotImplementedError):
            kfun.eval_grad(x)
        with self.assertRaises(NotImplementedError):
            kfun.eval_inv_grad(x)

        # test all variants of Khan functions
        for fun in [
            pyoptgra.khan_function_sin,
            pyoptgra.khan_function_tanh,
            lambda _lb, _ub, _ug: pyoptgra.khan_function_triangle(_lb, _ub, 1, _ug),
            lambda _lb, _ub, _ug: pyoptgra.khan_function_triangle(_lb, _ub, 3, _ug),
            lambda _lb, _ub, _ug: pyoptgra.khan_function_triangle(_lb, _ub, 4, _ug),
        ]:
            for unity_gradient in [True, False]:  # test both variants
                lb = [-10, 0, -np.inf, -np.inf, -20]
                ub = [10, 30, np.inf, -np.inf, -10]

                kfun = fun(lb, ub, unity_gradient)

                # check function and its inversion
                x = np.asarray([-1, 2, 4, 4, -15.0])
                x_optgra = kfun.eval_inv(x)
                x_check = kfun.eval(x_optgra)
                np.testing.assert_allclose(x, x_check, atol=1e-10)

                # check gradient and its inversion
                dx_dxog = kfun.eval_inv_grad(x)
                dxog_dx = kfun.eval_grad(x_optgra)
                check_mat = dx_dxog @ dxog_dx  # expect unity matrix
                np.testing.assert_allclose(check_mat, np.eye(5), atol=1e-10)

                # compare with numerical gradient
                dx_dxog_num = pygmo.estimate_gradient_h(
                    lambda _x: kfun.eval_inv(_x), x, dx=1e-7
                ).reshape(5, 5)
                dxog_dx_num = pygmo.estimate_gradient_h(
                    lambda _x: kfun.eval(_x), x_optgra, dx=1e-7
                ).reshape(5, 5)
                np.testing.assert_allclose(dx_dxog_num, dx_dxog, atol=1e-7)
                np.testing.assert_allclose(dxog_dx_num, dxog_dx, atol=1e-7)

                if unity_gradient:  # Check if gradient at mid-point is unity
                    with np.errstate(invalid="ignore"):
                        xmid = (np.asarray(ub) + np.asarray(lb)) / 2
                    dxog_dx0 = kfun.eval_grad(xmid)
                    np.testing.assert_allclose(dxog_dx0, np.eye(5), atol=1e-10)

                # one-sided bound is not supported
                ub = [10, 30, np.inf, -np.inf, np.inf]
                with self.assertRaises(ValueError):
                    fun(lb, ub, unity_gradient)

    def get_name_test(self):
        algo = pygmo.algorithm(pyoptgra.optgra())
        self.assertEqual(algo.get_name(), "Optgra")

    def get_extra_info_test(self):
        algo = pygmo.algorithm(pyoptgra.optgra())
        self.assertTrue("iterations" in algo.get_extra_info())

    def verbosity_test(self):
        algo = pygmo.algorithm(pyoptgra.optgra(log_level=1))
        with self.assertRaises(ValueError):
            algo.set_verbosity(1)

    def triangle_test(self):
        """Test the triangular_wave_fourier function"""
        # Test that the function returns zero when N=0.
        x = np.linspace(-np.pi, np.pi, 11)

        # Test zero terms
        tri = pyoptgra.triangular_wave_fourier(0, x)
        np.testing.assert_array_almost_equal(tri, np.zeros_like(x))

        # Test single term (first harmonic only)
        tri = pyoptgra.triangular_wave_fourier(1, x)
        expected = np.sin(x)
        np.testing.assert_array_almost_equal(tri, expected)

        # Test normalization when using more than one term
        tri = pyoptgra.triangular_wave_fourier(4, np.pi / 2)
        self.assertAlmostEqual(tri, 1.0)

        # Test symmetry T(-x) = -T(x)
        tri1 = pyoptgra.triangular_wave_fourier(5, x)
        tri2 = -pyoptgra.triangular_wave_fourier(5, -x)
        np.testing.assert_array_almost_equal(tri1, tri2)

        # test inverse
        x = np.linspace(-np.pi / 2, np.pi / 2, 11)
        tri = pyoptgra.triangular_wave_fourier(5, x)
        x_check = pyoptgra.inverse_triangular_wave(5, tri)
        np.testing.assert_array_almost_equal(x, x_check)

        # test gradient
        tri_grad = pyoptgra.triangular_wave_fourier_grad(5, x)
        n = len(x)
        tri_grad_num = np.diag(
            pygmo.estimate_gradient_h(
                lambda _x: pyoptgra.triangular_wave_fourier(5, _x), x
            ).reshape(n, n)
        )
        np.testing.assert_array_almost_equal(tri_grad, tri_grad_num)

        tri_grad = pyoptgra.triangular_wave_fourier_grad(0, x)
        np.testing.assert_array_equal(tri_grad, np.zeros_like(x, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
