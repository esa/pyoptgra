/*
 * Copyright 2008, 2021 European Space Agency
 *
 * This file is part of pyoptgra, a pygmo affiliated library.
 *
 * This Source Code Form is available under two different licenses.
 * You may choose to license and use it under version 3 of the
 * GNU General Public License or under the
 * ESA Software Community Licence (ESCL) 2.4 Weak Copyleft.
 * We explicitly reserve the right to release future versions of 
 * Pyoptgra and Optgra under different licenses.
 * If copies of GPL3 and ESCL 2.4 were not distributed with this
 * file, you can obtain them at https://www.gnu.org/licenses/gpl-3.0.txt
 * and https://essr.esa.int/license/european-space-agency-community-license-v2-4-weak-copyleft
 */

#include <iostream>
#include <vector>
#include <tuple>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "wrapper.hpp"
#include "test-callables.hpp"

using std::cout;
using std::endl;
using std::vector;

using namespace optgra;
using Catch::Approx;

TEST_CASE( "Wrapper optimizes", "[wrapper-optimize]" )
{
	std::vector<double> initial_x = {1,1,1,1,1};
	int dim = initial_x.size();
	std::vector<double> bestx, bestf;
	int finopt;
	
	std::tie(bestx, bestf, finopt) = optimize(initial_x, {0,-1}, f, g, true, 150, 90, 10, 1, {1e-12, 1e-12});

    // check that bestx is close to 0, 1, 2, 3, 4
    REQUIRE (bestx[0] == Approx(0.0).margin(0.01));
    REQUIRE (bestx[1] == Approx(1.0).margin(0.01));
    REQUIRE (bestx[2] == Approx(2.0).margin(0.01));
    REQUIRE (bestx[3] == Approx(3.0).margin(0.01));
    REQUIRE (bestx[4] == Approx(4.0).margin(0.01));
  
    // check that bestf is close to 0 10
    REQUIRE (bestf[0] == Approx(0.0).margin(0.01));
    REQUIRE (bestf[1] == Approx(10.0).margin(0.01));
}

TEST_CASE( "Wrapper computes sensitivity matrices", "[wrapper-sensitivity]" )
{
	// Testing sensitivity

	std::vector<double> sens_x = {0.000183705, 1, 1.99982, 2.99963, 3.99945};
	std::vector<double> sens_f = {5.06211e-07, 9.99908};

    std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
     std::vector<std::vector<double>>, std::vector<std::vector<double>>> matrixtuple = compute_sensitivity_matrices(sens_x, {0,-1}, f, g, true);

    std::tuple<sensitivity_state, std::vector<double>> state_tuple = prepare_sensitivity_state(sens_x, {0,-1}, f, g, true);

    std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
     std::vector<std::vector<double>>, std::vector<std::vector<double>>> matrixtuple_indirect 
     = get_sensitivity_matrices(std::get<0>(state_tuple), {0,0,0,0,0}, {0, -1});

    REQUIRE (std::get<0>(matrixtuple) == std::get<0>(matrixtuple_indirect));
    REQUIRE (std::get<1>(matrixtuple) == std::get<1>(matrixtuple_indirect));
    REQUIRE (std::get<2>(matrixtuple) == std::get<2>(matrixtuple_indirect));
    REQUIRE (std::get<3>(matrixtuple) == std::get<3>(matrixtuple_indirect));
    REQUIRE (std::get<4>(matrixtuple) == std::get<4>(matrixtuple_indirect));
}

TEST_CASE( "Wrapper computes sensitivity state", "[wrapper-sensitivity-state]" )
{
	int num_variables = 1;
	int num_constraints = 1;

	vector<double> senvar(num_variables);
    vector<double> senqua(num_constraints+1);
    vector<double> sencon(num_constraints+1);
    vector<int> senact(num_constraints+1);
    vector<double> sender((num_constraints+1)*num_variables);
    vector<int> actcon(num_constraints+1);
    vector<int> conact(num_constraints+4);
    vector<double> conred((num_constraints+3)*num_variables);
    vector<double> conder((num_constraints+3)*num_variables);

    sensitivity_state tuplecache;
    std::tie(tuplecache, std::ignore) = prepare_sensitivity_state({10}, {-1,-1}, f_simple, g_simple, true);
    std::tie(senvar, senqua, sencon, senact, sender, actcon, conact, conred, conder) = tuplecache;

    cout << "senact: ";
    for (int i = 0; i < num_constraints+1; i++) {
            cout << senact[i] << " ";
    }
    cout << endl;

    cout << "actcon: ";
    for (int i = 0; i < num_constraints+1; i++) {
            cout << senact[i] << " ";
    }
    cout << endl;

    cout << "conact: ";
    for (int i = 0; i < num_constraints+1; i++) {
            cout << senact[i] << " ";
    }
    cout << endl;

    std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
     std::vector<std::vector<double>>, std::vector<std::vector<double>>> matrixtuple_indirect
     = get_sensitivity_matrices(tuplecache, {0}, {-1, -1});

     cout << "constraints_active: ";
    for (int i = 0; i < num_constraints+1; i++) {
            cout << std::get<0>(matrixtuple_indirect)[i] << " ";
    }
    cout << endl;
}

TEST_CASE( "Wrapper performs sensitivity updates", "[wrapper-sensitivity-update-delta]" )
{
    const std::vector<int> variable_types = {0};
    const std::vector<int> constraint_types = {-1,-1};

    std::vector<double> x = {10};
    fitness_callback fitness = f_simple;
    gradient_callback gradient = g_simple;
    std::vector<double> delta = {-2};
    sensitivity_state state_tuple = std::get<0>(prepare_sensitivity_state(x, constraint_types, fitness, gradient, true));

    std::vector<double> bestx, bestf, best_orig;
    int finopt;
    std::tie(bestx, bestf, finopt) = sensitivity_update_constraint_delta(state_tuple, variable_types, constraint_types, delta);

    REQUIRE( bestx[0] == Approx(12.0) );
    best_orig = fitness(bestx);
    REQUIRE( best_orig[0] == Approx(delta[0]) );

}