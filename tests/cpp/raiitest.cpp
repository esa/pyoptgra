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

TEST_CASE( "RAII initialization works and allows optimization", "[raii-exec]" )
{
	std::vector<double> initial_x = {1,1,1,1,1};
	std::vector<double> bestx, bestf;
	int finopt;
	
	int num_variables = initial_x.size();

    int derivatives_computation = 1;
    int num_constraints = 1;

    std::vector<int> variable_types(num_variables, 0);
    
    optgra_raii raii_object(variable_types, {0,-1}, 150, 150, 10, 1, {1e-12, 1e-12});
    
    std::tie(bestx, bestf, finopt) = raii_object.exec(initial_x, f, g);

    // check that bestx is close to 0, 1, 2, 3, 4
    REQUIRE (bestx[0] == Approx(0.0).margin(0.001));
    REQUIRE (bestx[1] == Approx(1.0).margin(0.001));
    REQUIRE (bestx[2] == Approx(2.0).margin(0.001));
    REQUIRE (bestx[3] == Approx(3.0).margin(0.001));
    REQUIRE (bestx[4] == Approx(4.0).margin(0.001));
  
    // check that bestf is close to 0 10
    REQUIRE (bestf[0] == Approx(0.0).margin(0.001));
    REQUIRE (bestf[1] == Approx(10.0).margin(0.001));
}
	
TEST_CASE( "RAII supports sensitivity matrices", "[raii-sensitivity-matrices]" )
{
	std::vector<double> sens_x = {0, 0.99, 2, 3, 4};
	int num_variables = sens_x.size();
	int derivatives_computation = 1;
    int num_constraints = 1;

	std::vector<int> variable_types(num_variables, 0);
	optgra_raii raii_object(variable_types, {0,-1});

	raii_object.initialize_sensitivity_data(sens_x, f, g);

	// maybe split off section here

	std::vector<int> constraint_status(num_constraints);
	std::vector<std::vector<double>> constraints_to_active_constraints(num_constraints+1);
    std::vector<std::vector<double>> constraints_to_parameters(num_constraints+1);
    std::vector<std::vector<double>> variables_to_active_constraints(num_variables);
    std::vector<std::vector<double>> variables_to_parameters(num_variables);

	std::tie(constraint_status, constraints_to_active_constraints, constraints_to_parameters,
	 variables_to_active_constraints, variables_to_parameters) = raii_object.sensitivity_matrices();

	cout << "Active constraints:" << endl;
	for (int i = 0; i < num_constraints; i++) {
		cout << constraint_status[i] << " ";
	}
	cout << endl;

	cout << "Constraints to active constraints:" << endl;
	for (int i = 0; i < num_constraints+1; i++) {
		for (int j = 0; j < num_constraints; j++) {
			cout << constraints_to_active_constraints[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Constraints to parameters:" << endl;
	for (int i = 0; i < num_constraints+1; i++) {
		for (int j = 0; j < num_variables; j++) {
			cout << constraints_to_parameters[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Variables to active constraints:" << endl;
	for (int i = 0; i < num_variables; i++) {
		for (int j = 0; j < num_constraints; j++) {
			cout << variables_to_active_constraints[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Variables to parameters:" << endl;
	for (int i = 0; i < num_variables; i++) {
		for (int j = 0; j < num_variables; j++) {
			cout << variables_to_parameters[i][j] << " ";
		}
		cout << endl;
	}
}

TEST_CASE( "RAII supports sensitivity updates with new callable", "[raii-sensitivity-update-same-function]" )
{
	std::vector<double> sens_x = {0, 0.99, 2, 3, 4};
	int num_variables = sens_x.size();
	int derivatives_computation = 1;
    int num_constraints = 1;

	std::vector<int> variable_types(num_variables, 0);
	optgra_raii raii_object(variable_types, {0,-1});

	raii_object.initialize_sensitivity_data(sens_x, f, g);

	std::vector<double> bestx, bestf, best_orig;
	int finopt;

	cout << endl << "Sensitivity Update Test" << endl;
	std::tie(bestx, bestf, finopt) = raii_object.sensitivity_update(f, g);

	// check that bestf is close to 0 10
    REQUIRE (bestf[0] == Approx(0.0).margin(0.001));
    REQUIRE (bestf[1] == Approx(10.0).margin(0.05));

	cout << "Best x:";
	for (int i = 0; i < num_variables; i++) {
		cout << bestx[i] << " ";
	}
	cout << endl;

	cout << "Best f:";
	for (int i = 0; i < 1 + 1; i++) {
		cout << bestf[i] << " ";
	}
	cout << endl;

	cout << "f(best_x):";
	best_orig = f(bestx);
	for (int i = 0; i < 1 + 1; i++) {
		cout << best_orig[i] << " ";
	}
	cout << endl;
}


TEST_CASE( "RAII supports sensitivity update with constraint delta and simple function", "[raii-update-delta-simple]" ) {
        const std::vector<int> variable_types = {0};
        const std::vector<int> constraint_types = {-1,-1};

        std::vector<double> x = {10};
        fitness_callback fitness = f_simple;
        gradient_callback gradient = g_simple;
        std::vector<double> delta = {-2};
        //int max_iterations = 150;// MAXITE
        //int max_correction_iterations = 90;// CORITE
        //double max_distance_per_iteration = 10;// VARMAX
        //double perturbation_for_snd_order_derivatives = 1; // VARSND) 

        int num_variables = variable_types.size();
        int num_constraints = constraint_types.size() - 1;
        //std::vector<double> autodiff_deltas = std::vector<double>(num_variables, 0.01);
        //int derivatives_computation = 2;

		optgra_raii raii_object(variable_types, constraint_types);

		raii_object.initialize_sensitivity_data(x, fitness, gradient);		

		std::vector<double> bestx, bestf, best_orig;
		int finopt;

		std::tie(bestx, bestf, finopt) = raii_object.sensitivity_update_delta(delta);

        REQUIRE( bestx[0] == Approx(12.0) );
        best_orig = fitness(bestx);
        REQUIRE( best_orig[0] == Approx(delta[0]) );
}

TEST_CASE( "RAII supports sensitivity updates with constraint delta and non-trivial function", "[raii-sensitivity-update-delta]" )
{
	std::vector<double> sens_x = {0, 0.99, 2, 3, 4};
	int num_variables = sens_x.size();
	int derivatives_computation = 1;
    int num_constraints = 1;

	std::vector<int> variable_types(num_variables, 0);
	optgra_raii raii_object(variable_types, {0,-1});//TODO: is this even correct

	raii_object.initialize_sensitivity_data(sens_x, f, g);

	cout << endl << "Sensitivity Update Constraint Delta Test" << endl;

	std::vector<double> delta = {-1};

	std::vector<double> bestx, bestf, best_orig;
	int finopt;

	std::tie(bestx, bestf, finopt) = raii_object.sensitivity_update_delta(delta);

	best_orig = f(bestx);
    REQUIRE( best_orig[0] == Approx(0.0).margin(0.001) );

	cout << "Best x:";
	for (int i = 0; i < num_variables; i++) {
		cout << bestx[i] << " ";
	}
	cout << endl;

	cout << "Best f:";
	for (int i = 0; i < 1 + 1; i++) {
		cout << bestf[i] << " ";
	}
	cout << endl;

	best_orig = f(bestx);
	cout << "f(best_x):";
	for (int i = 0; i < 1 + 1; i++) {
		cout << best_orig[i] << " ";
	}
	cout << endl;

}
	
TEST_CASE("RAII sensitivity_matrices with simple function", "[raii-sensitivity-matrices-simple]" )
{
    optgra_raii raii_object({0}, {-1,-1}, 150, 90, 1);

 	raii_object.initialize_sensitivity_data({2}, f_simple, g_simple);

 	int num_variables = 1;
    int num_constraints = 1;

 	std::vector<int> constraint_status(num_constraints);
	std::vector<std::vector<double>> constraints_to_active_constraints(num_constraints+1);
    std::vector<std::vector<double>> constraints_to_parameters(num_constraints+1);
    std::vector<std::vector<double>> variables_to_active_constraints(num_variables);
    std::vector<std::vector<double>> variables_to_parameters(num_variables);

 	std::tie(constraint_status, constraints_to_active_constraints, constraints_to_parameters,
	 variables_to_active_constraints, variables_to_parameters) = raii_object.sensitivity_matrices();

 	num_constraints = 1;
 	num_variables = 1;

	cout << "Active constraints:" << endl;
	for (int i = 0; i < num_constraints; i++) {
		cout << constraint_status[i] << " ";
	}
	cout << endl;

	cout << "Constraints to active constraints:" << endl;
	for (int i = 0; i < num_constraints+1; i++) {
		for (int j = 0; j < num_constraints; j++) {
			cout << constraints_to_active_constraints[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Constraints to parameters:" << endl;
	for (int i = 0; i < num_constraints+1; i++) {
		for (int j = 0; j < num_variables; j++) {
			cout << constraints_to_parameters[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Variables to active constraints:" << endl;
	for (int i = 0; i < num_variables; i++) {
		for (int j = 0; j < num_constraints; j++) {
			cout << variables_to_active_constraints[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Variables to parameters:" << endl;
	for (int i = 0; i < num_variables; i++) {
		for (int j = 0; j < num_variables; j++) {
			cout << variables_to_parameters[i][j] << " ";
		}
		cout << endl;
	}
}

TEST_CASE( "RAII supports sensitvity update with saved state", "[raii-update-delta-simple-state]" ) {
        const std::vector<int> variable_types = {0};
        const std::vector<int> constraint_types = {-1,-1};

        std::vector<double> x = {10};
        fitness_callback fitness = f_simple;
        gradient_callback gradient = g_simple;
        std::vector<double> delta = {-2};
        
        int num_variables = variable_types.size();
        int num_constraints = constraint_types.size() - 1;
        
        sensitivity_state sens_state;
        {
        	optgra_raii raii_object(variable_types, constraint_types);

			raii_object.initialize_sensitivity_data(x, fitness, gradient);

        	sens_state = raii_object.get_sensitivity_state_data();
        }

        std::vector<double> bestx, bestf, best_orig;
		int finopt;

		{
        	optgra_raii raii_object(variable_types, constraint_types);

        	raii_object.set_sensitivity_state_data(sens_state);

        	std::tie(bestx, bestf, finopt) = raii_object.sensitivity_update_delta(delta);
        }

        REQUIRE( bestx[0] == Approx(12.0) );
        best_orig = fitness(bestx);
        REQUIRE( best_orig[0] == Approx(delta[0]) );
}
