#include <iostream>
#include <vector>
#include <tuple>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "wrapper.hpp"

using std::cout;
using std::endl;
using std::vector;

using namespace optgra;
using Catch::Approx;

///Fitness callable implementing a constraint function of \sum_0^4 (x_i - i)^2 and a merit function of \sum_0^4 x_i
std::vector<double> f(std::vector<double> x) {
	std::vector<double> con(2);
	con[0] = 0;
	con[1] = 0;
	cout << "f called with ";
	for (int i = 0; i < 5; i++) {
		con[0] += (x[i]-i)*(x[i]-i);
		con[1] += x[i];
		cout << x[i] << " ";
	}
	cout << endl;
	return con;
}

///Gradient callable implementing a constraint function of \sum_0^4 (x_i - i)^2 and a merit function of \sum_0^4 x_i 
std::vector<std::vector<double>> g(std::vector<double> x) {
	cout << "g called with ";
	int x_dim = x.size();
	int num_constraints = 2;
	std::vector<std::vector<double>> der(2);
	der[0].resize(x_dim);
	der[1].resize(x_dim);

	for (int i = 0; i < x_dim; i++) {
		der[0][i] = 2*(x[i]-i);
		der[1][i] = 1;
		cout << x[i] << " ";
	}
	
	cout << endl;
	return der;
}

std::vector<double> f_simple(std::vector<double> x) {
	std::vector<double> con(2);
	con[0] = 10 - x[0];
	con[1] = x[0];
	cout << "f_simple called with " << x[0];
	cout << endl;
	return con;
}


std::vector<std::vector<double>> g_simple(std::vector<double> x) {
	cout << "g_simple called with " << x[0] << endl;
	std::vector<std::vector<double>> der(2);
	der[0].resize(1);
	der[1].resize(1);

	der[0][0] = -1;
	der[1][0] = 1;
	return der;
}

TEST_CASE( "RAII initialization works and allows optimization", "[raii-exec]" )
{
	std::vector<double> initial_x = {1,1,1,1,1};
	std::vector<double> bestx, bestf;
	int finopt;
	
	int num_variables = initial_x.size();

    int derivatives_computation = 1;
    int num_constraints = 1;

    std::vector<int> variable_types(num_variables, 0);
    
    optgra_raii raii_object(variable_types, {0,-1});
    
    std::tie(bestx, bestf, finopt) = raii_object.exec(initial_x, f, g);

    // check that bestx is close to 0, 1, 2, 3, 4
    REQUIRE (bestx[0] == Approx(0.0));
    REQUIRE (bestx[1] == Approx(1.0));
    REQUIRE (bestx[2] == Approx(2.0));
    REQUIRE (bestx[3] == Approx(3.0));
    REQUIRE (bestx[4] == Approx(4.0));
  
    // check that bestf is close to 0 10
    REQUIRE (bestf[0] == Approx(0.0));
    REQUIRE (bestf[1] == Approx(10.0));
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

TEST_CASE( "RAII supports sensitivity updates with new callable", "[raii-sensitivity-update]" )
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

TEST_CASE( "RAII supports sensitivity updates with constraint delta", "[raii-sensitivity-update-delta]" )
{
	std::vector<double> sens_x = {0, 0.99, 2, 3, 4};
	int num_variables = sens_x.size();
	int derivatives_computation = 1;
    int num_constraints = 1;

	std::vector<int> variable_types(num_variables, 0);
	optgra_raii raii_object(variable_types, {0,-1});

	raii_object.initialize_sensitivity_data(sens_x, f, g);

	cout << endl << "Sensitivity Update Constraint Delta Test" << endl;

	std::vector<double> delta = {-1};

	std::vector<double> bestx, bestf, best_orig;
	int finopt;

	std::tie(bestx, bestf, finopt) = raii_object.sensitivity_update_delta(delta);

	REQUIRE( bestf[0] == Approx(0.0) );

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