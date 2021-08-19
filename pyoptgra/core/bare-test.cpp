#include <iostream>
#include <vector>
#include <tuple>

#include "wrapper.hpp"

using std::cout;
using std::endl;
using std::vector;

using namespace optgra;

std::vector<double> f_simple(std::vector<double> x) {
        std::vector<double> con(2);
        con[0] = 10 - x[0];
        con[1] = 2*x[0];
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
        der[1][0] = 2;
        return der;
}

std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
     std::vector<std::vector<double>>, std::vector<std::vector<double>>> call_optgra(const std::vector<int> &variable_types, const std::vector<int> &constraint_types,
        std::vector<double> x,
        fitness_callback fitness,
        gradient_callback gradient,
        int max_iterations = 150, // MAXITE
        int max_correction_iterations = 90, // CORITE
        double max_distance_per_iteration = 1, // VARMAX
        double perturbation_for_snd_order_derivatives = 1) // VARSND) 
        {

        int num_variables = variable_types.size();
        int num_constraints = constraint_types.size() - 1;
        std::vector<double> autodiff_deltas = std::vector<double>(num_variables, 0.01);
        int derivatives_computation = 2;

        oginit_(&num_variables, &num_constraints);
        ogctyp_(constraint_types.data());
        ogderi_(&derivatives_computation, autodiff_deltas.data());
        ogdist_(&max_distance_per_iteration, &perturbation_for_snd_order_derivatives);

        ogvtyp_(variable_types.data());

        // Haven't figured out what the others do, but maxiter is an upper bound anyway
        int otheriters = max_iterations; // TODO: figure out what it does.
        ogiter_(&max_iterations, &max_correction_iterations, &otheriters, &otheriters, &otheriters);

        int optimization_method = 2;
        ogomet_(&optimization_method);

        int log_unit = 6;
        int log_level = 1;
        ogwlog_(&log_unit, &log_level);

        int finopt = 0;
        int finite = 0;

        std::vector<double> valvar(x);
        std::vector<double> valcon(num_constraints+1);

        static_callable_store::set_fitness_callable(fitness);
        static_callable_store::set_gradient_callable(gradient);
        static_callable_store::set_x_dim(num_variables);
        static_callable_store::set_c_dim(num_constraints+1);

        //ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
        // static_callable_store::fitness, static_callable_store::gradient);

        int sensitivity_mode = -1;
        ogsopt_(&sensitivity_mode);
        
        valvar = x;
        ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
         static_callable_store::fitness, static_callable_store::gradient);

        //call and output oggsst to compare
        vector<double> senvar(num_variables);
        vector<double> senqua(num_constraints+1);
        vector<double> sencon(num_constraints+1);
        vector<int> senact(num_constraints+1);
        vector<double> sender((num_constraints+1)*num_variables);
        vector<int> actcon(num_constraints+1);
        vector<int> conact(num_constraints+4);
        vector<double> conred((num_constraints+3)*num_variables);

        oggsst_(senvar.data(), senqua.data(), sencon.data(), senact.data(), sender.data(), actcon.data(), conact.data(), conred.data());

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

        int x_dim = num_variables;
        std::vector<int> constraint_status(num_constraints);
        std::vector<double> concon((num_constraints+1)*num_constraints);
        std::vector<double> convar((num_constraints+1)*x_dim);
        std::vector<double> varcon(x_dim*num_constraints);
        std::vector<double> varvar(x_dim*x_dim);

        // call ogsens
        ogsens_(constraint_status.data(), concon.data(), convar.data(), varcon.data(), varvar.data());
        
        // allocate unflattened sensitivity matrices
        std::vector<std::vector<double>> constraints_to_active_constraints(num_constraints+1);
        std::vector<std::vector<double>> constraints_to_parameters(num_constraints+1);
        std::vector<std::vector<double>> variables_to_active_constraints(x_dim);
        std::vector<std::vector<double>> variables_to_parameters(x_dim);

        // copy values for constraints_to_active_constraints and constraints_to_parameters
        for ( int i = 0; i < (num_constraints+1); i++) {
            constraints_to_active_constraints[i].resize(num_constraints);
            constraints_to_parameters[i].resize(x_dim);

            for (int j = 0; j < num_constraints; j++) {
                constraints_to_active_constraints[i][j] = concon[j*num_constraints+i];
            }

            for (int j = 0; j < x_dim; j++) {
                constraints_to_parameters[i][j] = convar[j*num_constraints+i];
            }
        }

        // copy values for variables_to_active_constraints and variables_to_parameters
        for ( int i = 0; i < x_dim; i++) {
            variables_to_active_constraints[i].resize(num_constraints);
            variables_to_parameters[i].resize(x_dim);

            for (int j = 0; j < num_constraints; j++) {
                variables_to_active_constraints[i][j] = varcon[j*x_dim+i];
            }

            for (int j = 0; j < x_dim; j++) {
                variables_to_parameters[i][j] = varvar[j*x_dim+i];
            }
        }

        return std::make_tuple(constraint_status, constraints_to_active_constraints, constraints_to_parameters,
         variables_to_active_constraints, variables_to_parameters);
}

int main(int argn, char** argc)
{
        int num_constraints = 1;
        int num_variables = 1;
        std::vector<int> constraint_status(num_constraints);
        std::vector<std::vector<double>> constraints_to_active_constraints(num_constraints+1);
        std::vector<std::vector<double>> constraints_to_parameters(num_constraints+1);
        std::vector<std::vector<double>> variables_to_active_constraints(num_variables);
        std::vector<std::vector<double>> variables_to_parameters(num_variables);

        std::tie(constraint_status, constraints_to_active_constraints, constraints_to_parameters,
         variables_to_active_constraints, variables_to_parameters) = call_optgra({0}, {-1,-1}, {10}, f_simple, g_simple);

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