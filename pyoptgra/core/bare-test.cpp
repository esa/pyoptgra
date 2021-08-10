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

vector<int> call_optgra(const std::vector<int> &variable_types, const std::vector<int> &constraint_types,
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

        int x_dim = num_variables;
        std::vector<int> constraint_status(num_constraints);
        std::vector<double> concon((num_constraints+1)*num_constraints);
        std::vector<double> convar((num_constraints+1)*x_dim);
        std::vector<double> varcon(x_dim*num_constraints);
        std::vector<double> varvar(x_dim*x_dim);

        // call ogsens
        ogsens_(constraint_status.data(), concon.data(), convar.data(), varcon.data(), varvar.data());
        return constraint_status;

}

int main(int argn, char** argc)
{
        std::vector<int> constraint_status = call_optgra({0}, {-1,-1}, {10}, f_simple, g_simple);
        int num_constraints = 1;
        cout << "Active constraints:" << endl;
        for (int i = 0; i < num_constraints; i++) {
                cout << constraint_status[i] << " ";
        }
        cout << endl;
}