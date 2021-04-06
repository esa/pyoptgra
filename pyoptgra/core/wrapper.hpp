#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <iostream>
#include <functional>
#include <mutex>

extern"C" {
    void ogclos_();
    void ogcsca_(double * scacon);
    void ogctyp_(const int* contyp);
    void ogcpri_(int * pricon);
    void ogderi_(int * dervar, double * pervar);
    void ogdist_(double * maxvar, double * sndvar);
    void ogeval_(double * valvar, double * valcon, int * dervar, double * dercon,
         void (*)(double*, double*), void (*)(double*, double*, double*));
    void ogexec_(double * valvar, double * valcon, int * finopt, int * finite,
        void (*)(double*, double*, int*), void (*)(double*, double*, double*));
    void oginit_(int * varnum, int * connum);
    void ogiter_(int * itemax, int * itecor, int * iteopt, int * itediv, int * itecnv);
    void ogomet_(int * metopt);
    void ogvstr_(char ** strvar, int * lenvar);
    void ogcstr_(char ** strcon, int * lencon);
    void ogvsca_(double * scavar);
    void ogwlog_(int * lunlog, int * levlog);
}

namespace optgra {

    typedef std::function<std::vector<double>(std::vector<double>)> fitness_callback;

    typedef std::function<std::vector<std::vector<double>>(std::vector<double>)> gradient_callback;

    using std::vector, std::tuple, std::function;

struct parameters {
	
};

/** This struct is just to connect the std::functions passed from python
 *  to the unholy mess of static function pointers, which are requried by Fortran.
 *  It is emphatically not thread safe.
 */

struct static_callable_store {

    static void fitness(double * x, double * out_f, int * inapplicable_flag) {
        std::vector<double> x_vector(x_dim);
        std::copy(x, x+x_dim, x_vector.begin());

        std::vector<double> fitness_vector = f_callable(x_vector);

        std::copy(fitness_vector.begin(), fitness_vector.end(), out_f);
    }

    static void gradient(double * x, double * out_f, double * out_derivatives) {
        fitness(x, out_f, 0); // this can probably be optimized

        std::vector<double> x_vector(x_dim);
        std::copy(x, x+x_dim, x_vector.begin());

        std::vector<std::vector<double>> gradient_vector = g_callable(x_vector);

        int num_constraints = gradient_vector.size();

        for ( int i = 0; i < num_constraints; i++) {
            for (int j = 0; j < x_dim; j++) {
                //std::cout << "Writing " << gradient_vector[i][j] << " to " << j*num_constraints+i;
                out_derivatives[j*num_constraints+i] = gradient_vector[i][j];
                //std::cout << " ... done" << std::endl;
            }
        }
        //std::cout << "All done" << std::endl;
    }

    static void set_fitness_callable(fitness_callback f) {
        f_callable = f;
    }

    static void set_gradient_callable(gradient_callback g) {
        g_callable = g;
    }

    static void set_x_dim(int dim) {
        x_dim = dim;
    }

    static fitness_callback f_callable;
    static gradient_callback g_callable;
    static int x_dim;
};
// static initialization
fitness_callback static_callable_store::f_callable;
gradient_callback static_callable_store::g_callable;
int static_callable_store::x_dim;

struct optgra_raii {

    optgra_raii() = delete;

    optgra_raii(int num_variables, const std::vector<int> &constraint_types,
	    int max_iterations = 10, // MAXITE
		int max_correction_iterations = 10, // CORITE
		double max_distance_per_iteration = 10, // VARMAX
		double perturbation_for_snd_order_derivatives = 1, // VARSND
		std::vector<double> convergence_thresholds = {},
		std::vector<double> variable_scaling_factors = {},
		std::vector<int> constraint_priorities = {},
		std::vector<std::string> variable_names = {},
		std::vector<std::string> constraint_names = {},
		int optimization_method = 2, // OPTMET
		int derivatives_computation = 1, //VARDER
		std::vector<double> autodiff_deltas = {},
		int log_level = 1
	) : num_variables(num_variables)
    {
        num_constraints = constraint_types.size() - 1;
        if (autodiff_deltas.size() == 0) {
            autodiff_deltas = std::vector<double>(num_variables, 0.001);
        } else if (autodiff_deltas.size() != num_variables) {
        	throw(std::invalid_argument("Got " + std::to_string(autodiff_deltas.size())
        	 + " autodiff deltas for " + std::to_string(num_variables) + " variables."));
        }

        // TODO: more sanity checks for parameters.

        // Ensure that at most one optgra_raii object is active at the same time
        optgra_mutex.lock();

        oginit_(&num_variables, &num_constraints);
        ogctyp_(constraint_types.data());
        ogderi_(&derivatives_computation, autodiff_deltas.data());
        ogdist_(&max_distance_per_iteration, &perturbation_for_snd_order_derivatives);

        // Haven't figured out what the others do, but maxiter is an upper bound anyway
        int otheriters = max_iterations; // TODO: figure out what it does.
        ogiter_(&max_iterations, &max_correction_iterations, &otheriters, &otheriters, &otheriters);

        ogomet_(&optimization_method);
        

        int log_unit = 6;
        ogwlog_(&log_unit, &log_level);

        if (variable_scaling_factors.size() > 0) {
        	if (variable_scaling_factors.size() != num_variables) {
        		throw(std::invalid_argument("Got " + std::to_string(variable_scaling_factors.size())
        		 + " scaling factors for " + std::to_string(num_variables) + " variables."));
        	}

        	ogvsca_(variable_scaling_factors.data());
        }

        if (convergence_thresholds.size() > 0) {
        	if (convergence_thresholds.size() != constraint_types.size()) {
	        	throw(std::invalid_argument("Got " + std::to_string(convergence_thresholds.size())
	        	 + " convergence thresholds for " + std::to_string(constraint_types.size()) + " constraints+fitness."));
	        }
	        ogcsca_(convergence_thresholds.data());
	    }

	    if (constraint_priorities.size() > 0) {
        	if (constraint_priorities.size() != constraint_types.size()) {
        		//TODO: Find out what the last priority is for!
	        	throw(std::invalid_argument("Got " + std::to_string(constraint_priorities.size())
	        	 + " constraint priorities for " + std::to_string(constraint_types.size()) + " constraints+fitness."));
	        }
	        ogcpri_(constraint_priorities.data());
	    }

        //TODO: figure out how string arrays are passed to fortran for variable names
        
    }

    std::tuple<std::vector<double>, std::vector<double>, int> exec(std::vector<double> initial_x,
    fitness_callback fitness, gradient_callback gradient) {

        if (int(initial_x.size()) != num_variables) {
            throw(std::invalid_argument("Expected " + std::to_string(num_variables) + ", but got " + std::to_string(initial_x.size())));
        }

        std::vector<double> valvar(initial_x);
        std::vector<double> valcon(num_constraints+1);

        static_callable_store::set_fitness_callable(fitness);
        static_callable_store::set_gradient_callable(gradient);
        static_callable_store::set_x_dim(initial_x.size());

        int finopt = 0;
        int finite = 0;
        ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
         static_callable_store::fitness, static_callable_store::gradient);

        return std::make_tuple(valvar, valcon, finopt);
    }

    ~optgra_raii()
    {
        ogclos_();
        optgra_mutex.unlock();
    }

private:
    int num_variables;
    int num_constraints;

    static std::mutex optgra_mutex;
};

std::mutex optgra_raii::optgra_mutex;

std::tuple<std::vector<double>, std::vector<double>, int> optimize(const std::vector<double> &initial_x,
 const std::vector<int> &constraint_types, fitness_callback fitness, gradient_callback gradient, bool has_gradient,
 	    int max_iterations = 10, // MAXITE
		int max_correction_iterations = 10, // CORITE
		double max_distance_per_iteration = 10, // VARMAX
		double perturbation_for_snd_order_derivatives = 1, // VARSND
		std::vector<double> convergence_thresholds = {},
		std::vector<double> variable_scaling_factors = {},
		std::vector<int> constraint_priorities = {},
		std::vector<std::string> variable_names = {},
		std::vector<std::string> constraint_names = {},
		int optimization_method = 2, // OPTMET
		int derivatives_computation = 1, //VARDER
		std::vector<double> autodiff_deltas = {},
		int log_level = 1
 ) {
    // initialization
    int num_variables = initial_x.size();

    if (derivatives_computation == 1 && !has_gradient) {
    	std::cout << "No user-defined gradient available, switching to numeric differentiation." << std::endl;
    	derivatives_computation = 3;
    }

    optgra_raii raii_object = optgra_raii(num_variables, constraint_types,
    	max_iterations, // MAXITE
		max_correction_iterations, // CORITE
		max_distance_per_iteration, // VARMAX
		perturbation_for_snd_order_derivatives, // VARSND
		convergence_thresholds,
		variable_scaling_factors,
		constraint_priorities,
		variable_names,
		constraint_names,
		optimization_method, // OPTMET
		derivatives_computation, //VARDER
		autodiff_deltas,
		log_level);

    return raii_object.exec(initial_x, fitness, gradient);
}

}

/**
 * oginit.F : Allocates and zeroes vectors, sets parameter values in common block to hardcoded defaults
 * ogvsca.F : Define variable scale factor - int[numvar]
 * ogvstr.F : Set variable names - str[numvar]
 * ogctyp.F : Sets types of constraints and merit function in common block
 * ogcpri.F : Sets constraint priorities in common block int[numcon+1]
 * ogcsca.F : Sets convergence thresholds of constraints and merit function in common block - double[NUMCON+1]
 * ogcstr.F : Sets names of constraints and merit function in common block
 * 
 * ogderi.F : Sets parameters for type of derivative computation in common block
 * ogomet.F : Set optimization method parameter in common block
 * ogiter.F : Set optimization parameters in common block (ITEMAX, ITECOR, ITEOPT, ITEDIV, ITECNV)
 * ogdist.F : Sets parameters for maximum distance per iteration and perturbation in common block
 * 
 * ogwlog.F : Define writing in log file
 * ogwmat.F : Something with writing to Matlab
 * ogwtab.F : writes units and verbosity(?) options for tabular output into the common block defined in ogdata.inc

*/
