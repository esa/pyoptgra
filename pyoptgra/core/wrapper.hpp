#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <iostream>
#include <functional>
#include <mutex>

extern"C" {
    void ogcdel_(double * delcon);
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
    void ogsens_(int * consta, double * concon, double * convar, double * varcon, double * varvar);
    void ogsopt_(int * optsen);
    void ogvstr_(char ** strvar, int * lenvar);
    void ogcstr_(char ** strcon, int * lencon);
    void ogvsca_(double * scavar);
    void ogwlog_(int * lunlog, int * levlog);
}


namespace optgra {

    typedef std::function<std::vector<double>(std::vector<double>)> fitness_callback;

    typedef std::function<std::vector<std::vector<double>>(std::vector<double>)> gradient_callback;

    using std::vector;
    using std::tuple;
    using std::function;

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
        if (int(fitness_vector.size()) != c_dim) {
            throw(std::invalid_argument("Got vector of size" + std::to_string(fitness_vector.size())
                 + " from fitness callable, but expected " + std::to_string(c_dim) + " constraints+fitness."));
        }

        std::copy(fitness_vector.begin(), fitness_vector.end(), out_f);
    }

    static void gradient(double * x, double * out_f, double * out_derivatives) {
        fitness(x, out_f, 0); // this can probably be optimized

        std::vector<double> x_vector(x_dim);
        std::copy(x, x+x_dim, x_vector.begin());

        std::vector<std::vector<double>> gradient_vector = g_callable(x_vector); //TODO: check for correct dimension of return value

        int num_constraints = gradient_vector.size();
        if (num_constraints != c_dim) {
            throw(std::invalid_argument("Got vector of size" + std::to_string(num_constraints)
                 + " from gradient callable, but expected " + std::to_string(c_dim) + " constraints+fitness."));
        }

        for ( int i = 0; i < num_constraints; i++) {
            if (int(gradient_vector[i].size()) != x_dim) {
                throw(std::invalid_argument("Got vector of size" + std::to_string(int(gradient_vector[i].size()))
                 + " from row " + std::to_string(i) + " of gradient callable, but expected " + std::to_string(x_dim) + " variables."));
            }
            for (int j = 0; j < x_dim; j++) {
                out_derivatives[j*num_constraints+i] = gradient_vector[i][j];
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

    static void set_c_dim(int dim) {
        c_dim = dim;
    }

    static fitness_callback f_callable;
    static gradient_callback g_callable;
    static int x_dim;
    static int c_dim;
};
// static initialization
fitness_callback static_callable_store::f_callable;
gradient_callback static_callable_store::g_callable;
int static_callable_store::x_dim;
int static_callable_store::c_dim;

struct optgra_raii {

    optgra_raii() = delete;

    optgra_raii(int num_variables, const std::vector<int> &constraint_types,
	    int max_iterations = 150, // MAXITE
		int max_correction_iterations = 90, // CORITE
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
            if (!convergence_thresholds[convergence_thresholds.size()-1] > 0) {
                throw(std::invalid_argument("Convergence threshold for merit function must be positive."));
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
        static_callable_store::set_c_dim(num_constraints+1);

        int finopt = 0;
        int finite = 0;
        ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
         static_callable_store::fitness, static_callable_store::gradient);

        // resetting callables to make sure that passed handles go out of scope
        static_callable_store::set_fitness_callable(fitness_callback());
        static_callable_store::set_gradient_callable(gradient_callback());

        return std::make_tuple(valvar, valcon, finopt);
    }

    std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
     std::vector<std::vector<double>>, std::vector<std::vector<double>>> sens(std::vector<double> initial_x, int sensitivity_mode,
     fitness_callback fitness, gradient_callback gradient, std::vector<double> constraint_deltas = {} ) {

        if (int(initial_x.size()) != num_variables) {
            throw(std::invalid_argument("Expected " + std::to_string(num_variables) + ", but got " + std::to_string(initial_x.size())));
        }

        if (!(sensitivity_mode == -1 || sensitivity_mode == 1 || sensitivity_mode == 2)) {
            throw(std::invalid_argument("Expected sensitivity_mode to be one of -1, 1, or 2, but got " + std::to_string(sensitivity_mode)));
        }

        std::vector<double> valvar(initial_x);
        std::vector<double> valcon(num_constraints+1);

        static_callable_store::set_fitness_callable(fitness);
        static_callable_store::set_gradient_callable(gradient);
        static_callable_store::set_x_dim(initial_x.size());
        static_callable_store::set_c_dim(num_constraints+1);

        ogsopt_(&sensitivity_mode);

        if (sensitivity_mode == 2) {
            if (constraint_deltas.size() != num_constraints) {
                throw(std::invalid_argument("Expected " + std::to_string(num_constraints) + ", but got " + std::to_string(constraint_deltas.size())));
            }
            ogcdel_(constraint_deltas.data());
        }

        int finopt = 0;
        int finite = 0;
        ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
         static_callable_store::fitness, static_callable_store::gradient);

        // resetting callables to make sure that passed handles go out of scope
        static_callable_store::set_fitness_callable(fitness_callback());
        static_callable_store::set_gradient_callable(gradient_callback());

        // allocate flattened sensitivity matrices
        int x_dim = initial_x.size();
        std::vector<int> constraint_status(num_constraints);
        std::vector<double> concon((num_constraints+1)*num_constraints);
        std::vector<double> convar((num_constraints+1)*x_dim);
        std::vector<double> varcon(x_dim*num_constraints);
        std::vector<double> varvar(x_dim*x_dim);

        // call ogsens
        ogsens_(constraint_status.data(), concon.data(), convar.data(), varcon.data(), varvar.data());
        /**
        C OUT | CONSTA(NUMCON)   | I*4 | CONSTRAINT STATUS (0=PAS 1=ACT)
        C OUT | CONCON(NUMCON+1, | R*8 | SENSITIVITY OF CONTRAINTS+MERIT W.R.T.
        C     |        NUMCON)   |     |                ACTIVE CONSTRAINTS
        C OUT | CONVAR(NUMCON+1, | R*8 | SENSITIVITY OF CONTRAINTS+MERIT W.R.T.
        C     |        NUMVAR)   |     |                PARAMETERS
        C OUT | VARCON(NUMVAR  , | R*8 | SENSITIVITY OF VARIABLES W.R.T.
        C     |        NUMCON)   |     |                ACTIVE CONSTRAINTS
        C OUT | VARVAR(NUMVAR  , | R*8 | SENSITIVITY OF VARIABLES W.R.T.
        C     |        NUMVAR)   |     |                PARAMETERS
        */

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

/// Main C++ wrapper function
/**
 * Call optgra to optimize a problem. Most of the parameters are identical to the constructor arguments of pyoptgra,
 *    but some additional ones are available.
 *
 * @param initial_x the initial guess for the decision vector
 * @param constraint_types types of constraints. Set 0 for equality constraints,
 *    -1 for inequality constraints that should be negative, 1 for positive inequality constraints
 * @param fitness a callable for the fitness values. It is called with the current x,
 *    expected output is an array of first all equality constraints, then all inequality constraints, and last the merit function
 * @param gradient a callable for the gradient values, optional. It is called with the current x,
 *    expected output is a two-dimensional array g, with g_ij being the gradient of constraint i with respect to input variable j.
 * @param has_gradient whether the problem has a gradient. If set to False, the gradient callable will not be called
 *    and numerical differentiation will be used instead.
 * @param max_iterations the maximum number of iterations. Optional, defaults to 150.
 * @param max_correction_iterations number of constraint correction iterations in the beginning.
 *    If no feasible solution is found within that many iterations, Optgra aborts. Optional, defaults to 90.
 * @param max_distance_per_iteration maximum scaled distance traveled in each iteration. Optional, defaults to 10 
 * @param perturbation_for_snd_order_derivatives used as delta for numerically computing second order errors
 *    of the constraints in the optimization step. Parameter VARSND in Fortran. Optional, defaults to 1
 * @param convergence_thresholds tolerance a constraint can deviate and still be considered fulfilled.
 *    Constraints with lower thresholds will be prioritized during optimization. Thresholds of 0 break the optimization process.
 * @param variable_scaling_factors scaling factors for the input variables.
 *    If passed, must be positive and as many as there are variables
 * @param constraint_priorities filter in which to consider constraints. Lower constraint priorities are fulfilled earlier.
 *    During the initial constraint correction phase, only constraints with a priority at most k
 *    are considered in iteration k. Defaults to zero, so that all constraints are considered
 *    from the beginning.
 * @param variable_names Not yet implemented
 * @param constraint_names Not yet implemented
 * @param optimization_method select 0 for steepest descent, 1 for modified spectral conjugate gradient method,
 *    2 for spectral conjugate gradient method and 3 for conjugate gradient method. Parameter OPTMET in Fortran.
 * @param derivatives_computation method to compute gradients. 0 is no gradient, 1 is the user-defined gradient function,
 *    2 is a numerical gradient with double differencing, 3 a numerical gradient with single differencing.
 *    Parameter VARDER in Fortran.
 * @param autodiff_deltas deltas used for each variable when computing the gradient numerically. Optional, defaults to 0.001.
 * @param log_level 0 has no output, 4 and higher have maximum output
 *
 * @return a tuple of the best value of x, the fitness of that x, and a status flag of optgra
 *
 * @throws unspecified any exception thrown by memory errors in standard containers
 */
std::tuple<std::vector<double>, std::vector<double>, int> optimize(const std::vector<double> &initial_x,
 const std::vector<int> &constraint_types, fitness_callback fitness, gradient_callback gradient, bool has_gradient,
 	    int max_iterations = 150, // MAXITE
		int max_correction_iterations = 90, // CORITE
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

std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
     std::vector<std::vector<double>>, std::vector<std::vector<double>>> sensitivity(const std::vector<double> &initial_x,
 const std::vector<int> &constraint_types, fitness_callback fitness, gradient_callback gradient, bool has_gradient,
        int sensitivity_mode, std::vector<double> constraint_deltas = {}
 ) {

    int num_variables = initial_x.size();

    optgra_raii raii_object = optgra_raii(num_variables, constraint_types);

    return raii_object.sens(initial_x, sensitivity_mode, fitness, gradient, constraint_deltas);
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
