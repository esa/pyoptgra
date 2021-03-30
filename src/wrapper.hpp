#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <iostream>
#include <pagmo/problem.hpp>

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

    typedef void (*fitness_callback)(double*, double*, int*);

    typedef void (*gradient_callback)(double*, double*, double*);

struct parameters {
	int max_iterations = 10; // MAXITE
	int max_correction_iterations = 10; // CORITE
	double max_distance_per_iteration = 10; // VARMAX
	double perturbation_for_snd_order_derivatives = 1; // VARSND
	std::vector<double> convergence_thresholds;
	std::vector<double> variable_scaling_factors;
	std::vector<int> constraint_priorities;
	std::vector<std::string> variable_names;
	std::vector<std::string> constraint_names;
	int optimization_method = 2; // OPTMET
	int derivatives_computation = 1; //VARDER
	std::vector<double> autodiff_deltas;
	int log_level = 1;
};

struct optgra_raii {

    optgra_raii() = delete;
    // TODO: Use a mutex to ensure that at most one object can be created concurrently

    optgra_raii(int num_variables, const std::vector<int> &constraint_types,
     parameters params) : num_variables(num_variables)
    {
        num_constraints = constraint_types.size() - 1;
        if (params.autodiff_deltas.size() == 0) {
            params.autodiff_deltas = std::vector<double>(num_variables, 0.001);
        } else if (params.autodiff_deltas.size() != num_variables) {
        	throw(std::invalid_argument("Got " + std::to_string(params.autodiff_deltas.size())
        	 + " autodiff deltas for " + std::to_string(num_variables) + " variables."));
        }

        // TODO: more sanity checks for parameters.

        oginit_(&num_variables, &num_constraints);
        ogctyp_(constraint_types.data());
        ogderi_(&params.derivatives_computation, params.autodiff_deltas.data());
        ogdist_(&params.max_distance_per_iteration, &params.perturbation_for_snd_order_derivatives);

        // Haven't figured out what the others do, but maxiter is an upper bound anyway
        int otheriters = params.max_iterations; // TODO: figure out what it does.
        ogiter_(&params.max_iterations, &params.max_correction_iterations, &otheriters, &otheriters, &otheriters);

        ogomet_(&params.optimization_method);
        

        int log_unit = 6;
        ogwlog_(&log_unit, &params.log_level);

        if (params.variable_scaling_factors.size() > 0) {
        	if (params.variable_scaling_factors.size() != num_variables) {
        		throw(std::invalid_argument("Got " + std::to_string(params.variable_scaling_factors.size())
        		 + " scaling factors for " + std::to_string(num_variables) + " variables."));
        	}

        	ogvsca_(params.variable_scaling_factors.data());
        }

        if (params.convergence_thresholds.size() > 0) {
        	if (params.convergence_thresholds.size() != constraint_types.size()) {
	        	throw(std::invalid_argument("Got " + std::to_string(params.convergence_thresholds.size())
	        	 + " convergence thresholds for " + std::to_string(constraint_types.size()) + " constraints+fitness."));
	        }
	        ogcsca_(params.convergence_thresholds.data());
	    }

	    if (params.constraint_priorities.size() > 0) {
        	if (params.constraint_priorities.size() != constraint_types.size()) {
        		//TODO: Find out what the last priority is for!
	        	throw(std::invalid_argument("Got " + std::to_string(params.constraint_priorities.size())
	        	 + " constraint priorities for " + std::to_string(constraint_types.size()) + " constraints+fitness."));
	        }
	        ogcpri_(params.constraint_priorities.data());
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

        int finopt = 0;
        int finite = 0;
        ogexec_(valvar.data(), valcon.data(), &finopt, &finite, fitness, gradient);

        return std::make_tuple(valvar, valcon, finopt);
    }

    ~optgra_raii()
    {
        ogclos_();
    }

private:
    int num_variables;
    int num_constraints;
};

struct problem_wrapper {
    // TODO: set mutex to ensure thread-safety
    static void set_problem(pagmo::problem &problem) {

        if (prob.get_nobj() != 1u) {
            throw(std::invalid_argument("Multiple objectives detected in " + prob.get_name() + " instance. Optgra cannot deal with them"));
        }
        if (prob.is_stochastic()) {
            throw(std::invalid_argument("The problem appears to be stochastic. Optgra cannot deal with it"));
        }

        // set problem, get problem dimension
        prob = problem;
        dim = prob.get_nx();

        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;

        // add equality constraints
        for (unsigned i = 0; i < prob.get_nec(); i++) {
            constraint_types.push_back(0);
        }

        // add inequality constraints
        for (unsigned i = 0; i < prob.get_nic(); i++) {
            constraint_types.push_back(-1);
        }

        //TODO: add box bounds

        // fitness function is the last and pagmo problems always minimize
        constraint_types.push_back(-1); 

    }

    static const std::vector<int>& get_constraint_types() {
        return constraint_types;
    }

    static void fitness(double * x, double * out_f, int * inapplicable_flag) {
        std::vector<double> x_vector(dim);
        std::copy(x, x+dim, x_vector.begin());

        std::vector<double> fitness_vector = prob.fitness(x_vector);
        unsigned f_length = fitness_vector.size();

        if (f_length != constraint_types.size()) {
            throw(std::runtime_error("Fitness returned " + std::to_string(f_length) 
                + " but expected " + std::to_string(constraint_types.size())));
        }

        // pagmo has fitness first, followed by constraints
        // optgra has constraints first and fitness last
        // we rotate to fit
        std::copy(fitness_vector.begin()+1, fitness_vector.end(), out_f);
        out_f[f_length-1] = fitness_vector[0];
    }

    static bool has_gradient() {
        return prob.has_gradient();
    }

    static void gradient(double * x, double * out_f, double * out_derivatives) {
        if (!has_gradient()) {
            throw(std::logic_error("Problem " + prob.get_name() +
                " has no gradient, but the gradient function was called. This is probably a state inconsistency of the problem wrapper."));
        }
        fitness(x, out_f, 0);

        std::vector<double> x_vector(dim);
        std::copy(x, x+dim, x_vector.begin());

        unsigned num_con = constraint_types.size()-1;
        // Zero out derivatives
        // The gradient structure of optgra is DERCON(NUMCON+1,NUMVAR)
        // Internally, optgra allocates some additional rows as working memory,
        // but we will not concern ourselves with zeroing them.
        std::fill(out_derivatives, (out_derivatives+(num_con+1)*dim), 0);

        pagmo::sparsity_pattern gs_map = prob.gradient_sparsity();
        static std::vector<double> compressed_gradient = prob.gradient(x_vector);

        //already checked by pagmo, more a reminder for myself
        assert(gs_map.size() == compressed_gradient.size());

        // copy gradient into dense array of optgra
        for (unsigned i = 0; i < gs_map.size(); i++) {
            int xi, fi, target_fi;
            std::tie(xi, fi) = gs_map[i];

            // again, have to rotate the gradients, as pagmo has fitness first, while optgra has fitness last
            if (fi == 0) {
                target_fi = num_con; // last element of array of size num_con+1, in 0 indexing
            } else {
                target_fi = fi - 1;
            }

            // TODO: check that row/column ordering is translated correctly to fortran.
            out_derivatives[target_fi*dim+xi] = compressed_gradient[i];
        }
    }

    static pagmo::problem& prob;
    static unsigned dim;
    static std::vector<int> constraint_types;
};

// static initialization, this is horrible.
pagmo::problem null_prob = pagmo::problem();
pagmo::problem& problem_wrapper::prob = null_prob;
unsigned problem_wrapper::dim;
std::vector<int> problem_wrapper::constraint_types;

template<class F, class G>
std::tuple<std::vector<double>, std::vector<double>, int> optimize(const std::vector<double> &initial_x,
 const std::vector<int> &constraint_types, F fitness, G gradient, bool has_gradient,
 parameters params = {}
 ) {

    // initialization
    int num_variables = initial_x.size();

    if (params.derivatives_computation == 1 && !has_gradient) {
    	std::cout << "No user-defined gradient available, switching to numeric differentiation." << std::endl;
    	params.derivatives_computation = 3;
    }

    optgra_raii raii_object = optgra_raii(num_variables, constraint_types, params);

    return raii_object.exec(initial_x, fitness, gradient);
}

std::tuple<std::vector<double>, std::vector<double>> optimize(pagmo::problem prob, const std::vector<double> &initial_x,
    const parameters params = {}) {
    problem_wrapper::set_problem(prob);
    auto result_tuple = optimize(initial_x, problem_wrapper::get_constraint_types(), problem_wrapper::fitness,
    problem_wrapper::gradient, prob.has_gradient(), params);

    std::vector<double> best_x = std::get<0>(result_tuple);
    std::vector<double> best_f = std::get<1>(result_tuple);
    int n_out = best_f.size();

    // reorder merit/constraints into format used by pagmo. That means fitness first, constraints then.
    // Alas, we need a right rotation, not a left one, which is why we cannot use std::rotate. Maybe move_backward?

    double fitness = best_f[n_out-1];
    for (int i = n_out-1; i > 0; i--) {
        best_f[i] = best_f[i-1];
    }
    best_f[0] = fitness;

    return std::make_tuple(best_x, best_f);
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

 Prioritize: convergence thresholds

*/
