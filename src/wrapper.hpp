#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <pagmo/problem.hpp>

extern"C" {
	void ogclos_();
	void ogctyp_(int * contyp);
	void ogderi_(int * dervar, double *pervar);
	void ogeval_(double * valvar, double * valcon, int * dervar, double * dercon,
	 	void (*)(double*, double*), void (*)(double*, double*, double*));
	void ogexec_(double * valvar, double * valcon, int * finopt, int * finite,
		void (*)(double*, double*, int*), void (*)(double*, double*, double*));
	void oginit_(int * varnum, int * connum);
}

namespace optgra {

    typedef void (*fitness_callback)(double*, double*, int*);

    typedef void (*gradient_callback)(double*, double*, double*);

struct optgra_raii {

    optgra_raii() = delete;
    // TODO: Use a mutex to ensure that at most one object can be created concurrently

    optgra_raii(int num_variables, const std::vector<int> &constraint_types,
     int difftype, std::vector<double> autodiff_deltas = {}) : num_variables(num_variables)
    {
        num_constraints = constraint_types.size() - 1;
        if (autodiff_deltas.size() == 0) {
            autodiff_deltas = std::vector<double>(num_variables, 0.1);
        }
        oginit_(&num_variables, &num_constraints);
        ogderi_(&difftype, autodiff_deltas.data());
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

    static void set_problem(pagmo::problem &prob) {

        if (prob.get_nobj() != 1u) {
            throw(std::invalid_argument("Multiple objectives detected in " + prob.get_name() + " instance. Optgra cannot deal with them"));
        }
        if (prob.is_stochastic()) {
            throw(std::invalid_argument("The problem appears to be stochastic. Optgra cannot deal with it"));
        }

        // get problem dimension
        prob = prob;
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

        std::copy(fitness_vector.begin()+1, fitness_vector.end(), out_f);
        out_f[f_length-1] = fitness_vector[0];
    }

    static void gradient(double * x, double * out_f, double * out_derivatives) {
        fitness(x, out_f, 0);

        // TODO unpack gradient
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
 const std::vector<int> &constraint_types, F fitness, G gradient, bool has_gradient, double autodiff_epsilon=0.1) {
	
    // initialization
    int num_variables = initial_x.size();

    int difftype; // user-defined gradient
    if (!has_gradient) {
        difftype = 3; // numeric differentiation
    } else {
        difftype = 1;
    }

    std::vector<double> autodiff_deltas(num_variables, autodiff_epsilon);

    optgra_raii raii_object = optgra_raii(num_variables, constraint_types, difftype, autodiff_deltas);

    return raii_object.exec(initial_x, fitness, gradient);
}

std::tuple<std::vector<double>, std::vector<double>> optimize(pagmo::problem prob, const std::vector<double> &initial_x) {
    problem_wrapper::set_problem(prob);
    auto result_tuple = optimize(initial_x, problem_wrapper::get_constraint_types(), problem_wrapper::fitness,
    problem_wrapper::gradient, prob.has_gradient());
    return std::make_tuple(std::get<0>(result_tuple), std::get<1>(result_tuple));
}

}

/**
 * oginit.F : Allocates and zeroes vectors, sets parameter values in common block to hardcoded defaults
 * ogvsca.F : Define variable scale factor
 * ogvstr.F : Set variable names
 * ogctyp.F : Sets types of constraints and merit function in common block
 * ogcpri.F : Sets constraint priorities in common block
 * ogcsca.F : Sets convergence thresholds of constraints and merit function in common block
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


C INP | TYPCON(NUMCON+1) | I*4 | CONSTRAINTS TYPE (1:NUMCON)
C     |                  |     | -> 1=GTE -1=LTE 0=EQU -2=DERIVED DATA
C     |                  |     | MERIT       TYPE (1+NUMCON)
C     |                  |     | -> 1=MAX -1=MIN

*/
