#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>

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
        num_constraints = constraint_types.size();
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

template<class F, class G>
std::tuple<std::vector<double>, std::vector<double>, int> optimize(const std::vector<double> &initial_x,
 const std::vector<int> &constraint_types, F fitness, G gradient, bool has_gradient, double autodiff_epsilon=0.1) {
	
    // initialization
    int num_variables = initial_x.size();
    int num_constraints = constraint_types.size();
    std::vector<double> valvar(initial_x);
    std::vector<double> valcon(num_constraints+1);

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

*/