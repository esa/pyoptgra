#pragma once

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

template<class F, class G>
std::tuple<std::vector<double>, std::vector<double>, int> optimize(const std::vector<double> &initial_x,
 const std::vector<int> &constraint_types, F fitness, G gradient, bool has_gradient, double autodiff_epsilon=0.1) {
	
	// initialization
	int num_variables = initial_x.size();
	int num_constraints = constraint_types.size();
	std::vector<double> valvar(initial_x);
	std::vector<double> valcon(num_constraints+1);
	oginit_(&num_variables, &num_constraints);

	int dervar = 1; // user-defined gradient
	if (!has_gradient) {
		dervar = 3; // numeric differentiation
	}

	std::vector<double> pervar(num_variables, autodiff_epsilon);
	ogderi_(&dervar, pervar.data());

	int finopt = 0;
	int finite = 0;
	ogexec_(valvar.data(), valcon.data(), &finopt, &finite, fitness, gradient);
	ogclos_();
	return std::make_tuple(valvar, valcon, finopt);
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