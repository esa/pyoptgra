#include <iostream>
#include <vector>
#include <tuple>

using std::cout;
using std::endl;
using std::vector;

extern"C" {
	void ogclos_();
	void ogderi_(int * dervar, double *pervar);
	void ogeval_(double * valvar, double * valcon, int * dervar, double * dercon,
	 	void (*)(double*, double*), void (*)(double*, double*, double*));
	void ogexec_(double * valvar, double * valcon, int * finopt, int * finite,
		void (*)(double*, double*, int*), void (*)(double*, double*, double*));
	void oginit_(int * varnum, int * connum);
}

void f(double* x, double *con, int *flag) {
	con[0] = 0;
	con[1] = 0;
	cout << "f called with ";
	for (int i = 0; i < 5; i++) {
		con[0] += (x[i]-i)*(x[i]-i);
		cout << x[i] << " ";
	}
	cout << " and flag " << *flag;
	cout << endl;
}

void g(double* x, double *con, double *der) {
	con[0] = 0;
	con[1] = 0;
	cout << "g called with ";
	for (int i = 0; i < 5; i++) {
		con[0] += 2*(x[i]-i);
		cout << x[i] << " ";
	}
	cout << endl;
}

template<class F, class G>
std::tuple<std::vector<double>, std::vector<double>, int> optimize(const std::vector<double> &initial_x,
 int num_constraints, F fitness, G gradient, bool has_gradient) {
	
	// initialization
	int num_variables = initial_x.size();
	std::vector<double> valvar(initial_x);
	std::vector<double> valcon(num_constraints+1);
	std::vector<double> dercon(num_variables * (num_constraints+1));
	oginit_(&num_variables, &num_constraints);

	int dervar = 1; // user-defined gradient
	if (!has_gradient) {
		dervar = 3; // numeric differentiation
	}

	vector<double> pervar(num_variables, 0);
	ogderi_(&dervar, pervar.data());

	int finopt = 0;
	int finite = 0;
	ogexec_(valvar.data(), valcon.data(), &finopt, &finite, fitness, gradient);
	ogclos_();
	return std::make_tuple(valvar, valcon, finopt);
}

int main(int argn, char** argc)
{
	std::vector<double> initial_x = {1,1,1,1,1};
	int dim = initial_x.size();
	std::vector<double> bestx, bestf;
	int finopt;
	std::tie(bestx, bestf, finopt) = optimize(initial_x, 1, f, g, false);

	cout << "Best x:";
	for (int i = 0; i < dim; i++) {
		cout << bestx[i] << " ";
	}
	cout << endl;

	cout << "Best f:";
	for (int i = 0; i < 1 + 1; i++) {
		cout << bestf[i] << " ";
	}
	cout << endl;

   return 0;
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