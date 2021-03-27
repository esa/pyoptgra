#include <iostream>
#include <vector>
#include <tuple>
#include <pagmo/problems/ackley.hpp>

#include "wrapper.hpp"

using std::cout;
using std::endl;
using std::vector;

using namespace optgra;

void f(double* x, double *con, int *flag) {
	con[0] = 0;
	con[1] = 0;
	cout << "f called with ";
	for (int i = 0; i < 5; i++) {
		con[0] += (x[i]-i)*(x[i]-i);
		con[1] += x[i];
		cout << x[i] << " ";
	}
	cout << " and flag " << *flag;
	cout << endl;
}

void g(double* x, double *con, double *der) {
	cout << "g called with ";

	for (int i = 0; i < 5; i++) {
		con[0] += (x[i]-i)*(x[i]-i);
		con[1] += x[i];
	}
	for (int i = 0; i < 5; i++) {
		der[i] = 2*(x[i]-i);
		cout << x[i] << " ";
	}
	//std::vector<double> dercon(num_variables * (num_constraints+1));
	cout << endl;
}

int main(int argn, char** argc)
{
	std::vector<double> initial_x = {1,1,1,1,1};
	int dim = initial_x.size();
	std::vector<double> bestx, bestf;
	int finopt;
	std::tie(bestx, bestf, finopt) = optimize(initial_x, {0,-1}, f, g, false);

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

	cout << "Testing pagmo problem" << endl;

	pagmo::problem test_prob(pagmo::ackley{10});
	initial_x = std::vector<double>(10,1);

	std::tie(bestx, bestf) = optimize(test_prob, initial_x);

	cout << "Best x:";
	for (int i = 0; i < bestx.size(); i++) {
		cout << bestx[i] << " ";
	}
	cout << endl;

	cout << "Best f:";
	for (int i = 0; i < bestf.size(); i++) {
		cout << bestf[i] << " ";
	}
	cout << endl;

   return 0;
}