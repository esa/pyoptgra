#include <iostream>
#include <vector>
#include <tuple>

#include "wrapper.hpp"

using std::cout;
using std::endl;
using std::vector;

using namespace optgra;

///Fitness callable implementing a constraint function of \sum_0^4 (x_i - i)^2 and a merit function of \sum_0^4 x_i
std::vector<double> f(std::vector<double> x) {
	std::vector<double> con(2);
	con[0] = 0;
	con[1] = 0;
	cout << "f called with ";
	for (int i = 0; i < 5; i++) {
		con[0] += (x[i]-i)*(x[i]-i);
		con[1] += x[i];
		cout << x[i] << " ";
	}
	cout << endl;
	return con;
}

///Fitness callable implementing a constraint function of \sum_0^4 (x_i - i)^2 and a merit function of \sum_0^4 x_i 
std::vector<std::vector<double>> g(std::vector<double> x) {
	cout << "g called with ";
	int x_dim = x.size();
	int num_constraints = 2;
	std::vector<std::vector<double>> der(2);
	der[0].resize(x_dim);
	der[1].resize(x_dim);

	for (int i = 0; i < x_dim; i++) {
		der[0][i] = 2*(x[i]-i);
		der[1][i] = 1;
		cout << x[i] << " ";
	}
	
	cout << endl;
	return der;
}

int main(int argn, char** argc)
{
	std::vector<double> initial_x = {1,1,1,1,1};
	int dim = initial_x.size();
	std::vector<double> bestx, bestf;
	int finopt;
	
	std::tie(bestx, bestf, finopt) = optimize(initial_x, {0,-1}, f, g, true);

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

	cout << "f(best_x):";
	auto best_orig = f(bestx);
	for (int i = 0; i < 1 + 1; i++) {
		cout << best_orig[i] << " ";
	}
	cout << endl;

	std::vector<double> sens_x = {0.000183705, 1, 1.99982, 2.99963, 3.99945};
	std::vector<double> sens_f = {5.06211e-07, 9.99908};

	//std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
    // std::vector<std::vector<double>>, std::vector<std::vector<double>>>

	std::ignore = compute_sensitivity_matrices(bestx, {0,-1}, f, g, true);

	std::ignore =  prepare_sensitivity_state(bestx, {0,-1}, f, g, true);

   return 0;
}