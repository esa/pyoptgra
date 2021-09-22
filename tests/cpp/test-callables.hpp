#include <vector>

using std::cout;
using std::endl;
using std::vector;

///Fitness callable implementing a constraint function of \sum_0^4 (x_i - i)^2 and a merit function of \sum_0^4 x_i
std::vector<double> f(std::vector<double> x) {
	std::vector<double> con(2);
	con[0] = 0;
	con[1] = 0;
	//cout << "f called with ";
	for (int i = 0; i < 5; i++) {
		con[0] += (x[i]-i)*(x[i]-i);
		con[1] += x[i];
		//cout << x[i] << " ";
	}
	//cout << endl;
	return con;
}

///Gradient callable implementing a constraint function of \sum_0^4 (x_i - i)^2 and a merit function of \sum_0^4 x_i 
std::vector<std::vector<double>> g(std::vector<double> x) {
	//cout << "g called with ";
	int x_dim = x.size();
	int num_constraints = 2;
	std::vector<std::vector<double>> der(2);
	der[0].resize(x_dim);
	der[1].resize(x_dim);

	for (int i = 0; i < x_dim; i++) {
		der[0][i] = 2*(x[i]-i);
		der[1][i] = 1;
		//cout << x[i] << " ";
	}
	
	//cout << endl;
	return der;
}

std::vector<double> f_simple(std::vector<double> x) {
	std::vector<double> con(2);
	con[0] = 10 - x[0];
	con[1] = 2*x[0];
	//cout << "f_simple called with " << x[0];
	//cout << endl;
	return con;
}


std::vector<std::vector<double>> g_simple(std::vector<double> x) {
	//cout << "g_simple called with " << x[0] << endl;
	std::vector<std::vector<double>> der(2);
	der[0].resize(1);
	der[1].resize(1);

	der[0][0] = -1;
	der[1][0] = 2;
	return der;
}
