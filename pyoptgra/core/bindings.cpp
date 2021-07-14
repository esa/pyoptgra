#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    m.doc() = "A python wrapper including Optgra";

    m.def("optimize", &optgra::optimize, "Optimize using optgra",
    	py::arg("initial_x"), py::arg("constraint_types"), py::arg("fitness_callback"), py::arg("gradient_callback"),
    	py::arg("has_gradient"), py::arg("max_iterations") = 10, py::arg("max_correction_iterations") = 10,
		py::arg("max_distance_per_iteration") = 10, py::arg("perturbation_for_snd_order_derivatives") = 1,
		py::arg("convergence_thresholds"), py::arg("variable_scaling_factors"), py::arg("constraint_priorities"),
		py::arg("variable_names"), py::arg("constraint_names"),	py::arg("optimization_method") = 2,
		py::arg("derivatives_computation") = 1,	py::arg("autodiff_deltas"),	py::arg("log_level") = 1);

    m.def("sensitivity", &optgra::sensitivity, "Calculate sensitivity matrices using optgra",
    	py::arg("x"), py::arg("con"), py::arg("constraint_types"), py::arg("fitness_callback"), py::arg("gradient_callback"),
    	py::arg("has_gradient"));
}
