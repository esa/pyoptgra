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
		py::arg("derivatives_computation") = 1,	py::arg("autodiff_deltas"),	py::arg("variable_types"), py::arg("log_level") = 1);

    m.def("compute_sensitivity_matrices", &optgra::compute_sensitivity_matrices, "Calculate sensitivity matrices using optgra",
    	py::arg("x"), py::arg("constraint_types"), py::arg("fitness_callback"), py::arg("gradient_callback"),
    	py::arg("has_gradient"), py::arg("max_distance_per_iteration") = 10, py::arg("perturbation_for_snd_order_derivatives") = 1,
        py::arg("variable_scaling_factors"),
        py::arg("variable_names"), py::arg("constraint_names"),
        py::arg("derivatives_computation") = 1, py::arg("autodiff_deltas"), py::arg("variable_types"), py::arg("log_level") = 1);

    m.def("prepare_sensitivity_state", &optgra::prepare_sensitivity_state, "Prepare a state tuple needed for sensitivity analysis",
        py::arg("x"), py::arg("constraint_types"), py::arg("fitness_callback"), py::arg("gradient_callback"),
        py::arg("has_gradient"), py::arg("max_distance_per_iteration") = 10, py::arg("perturbation_for_snd_order_derivatives") = 1,
        py::arg("variable_scaling_factors"),
        py::arg("derivatives_computation") = 1, py::arg("autodiff_deltas"), py::arg("variable_types"), py::arg("log_level") = 1);

    m.def("get_sensitivity_matrices", &optgra::get_sensitivity_matrices, "Get sensitivity matrices from prepared sensitivity state",
        py::arg("variable_types"), py::arg("constraint_types"), py::arg("state_tuple"));

    m.def("sensitivity_update_new_callable", &optgra::sensitivity_update_new_callable, "Perform a single step given a new callable",
        py::arg("state_tuple"), py::arg("variable_types"),
        py::arg("constraint_types"), py::arg("fitness_callback"), py::arg("gradient_callback"), py::arg("has_gradient"),
        py::arg("max_distance_per_iteration") = 10, py::arg("perturbation_for_snd_order_derivatives") = 1,
        py::arg("variable_scaling_factors"),
        py::arg("derivatives_computation") = 1, py::arg("autodiff_deltas"), py::arg("log_level") = 1);

    m.def("sensitivity_update_constraint_delta", &optgra::sensitivity_update_constraint_delta,
        "Using a linear approximation, perform a single step with modified constraints", py::arg("state_tuple"), py::arg("variable_types"),
        py::arg("constraint_types"), py::arg("delta"),
        py::arg("max_distance_per_iteration") = 10, py::arg("perturbation_for_snd_order_derivatives") = 1,
        py::arg("variable_scaling_factors"),
        py::arg("log_level") = 1);

}