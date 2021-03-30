#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyoptgra, m) {
    m.doc() = "A python wrapper including Optgra"; // optional module docstring


    m.def("optimize", &optgra::optimize, "Optimize using optgra",
    	py::arg("initial_x"), py::arg("constraint_types"), py::arg("fitness_callback"), py::arg("gradient_callback"),
    	py::arg("has_gradient"), py::arg("params"));
}

