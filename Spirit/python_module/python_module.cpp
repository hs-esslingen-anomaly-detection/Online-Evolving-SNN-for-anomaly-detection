#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "../spirit.h"

namespace py = pybind11;

//TODO: Add documentation for attributes & functions of py module
PYBIND11_MODULE(spirit_module, m) {
    py::class_<spirit>(m, "spirit")
            .def(py::init<unsigned int, unsigned int, int, double, double, double, bool>(),
                    py::arg("dimensionality"), py::arg("window_size"), py::arg("initial_eigencomponents") = 1,
                    py::arg("exp_forgetting_factor") = 0.96, py::arg("lower_energy_threshold") = 0.95, 
                    py::arg("upper_energy_threshold") = 0.98, py::arg("debug") = false)
            .def("update", (double (spirit::*)(const std::vector<double>&)) &spirit::update, py::arg("x")) //Python equivalent: list
            .def("update", (double (spirit::*)(const Eigen::VectorXd&)) &spirit::update, py::arg("x")) //Python equivalent: numpy array
            .def("run", (std::vector<double>(spirit::*)(const std::vector<Eigen::VectorXd>&))&spirit::run, py::arg("series"))
            .def("run", (std::vector<double>(spirit::*)(const std::vector<std::vector<double>>&))&spirit::run, py::arg("series"))
            .def("get_reconstruction", &spirit::get_reconstruction);
}