//
// Created by tobias on 27.03.20.
//

//
// Created by tobias on 27.03.20.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "../grf.h"

namespace py = pybind11;

PYBIND11_MODULE(grf_module, m)  {

    m.doc() = "Python module for the C++ implementation for a multidimensional gaussian receptive field for streaming "
              "data based on the algorithm proposed by Panuku et. al. "
              "(https://link.springer.com/chapter/10.1007/978-3-540-69158-7_9)";

    py::class_<grf>(m, "grf")
            .def(py::init<int, int, double, int, bool>(), pybind11::arg("n_neurons"), pybind11::arg("dim"), pybind11::arg("alpha"),
                                                          pybind11::arg("avg_window"), pybind11::arg("debug"))
            .def("update", (std::vector<double> (grf::*)(const std::vector<double>&))&grf::update, pybind11::arg("x"))
            .def("update", (std::vector<double> (grf::*)(const Eigen::VectorXd&))&grf::update, pybind11::arg("x"))
            .def("get_clusters",&grf::get_clusters)
            .def("get_covariance_matrices", &grf::get_covariance_matrices);

    py::class_<kmeans>(m, "kmeans")
            .def(py::init<int, double>())
            .def("update", (std::vector<kmeans::cluster> (kmeans::*)(const std::vector<double>&))&kmeans::update)
            .def("update", (std::vector<kmeans::cluster> (kmeans::*)(const Eigen::VectorXd &))&kmeans::update)
            .def("get_clusters", &kmeans::get_clusters);

    py::class_<kmeans::cluster>(m, "cluster")
            .def(py::init<const Eigen::VectorXd&, double>())
            .def("update", &kmeans::cluster::update)
            .def("get_center", &kmeans::cluster::get_center)
            .def("get_number_of_items", &kmeans::cluster::get_number_of_items)
            .def("get_elements", &kmeans::cluster::get_elements)
            .def("get_debug_idxs", &kmeans::cluster::get_debug_idxs);
}