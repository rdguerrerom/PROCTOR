#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "simulation.hpp"

namespace py = pybind11;
using namespace PROCTOR;

PYBIND11_PLUGIN(PROCTOR) {
  py::module mP("PROCTOR", R"doc(
        Python module
        -----------------------
        .. currentmodule:: PROCTOR
        .. autosummary::
           :toctree: _generate

           Simulation0D
           Simulation1D
    )doc");

  py::class_<Simulation1D>(mP, "Simulation1D")
      .def(py::init<>())
      .def(py::init<const Eigen::VectorXd & /*x*/, double /*dx*/, double /*dt*/,
                    int /*n_grid_points*/, int /*n_singlets*/,
                    int /*n_doublets*/, int /*n_triplets*/>())
      // Expossing the setters
      .def("set_initial_wave_packet", &Simulation1D::set_initial_wave_packet )
      .def("set_PES", &Simulation1D::set_PES )
      .def("set_DM", &Simulation1D::set_DM )
      .def("set_NACME", &Simulation1D::set_NACME )
      .def("set_TDM", &Simulation1D::set_TDM )
      .def("set_SOC", &Simulation1D::set_SOC )
      // Exposing tthe getters
      .def("electronic_PES_complete", &Simulation1D::electronic_PES_complete )
      .def("electronic_DM_complete", &Simulation1D::electronic_DM_complete )
      .def("electronic_TDM_complete", &Simulation1D::electronic_TDM_complete )
      .def("electronic_NACME_complete", &Simulation1D::electronic_NACME_complete );

  return mP.ptr();
}
