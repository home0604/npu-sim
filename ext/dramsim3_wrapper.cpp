#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <string>
#include <functional>
#include "dramsim3.h"

namespace py = pybind11;

PYBIND11_MODULE(dramsim3_py, m) {
    m.doc() = "DRAMSim3 Python bindings via pybind11";

    py::class_<dramsim3::MemorySystem>(m, "MemorySystem")
        .def(py::init([](const std::string& config_file,
                         const std::string& output_dir,
                         std::function<void(uint64_t)> read_cb,
                         std::function<void(uint64_t)> write_cb) {
            auto* mem = new dramsim3::MemorySystem(config_file, output_dir, read_cb, write_cb);
            return mem;
        }))
        .def("WillAcceptTransaction", &dramsim3::MemorySystem::WillAcceptTransaction,
             py::arg("addr"), py::arg("is_write"))
        .def("AddTransaction", &dramsim3::MemorySystem::AddTransaction,
             py::arg("addr"), py::arg("is_write"))
        .def("ClockTick", &dramsim3::MemorySystem::ClockTick)
        .def("PrintStats", &dramsim3::MemorySystem::PrintStats)
        .def("ResetStats", &dramsim3::MemorySystem::ResetStats);
}
