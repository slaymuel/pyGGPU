#include "pybind11/pybind11.h"
#include "pyggpu.h"
#include "kernel_registry.h"

namespace py = pybind11;

PYBIND11_MODULE(pyggpu, m) {
    py::class_<PyGGpu>(m, "PyGGpu")
        .def(py::init<>())
        .def("launch", [](PyGGpu &self,
                        const std::string &kernelName,
                        KernelTarget target,
                        py::object ir,
                        py::tuple args,
                        py::dict kwargs) {

            // Deserialize the IR from Python to C++
            std::string serialized = py::bytes(ir.attr("SerializeToString")());
            ir::IR cpp_ir;
            cpp_ir.ParseFromString(serialized);

            self.launch(kernelName, target, cpp_ir, args, kwargs);
        })
        .def_static("dumpSignatures", &pyggpu::kernel_registry::dumpSignatures);

    // Also expose the kernel target enum to Python
    py::enum_<KernelTarget>(m, "KernelTarget")
        .value("CPU", KernelTarget::CPU)
        .value("PTX", KernelTarget::PTX)
        .export_values();
}