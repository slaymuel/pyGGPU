#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../serialization/ir.pb.h"
#include "kernel.h"
#include "kernel_engine.h"
#include "kernel_registry.h"
#include "compiler.h"
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>

namespace py = pybind11;
using namespace pyggpu;

class PyGGpu{
public:
    PyGGpu(){

    }

    ~PyGGpu() = default;

    void launch(const std::string& kernel_name, KernelTarget target, const ir::IR& ir, py::tuple args, py::dict kwargs){
        // std::string mangled_name = getMangledName(kernel_name, args, kwargs);
        // Check if a kernel is already compiled.
        if(!kernel_engine.isKernelCompiled(kernel_name))
            compile(kernel_name, target, ir, args, kwargs);

        kernel_engine.launchKernel(kernel_name, target, ir, args, kwargs);
        //compile(kernel_name, target, ir, args, kwargs);
    }

    void getKernel();

    std::string getMangledName(
        const std::string& kernel_name,
        const py::tuple& args,
        const py::dict& kwargs)
    {
        std::stringstream ss;
        ss << kernel_name << "__";

        // Positional args
        for (auto item : args) {
            py::object obj = py::reinterpret_borrow<py::object>(item);
            ss << getArgType(obj) << "_";
        }

        // Keyword args (sorted for determinism)
        std::vector<std::string> keys;
        for (auto kv : kwargs) keys.push_back(kv.first.cast<std::string>());
        std::sort(keys.begin(), keys.end());

        for (auto& key : keys) {
            py::object obj = kwargs[key.c_str()];
            ss << getArgType(obj) << "_";
        }

        return ss.str();
    }

    std::string getArgType(const py::object& obj) {
        py::array arr = obj.attr("array").cast<py::array>();
        char dtype = arr.dtype().kind();
        int itemsize = arr.itemsize();

        if (dtype == 'f' && itemsize == 4) return "f32";
        if (dtype == 'f' && itemsize == 8) return "f64";
        if (dtype == 'i' && itemsize == 4) return "i32";
        if (dtype == 'i' && itemsize == 8) return "i64";

        throw std::runtime_error("Unsupported dtype");
    }


    void compile(const std::string& kernel_name, KernelTarget target, const ir::IR& ir, py::tuple& args, py::dict& kwargs){
        kernel_engine.compile(kernel_name, target, ir, args, kwargs);
    }

private:

    pyggpu::Compiler compiler;
    std::unordered_map<std::string, std::unique_ptr<pyggpu::BaseKernel>> kernel_cache;
    KernelEngine kernel_engine;
};