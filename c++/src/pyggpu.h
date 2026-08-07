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

class PyGGpu{
public:
    PyGGpu() = default;
    ~PyGGpu() = default;

    void launch(const std::string& kernel_name, const ir::IR& ir, py::tuple args, py::dict kwargs){
        // Check if a kernel is already compiled.
        compile(kernel_name, ir, args, kwargs);
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


    void compile(const std::string& kernel_name, const ir::IR& ir, py::tuple& args, py::dict& kwargs){
        std::string mangled_name = kernel_name;//getMangledName(kernel_name, args, kwargs);
        if(!fn_cache.contains(mangled_name)){
            std::cout << "Compiling kernel: " << mangled_name << std::endl;
            auto kernel = compiler.createKernel(kernel_name, ir, args, kwargs);
            fn_cache.insert_or_assign(mangled_name, std::move(kernel));
            std::cout << "Kernel compiled and cached: " << mangled_name << std::endl;
            fn_cache[mangled_name]->launch();
            return;
        } else {
            std::cout << "Kernel already compiled: " << mangled_name << std::endl;
            return;
        }

        // Parsing is done and we do not need the GIL anymore. 
        // Do not touch Python objects after this point.
        {
            py::gil_scoped_release release;
        }
    }

private:

    pyggpu::Compiler compiler;
    std::unordered_map<std::string, std::unique_ptr<pyggpu::BaseKernel>> fn_cache;

};