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

namespace pyggpu {

class KernelEngine {
public:
    KernelEngine() = default;
    ~KernelEngine() = default;

    /*
    
    */
    void launchKernel(const std::string& kernel_name, KernelTarget target, const ir::IR& ir, py::tuple args, py::dict kwargs) {
        kernel_cache[kernel_name]->launchFromTuple(args);
    }

    void compile(const std::string& kernel_name, KernelTarget target, const ir::IR& ir, py::tuple& args, py::dict& kwargs){
        std::string mangled_name = kernel_name;//getMangledName(kernel_name, args, kwargs);
        if(!kernel_cache.contains(mangled_name)){
            switch(target){
                case KernelTarget::CPU:
                    compileCPU(kernel_name, ir, args, kwargs);
                    break;
                case KernelTarget::PTX:
                    compilePTX(kernel_name, ir, args, kwargs);
                    break;
                default:
                    throw std::runtime_error("Unsupported kernel target");
            }
        } else {
            throw std::runtime_error("Kernel already compiled: " + mangled_name);
        }

        // Parsing is done and we do not need the GIL anymore. 
        // Do not touch Python objects after this point.
        {
            py::gil_scoped_release release;
        }
    }

    void compileCPU(const std::string& kernel_name, const ir::IR& ir, py::tuple& args, py::dict& kwargs){
        std::cout << "Compiling kernel: " << kernel_name << std::endl;
        //auto kernel = compiler.compile(kernel_name, ir, args, kwargs);
        auto compiled = compiler.compile(kernel_name, KernelTarget::CPU, ir, args, kwargs);
        auto handle = std::get<void*>(compiled.handle);
        auto kernel = kernel_registry::createKernel(kernel_name, compiled.sig_id, handle);

        kernel_cache.insert_or_assign(kernel_name, std::move(kernel));
        std::cout << "Kernel compiled and cached: " << kernel_name << std::endl;
        kernel_cache[kernel_name]->launchFromTuple(args);
    }
    void compilePTX(const std::string& kernel_name, const ir::IR& ir, py::tuple& args, py::dict& kwargs){
        std::cout << "Compiling PTX kernel: " << kernel_name << std::endl;
        auto compiled = compiler.compile(kernel_name, KernelTarget::PTX, ir, args, kwargs);
        auto handle = std::get<CUfunction>(compiled.handle);
        auto kernel = kernel_registry::createKernel(kernel_name, compiled.sig_id, handle);

        kernel_cache.insert_or_assign(kernel_name, std::move(kernel));
        std::cout << "PTX Kernel compiled and cached: " << kernel_name << std::endl;
        kernel_cache[kernel_name]->launchFromTuple(args);
    }

    bool isKernelCompiled(const std::string& kernel_name){
        return kernel_cache.contains(kernel_name);
    }

private:

    pyggpu::Compiler compiler;
    std::unordered_map<std::string, std::unique_ptr<pyggpu::BaseKernel>> kernel_cache;
};

} // namespace pyggpu