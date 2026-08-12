#pragma once

#include <string>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../serialization/ir.pb.h"
#include "signature.h"
#include "type_mapping.h"

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

namespace pyggpu {

struct KernelArg {
    enum Kind { Buffer, Scalar };
    IRBaseType ir_type;
    Kind kind;
    std::string name;
    llvm::Type* llvmType;
    llvm::Type* elementType;
    void* rawPtr;
    int64_t intValue;
    double floatValue;
};

struct CompiledFn {
    void* fn_ptr = nullptr;  
    SignatureID sig_id = 0;  
};

class Compiler{
public:
    Compiler();
    CompiledFn compile(std::string kernel_name, const ir::IR& ir, const py::tuple& args, const py::dict& kwargs);

private:
    void lowerToLLVM(
        const ir::IR &ir, 
        const std::unordered_map<std::string, KernelArg>& kernel_args, 
        llvm::Function* fn
    );

    llvm::Value* getBufferPointer(const std::string& name, llvm::Function* fn);

    void* extractPointerFromPython(py::tuple args, py::dict kwargs, const std::string& name);

    std::unordered_map<std::string, KernelArg> createKernelArguments(
        const ir::IR& ir, 
        const py::tuple& args, 
        const py::dict& kwargs,
        llvm::LLVMContext& ctx
    );

    llvm::Function* createKernelFunction(
        const std::string& kernel_name,
        const std::unordered_map<std::string, KernelArg>& kernel_args,
        llvm::Module& module
    );

    llvm::Value* getValue(
        const ir::Operand &operand,
        const std::unordered_map<std::string, llvm::Value*> &ssa,
        llvm::IRBuilder<> &builder
    );

    llvm::Value* castValueToType(llvm::Value* value, llvm::Type* targetType, llvm::IRBuilder<>& builder);

    std::unique_ptr<llvm::orc::LLJIT> jit;
};

} // namespace pyggpu