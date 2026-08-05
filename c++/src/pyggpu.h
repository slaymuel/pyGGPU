#include <iostream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../serialization/ir.pb.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

namespace py = pybind11;

struct KernelArg {
    enum Kind { Buffer, Scalar };
    Kind kind;
    std::string name;
    llvm::Type* llvmType;
    void* rawPtr;
    int64_t intValue;
    double floatValue;
};


class PyGGpu{
public:
    PyGGpu() = default;
    ~PyGGpu() = default;

    void launch(const std::string& kernelName, const ir::IR& ir, py::tuple args, py::dict kwargs){
        // Check if a kernel is already compiled.
        compile(kernelName, ir, args, kwargs);
    }

    void compile(const std::string& kernelName, const ir::IR& ir, py::tuple& args, py::dict& kwargs){
        llvm::LLVMContext ctx;        // create context here
        llvm::Module module("kernel", ctx);
        auto kernel_args = createKernalArguments(ir, args, kwargs, module.getContext());
        std::cout << "Kernel arguments created." << std::endl;
        auto *fn = createKernelFunction(kernelName, kernel_args, module);
        std::cout << "Kernel function created. Lowering to LLVM..." << std::endl;
        lowerToLLVM(ir, kernel_args, fn);

        //for (size_t i = 0; i < args.size(); ++i) {
        //    py::handle item = args[i];
        //    std::string name = "arg" + std::to_string(i);   // synthetic name
        //    std::cout << "Processing argument: " << name << std::endl;
        //    //process_argument(name, item);
        //}
        //
        //for (auto item : kwargs) {
        //    std::string name = py::cast<std::string>(item.first);
        //    py::handle value = item.second;
        //    std::cout << "Processing keyword argument: " << name << std::endl;
        //    //process_argument(name, value);
        //}

        // Parsing is done and we do not need the GIL anymore. 
        // Do not touch Python objects after this point.
        {
            py::gil_scoped_release release;
        }
    }

    llvm::Value* getBufferPointer(const std::string& name, llvm::Function* fn) {
        for (auto& arg : fn->args())
            if (arg.getName() == name)
                return &arg;
        throw std::runtime_error("Unknown buffer: " + name);
    }

    void* extractPointerFromPython(py::tuple args, py::dict kwargs, const std::string& name) {
        // kwargs
        if (kwargs.contains(name)) {
            py::array arr = kwargs[py::str(name)].cast<py::array>();
            
            py::buffer_info info = arr.request();
            return info.ptr;
        }

        // args
        for (auto item : args) {
            py::object obj = py::reinterpret_borrow<py::object>(item);
            
            if (py::hasattr(obj, "name") && obj.attr("name").cast<std::string>() == name) {
                py::array arr = obj.attr("value").cast<py::array>();
                // py::array arr = obj.cast<py::array>();
                py::buffer_info info = arr.request();
                return info.ptr;
            }
        }

        throw std::runtime_error("Argument not found: " + name);
    }


    std::unordered_map<std::string, KernelArg> createKernalArguments(
        const ir::IR& ir, 
        const py::tuple& args, 
        const py::dict& kwargs,
        llvm::LLVMContext& ctx
    ) {
        std::unordered_map<std::string, KernelArg> kernelArgs;

        for (const auto& binding : ir.env().bindings()) {
            KernelArg arg;
            arg.name = binding.name();

            if (binding.value().has_buffer()) {
                arg.kind = KernelArg::Buffer;

                auto& buf = binding.value().buffer();
                std::string dtype = buf.dtype();

                llvm::Type* elemType = nullptr;
                if (dtype == "float32") elemType = llvm::Type::getFloatTy(ctx);
                else if (dtype == "float64") elemType = llvm::Type::getDoubleTy(ctx);
                else if (dtype == "int32") elemType = llvm::Type::getInt32Ty(ctx);
                else if (dtype == "int64") elemType = llvm::Type::getInt64Ty(ctx);
                else throw std::runtime_error("Unsupported dtype: " + dtype);

                arg.llvmType = llvm::PointerType::getUnqual(elemType);
                arg.rawPtr = extractPointerFromPython(args, kwargs, arg.name);
            }
            else if (binding.value().has_int_const()) {
                arg.kind = KernelArg::Scalar;
                arg.llvmType = llvm::Type::getInt64Ty(ctx);
                arg.intValue = binding.value().int_const();
            }
            else if (binding.value().has_float_const()) {
                arg.kind = KernelArg::Scalar;
                arg.llvmType = llvm::Type::getDoubleTy(ctx);
                arg.floatValue = binding.value().float_const();
            }

            kernelArgs[arg.name] = arg;
        }

        return kernelArgs;
    }

    void lowerToLLVM(
        const ir::IR &ir, 
        const std::unordered_map<std::string, KernelArg>& kernel_args, 
        llvm::Function* fn
    ) {
        llvm::IRBuilder<> builder(&fn->getEntryBlock());
        std::cout << "Lowering IR to LLVM..." << std::endl;
        std::unordered_map<std::string, llvm::Value*> ssa;

        for (const auto &op : ir.ops()) {
            std::cout << "Lowering operation: " << op.DebugString() << std::endl;
            switch (op.kind_case()) {
                case ir::Operation::kAdd: {
                    auto lhs = getValue(op.add().lhs(), ssa, builder);
                    auto rhs = getValue(op.add().rhs(), ssa, builder);
                    auto res = builder.CreateAdd(lhs, rhs, op.add().result());
                    ssa[op.add().result()] = res;
                    break;
                }
                case ir::Operation::kLoad: {
                    auto buf = getBufferPointer(op.load().buffer(), fn);
                    auto idx = getValue(op.load().index(), ssa, builder);
                    // Get the pointer to the element at the given index
                    auto elemTy = kernel_args.at(op.load().buffer()).llvmType;
                    auto ptr = builder.CreateGEP(elemTy, buf, idx);
                    auto val = builder.CreateLoad(elemTy, ptr);
                    ssa[op.load().result()] = val;
                    break;
                }
                case ir::Operation::kStore: {
                    auto buf = getBufferPointer(op.store().buffer(), fn);
                    auto idx = getValue(op.store().index(), ssa, builder);
                    auto elemTy = kernel_args.at(op.store().buffer()).llvmType;
                    auto ptr = builder.CreateGEP(elemTy, buf, idx);
                    auto val = getValue(op.store().value(), ssa, builder);
                    builder.CreateStore(val, ptr);
                    break;
                }
                case ir::Operation::kConstant: {
                    auto val = getValue(op.constant().value(), ssa, builder);
                    ssa[op.constant().result()] = val;
                    break;
                }
                case ir::Operation::kRet: {
                    auto retVal = getValue(op.ret().value(), ssa, builder);
                    builder.CreateRet(retVal);
                    break;
                }
                default:
                    break;
            }
        }

        builder.CreateRetVoid();
    }

private:
    llvm::Function* createKernelFunction(
        const std::string& kernelName,
        const std::unordered_map<std::string, KernelArg>& kernel_args,
        llvm::Module& module
    ) {
        std::vector<KernelArg> kernelArgs;
        kernelArgs.reserve(kernel_args.size());
        for (const auto& kv : kernel_args)
            kernelArgs.push_back(kv.second);

        std::vector<llvm::Type*> argTypes;
        for (auto& arg : kernelArgs)
            argTypes.push_back(arg.llvmType);

        auto *fnType = llvm::FunctionType::get(
            llvm::Type::getVoidTy(module.getContext()),
            argTypes,
            false
        );

        llvm::Function* fn = llvm::Function::Create(
            fnType,
            llvm::Function::ExternalLinkage,
            kernelName,
            module
        );

        llvm::BasicBlock::Create(module.getContext(), "entry", fn);

        int idx = 0;
        for (auto& arg : fn->args())
            arg.setName(kernelArgs[idx++].name);

        return fn;
    }

    llvm::Value* getValue(
        const ir::Operand &operand,
        const std::unordered_map<std::string, llvm::Value*> &ssa,
        llvm::IRBuilder<> &builder
    ) {
        switch (operand.kind_case()) {
            case ir::Operand::kSsa: {
                auto it = ssa.find(operand.ssa());
                if (it == ssa.end()) {
                    throw std::runtime_error("Unknown SSA value: " + operand.ssa());
                }
                return it->second;
            }
            case ir::Operand::kIntConst:
                return builder.getInt64(operand.int_const());
            case ir::Operand::kFloatConst:
                return llvm::ConstantFP::get(builder.getDoubleTy(), operand.float_const());
            case ir::Operand::kBoolConst:
                return builder.getInt1(operand.bool_const());
            default:
                throw std::runtime_error("Unsupported operand kind");
        }
    }

};