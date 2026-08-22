#include "compiler.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <ranges>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../serialization/ir.pb.h"
#include "type_mapping.h"


#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/MC/TargetRegistry.h>   
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h> 

namespace py = pybind11;

namespace {
} // namespace

namespace pyggpu {

Compiler::Compiler(){
    // Initialization should only run once, so wrap in static lambda.
    static const bool llvmTargetReady = []() {
        if (llvm::InitializeNativeTarget()) {
            throw std::runtime_error("Failed to initialize native LLVM target");
        }
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        return true;
    }();
    // Avoid compiler warnings.
    (void)llvmTargetReady;

    // Create the JIT compiler
    auto jit = llvm::orc::LLJITBuilder().create();
    if (!jit) {
        llvm::errs() << "Failed to create LLJIT: ";
        llvm::logAllUnhandledErrors(jit.takeError(), llvm::errs(), "");
        exit(1);
    }
    this->jit = std::move(*jit);

    // Initialize CUDA
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext cu_context;
    cuCtxCreate(&cu_context, nullptr, 0, device);
}

// Destructor
Compiler::~Compiler() {
    // Clean up the JIT compiler
}

CompiledFn Compiler::compile(std::string kernel_name, KernelTarget target, const ir::IR& ir, const py::tuple& args, const py::dict& kwargs) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("kernel", *ctx);

    // Setup the kernel arguments. These should probably already be sorted here.
    auto kernel_args = createKernelArguments(ir, args, kwargs, module->getContext());
    std::cout << "Kernel arguments created." << std::endl;
    auto *fn = createKernelFunction(kernel_name, kernel_args, *module);
    std::cout << "Kernel function created. Lowering to LLVM..." << std::endl;

    // Lower the IR to LLVM IR
    lowerToLLVM(ir, kernel_args, fn);
    std::cout << "Lowering complete." << std::endl;

    // Register the module with the JIT (lazy compilation)
    std::cout << "Adding module to JIT..." << std::endl;
    llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(ctx));
    auto err = this->jit->addIRModule(std::move(tsm));
    if (err) {
        llvm::errs() << "Failed to add module\n";
        exit(1);
    }

    // Look up the generated kernel function by its actual name.
    std::cout << "Fetching kernel symbol: " << kernel_name << std::endl;
    auto sym = this->jit->lookup(kernel_name);
    if (!sym) {
        llvm::errs() << "Failed to find symbol '" << kernel_name << "'\n";
        exit(1);
    }
    void* kernel_ptr = sym->toPtr<void*>();

    // Sort kernel arguments by name for consistency (so that we get the same signature ID for the same kernel)
    std::vector<KernelArg> sorted_kernel_args;
    sorted_kernel_args.reserve(kernel_args.size());
    for (const auto& kv : kernel_args)
        sorted_kernel_args.push_back(kv.second);
    std::sort(sorted_kernel_args.begin(), sorted_kernel_args.end(), [](const KernelArg& a, const KernelArg& b) {
        return a.name < b.name;
    });

    // Extract the argument types for the signature ID.
    // This is way too ugly and needs to change. We already have the kernel args.....
    auto arg_type_view = sorted_kernel_args | std::views::transform([](const KernelArg& arg) {
        return arg.ir_type;
    });
    std::vector<IRBaseType> arg_types;
    arg_types.reserve(sorted_kernel_args.size());
    std::ranges::copy(arg_type_view, std::back_inserter(arg_types));

    // Get the signature ID from the registry
    auto sig_id = computeIdFromIR(arg_types);
    return CompiledFn{kernel_ptr, sig_id};
}

void Compiler::compilePTX(std::string kernel_name, llvm::Module& module, llvm::Function* fn) {
    // Set target triple & data layout on your LLVM module
    std::string triple = "nvptx64-nvidia-cuda";
    module.setTargetTriple(triple);

    std::string err_str;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, err_str);
    std::unique_ptr<llvm::TargetMachine> tm(
        target->createTargetMachine(triple, "sm_80", "", {}, std::nullopt)
    );
    module.setDataLayout(tm->createDataLayout());

    // Emit PTX code to a string
    std::string ptx_output;
    llvm::raw_string_ostream ss(ptx_output);
    llvm::buffer_ostream write_stream(ss);
    llvm::legacy::PassManager pm;
    tm->addPassesToEmitFile(pm, write_stream, nullptr, llvm::CodeGenFileType::AssemblyFile);
    pm.run(module);

    CUmodule cu_module;
    // JIT compile the PTX to SASS
    CUresult res = cuModuleLoadData(&cu_module, ptx_output.c_str());
    if (res != CUDA_SUCCESS) {
        // In a real pybind11 app, throw a std::runtime_error here 
        // so Python catches it as an Exception!
        throw std::runtime_error("Failed to load PTX module! Error code: " + std::to_string(res));
    }

    CUfunction kernel;
    cuModuleGetFunction(&kernel, cu_module, kernel_name.c_str());
}

void Compiler::lowerToLLVM(
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
                if (lhs->getType() != rhs->getType()) {
                    throw std::runtime_error("Type mismatch in add operands");
                }
                llvm::Value* res = nullptr;
                if (lhs->getType()->isFloatingPointTy()) {
                    res = builder.CreateFAdd(lhs, rhs, op.add().result());
                } else {
                    res = builder.CreateAdd(lhs, rhs, op.add().result());
                }
                ssa[op.add().result()] = res;
                break;
            }
            case ir::Operation::kLoad: {
                auto buf = getBufferPointer(op.load().buffer(), fn);
                auto idx = getValue(op.load().index(), ssa, builder);
                // Get the pointer to the element at the given index
                auto elemTy = kernel_args.at(op.load().buffer()).elementType;
                auto ptr = builder.CreateGEP(elemTy, buf, idx);
                auto val = builder.CreateLoad(elemTy, ptr);
                ssa[op.load().result()] = val;
                break;
            }
            case ir::Operation::kStore: {
                auto buf = getBufferPointer(op.store().buffer(), fn);
                auto idx = getValue(op.store().index(), ssa, builder);
                auto elemTy = kernel_args.at(op.store().buffer()).elementType;
                auto ptr = builder.CreateGEP(elemTy, buf, idx);
                auto val = getValue(op.store().value(), ssa, builder);
                val = castValueToType(val, elemTy, builder);
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

llvm::Value* Compiler::getBufferPointer(const std::string& name, llvm::Function* fn) {
    for (auto& arg : fn->args())
        if (arg.getName() == name)
            return &arg;
    throw std::runtime_error("Unknown buffer: " + name);
}

void* Compiler::extractPointerFromPython(py::tuple args, py::dict kwargs, const std::string& name) {
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
            py::buffer_info info = arr.request();
            return info.ptr;
        }
    }

    throw std::runtime_error("Argument not found: " + name);
}

std::unordered_map<std::string, KernelArg> Compiler::createKernelArguments(
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
            IRBaseType ir_type;
            if (dtype == "float32") {
                elemType = llvm::Type::getFloatTy(ctx);
                ir_type = IRBaseType::F32Ptr;
            } else if (dtype == "float64"){
                elemType = llvm::Type::getDoubleTy(ctx);
                ir_type = IRBaseType::F64Ptr;
            } else if (dtype == "int32") {
                elemType = llvm::Type::getInt32Ty(ctx);
                ir_type = IRBaseType::I32Ptr;
            } else if (dtype == "int64") {
                elemType = llvm::Type::getInt64Ty(ctx);
                ir_type = IRBaseType::I64Ptr;
            } else throw std::runtime_error("Unsupported dtype: " + dtype);
            arg.ir_type = ir_type;
            arg.elementType = elemType;
            arg.llvmType = llvm::PointerType::getUnqual(elemType);
            arg.rawPtr = extractPointerFromPython(args, kwargs, arg.name);
        }
        else if (binding.value().has_int_const()) {
            arg.kind = KernelArg::Scalar;
            arg.ir_type = IRBaseType::I64;
            arg.elementType = nullptr;
            arg.llvmType = llvm::Type::getInt64Ty(ctx);
            arg.intValue = binding.value().int_const();
        }
        else if (binding.value().has_float_const()) {
            arg.kind = KernelArg::Scalar;
            arg.ir_type = IRBaseType::F64;
            arg.elementType = nullptr;
            arg.llvmType = llvm::Type::getDoubleTy(ctx);
            arg.floatValue = binding.value().float_const();
        }

        kernelArgs[arg.name] = arg;
    }

    return kernelArgs;
}

llvm::Function* Compiler::createKernelFunction(
    const std::string& kernel_name,
    const std::unordered_map<std::string, KernelArg>& kernel_args,
    llvm::Module& module
) {
    std::vector<KernelArg> kernel_arg_vector;
    kernel_arg_vector.reserve(kernel_args.size());
    for (const auto& kv : kernel_args)
        kernel_arg_vector.push_back(kv.second);
    std::sort(kernel_arg_vector.begin(), kernel_arg_vector.end(), [](const KernelArg& a, const KernelArg& b) {
        return a.name < b.name;
    });

    std::vector<llvm::Type*> argTypes;
    for (auto& arg : kernel_arg_vector)
        argTypes.push_back(arg.llvmType);

    auto *fnType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(module.getContext()),
        argTypes,
        false
    );

    llvm::Function* fn = llvm::Function::Create(
        fnType,
        llvm::Function::ExternalLinkage,
        kernel_name,
        module
    );

    llvm::BasicBlock::Create(module.getContext(), "entry", fn);

    int idx = 0;
    for (auto& arg : fn->args())
        arg.setName(kernel_arg_vector[idx++].name);

    return fn;
}

llvm::Value* Compiler::getValue(
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

llvm::Value* Compiler::castValueToType(llvm::Value* value, llvm::Type* targetType, llvm::IRBuilder<>& builder) {
    llvm::Type* srcType = value->getType();
    if (srcType == targetType) {
        return value;
    }

    if (srcType->isIntegerTy() && targetType->isFloatingPointTy()) {
        return builder.CreateSIToFP(value, targetType);
    }

    if (srcType->isFloatingPointTy() && targetType->isIntegerTy()) {
        return builder.CreateFPToSI(value, targetType);
    }

    if (srcType->isIntegerTy() && targetType->isIntegerTy()) {
        return builder.CreateIntCast(value, targetType, true);
    }

    if (srcType->isFloatingPointTy() && targetType->isFloatingPointTy()) {
        return builder.CreateFPCast(value, targetType);
    }

    throw std::runtime_error("Unsupported cast in store");
}

IRBaseType getArgType(const py::object& obj) {
    py::array arr = obj.attr("array").cast<py::array>();
    char dtype = arr.dtype().kind();
    int itemsize = arr.itemsize();

    if (dtype == 'f' && itemsize == 4) return IRBaseType::F32;
    if (dtype == 'f' && itemsize == 8) return IRBaseType::F64;
    if (dtype == 'i' && itemsize == 4) return IRBaseType::I32;
    if (dtype == 'i' && itemsize == 8) return IRBaseType::I64;

    throw std::runtime_error("Unsupported dtype");
}

} // namespace pyggpu