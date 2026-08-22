#pragma once

#include <iostream>
#include <string>
#include <pybind11/numpy.h>
#include "signature.h"
#include <cuda.h>

namespace py = pybind11;

namespace pyggpu{

enum class KernelTarget {
    CPU,
    GPU, // Automatically check which GPU is available and use it. If none is available, fallback to CPU.
    NVIDIA, // Will perhaps always be PTX (with some Cuda additions)?
    AMD,
    PTX,
    SPIR_V
};

class BaseKernel {
public:
    virtual ~BaseKernel() = default;
    virtual void launchFromTuple(py::tuple py_args) = 0;

protected:
    template <typename T>
    static T castPythonArg(const pybind11::handle& item) {
        py::object obj = py::reinterpret_borrow<py::object>(item);

        // Get the value from the KernelArg wrapper
        py::object inner_val = obj;
        if (py::hasattr(obj, "value")) {
            inner_val = obj.attr("value");
        }

        // If it is an array
        if constexpr (std::is_pointer_v<T>) {
            uintptr_t ptr_address = 0;
            // First have to cast to uintptr_t since the value might not be a Pybind11 registered object
            // could be a numpy array, torch tensor etc.
            if (py::isinstance<py::array>(inner_val)) {
                // Get raw buffer pointer from NumPy array
                auto arr = inner_val.cast<py::array>();
                ptr_address = reinterpret_cast<uintptr_t>(arr.data());
            } else if (py::hasattr(inner_val, "data_ptr")) {
                // PyTorch tensor or CuPy array with .data_ptr()
                ptr_address = inner_val.attr("data_ptr")().cast<uintptr_t>();
            } else if (py::hasattr(inner_val, "ptr")) {
                // Ctypes or custom pointer object
                ptr_address = inner_val.attr("ptr").cast<uintptr_t>();
            } else {
                // Assume inner_val is an integer memory address (e.g., int/uintptr_t)
                ptr_address = inner_val.cast<uintptr_t>();
            }

            return reinterpret_cast<T>(ptr_address);

        } 
        // Scalar
        else {
            if (py::isinstance<py::array>(inner_val)) {
                inner_val = inner_val.attr("item")();
            }
            return inner_val.cast<T>();
        }
    }
};

template <typename Sig>
class CPUKernel;

template <typename... Args>
class CPUKernel<Signature<Args...>> : public BaseKernel {
public:
    using Fn = void(*)(Args...);

    explicit CPUKernel(const std::string& symbol, void(*fn_ptr)(Args...))
    : symbol(symbol) {
        this->fn = reinterpret_cast<Fn>(fn_ptr);
        std::cout << "Kernel created with symbol: " << symbol << std::endl;
    }
    
    template <typename... LaunchArgs>
    void launch(LaunchArgs&&... args) {
        fn(std::forward<LaunchArgs>(args)...);
    }

    void launchFromTuple(py::tuple py_args) override {
        if (py_args.size() != sizeof...(Args)) {
            throw std::runtime_error("Argument count mismatch!");
        }
        
        launchFromTupleImpl(py_args, std::index_sequence_for<Args...>{});
    }

    Fn getPtr() const { return fn; }

private:
    Fn fn;
    std::string symbol;

    template <std::size_t... Is>
    void launchFromTupleImpl(pybind11::tuple args, std::index_sequence<Is...>) {
        using TupleT = std::tuple<Args...>;
        
        launch(
            castPythonArg<std::tuple_element_t<Is, TupleT>>(args[Is])...
        );
    }
};

template <typename Sig>
class PTXKernel;

template <typename... Args>
class PTXKernel<Signature<Args...>> : public BaseKernel {
public:
    PTXKernel(const CUfunction& fn, const std::string& kernel_name, SignatureID sig_id)
    : fn(fn), kernel_name(kernel_name), sig_id(sig_id) {
        std::cout << "PTXKernel created with kernel name: " << kernel_name << " and signature ID: " << sig_id << std::endl;
    }

    template <typename... LaunchArgs>
    void launch(LaunchArgs&&... args) {
        fn(std::forward<LaunchArgs>(args)...);
    }

    void launchFromTuple(py::tuple py_args) override {
        // Define grid and block dimensions
        int gridDimX = 128, blockDimX = 256;
        
        // Set up kernel argument pointers
        void* args[] = { /* pointers to device memory buffers */ };

        // Launch Kernel
        cuLaunchKernel(
            fn,
            gridDimX, 1, 1,    // Grid dimensions
            blockDimX, 1, 1,   // Block dimensions
            0, nullptr,        // Shared memory bytes & stream
            args, nullptr      // Kernel parameters
        );

        cuCtxSynchronize();
        //cuModuleUnload(cu_module);
        //cuCtxDestroy(cu_context);
    }

private:
    CUfunction fn;
    std::string kernel_name;
    SignatureID sig_id;
};

} // namespace pyggpu