#pragma once

#include <iostream>
#include <string>
#include <pybind11/numpy.h>
#include "signature.h"

namespace py = pybind11;

namespace pyggpu{

class BaseKernel {
public:
    virtual ~BaseKernel() = default;
    virtual void launchFromTuple(py::tuple py_args) = 0;
    // virtual void* getPtr() const = 0;
};

template <typename Sig>
class Kernel;

template <typename... Args>
class Kernel<Signature<Args...>> : public BaseKernel {
public:
    using Fn = void(*)(Args...);

    explicit Kernel(const std::string& symbol, void(*fn_ptr)(Args...))
    : symbol(symbol) {
        this->fn = reinterpret_cast<Fn>(fn_ptr);
        std::cout << "Kernel created with symbol: " << symbol << std::endl;
    }
    
    template <typename... LaunchArgs>
    void launch(LaunchArgs&&... args) {
        fn(std::forward<LaunchArgs>(args)...);
    }

    //void launch() override{
    //    // Here we can check if the passed arguments already exist on the gpu or if we need to copy them.
    //    std::cout << "Launching kernel with symbol: " << symbol << std::endl;
    //    float ai[5] = {0};
    //    float aj[5] = {1, 2, 3, 4, 5};
    //    std::cout << "Launching kernel..." << std::endl;
    //    //reinterpret_cast<void(*)(float*, float*)>(fn)(ai, aj);
    //    //fn(ai, aj);
    //    std::cout << "Kernel done." << std::endl;
    //    for(auto i = 0; i < 5; ++i) {
    //        std::cout << "ai[" << i << "] = " << ai[i] << std::endl;
    //    }
    //    for(auto i = 0; i < 5; ++i) {
    //        std::cout << "aj[" << i << "] = " << aj[i] << std::endl;
    //    }
    //}

    void launchFromTuple(py::tuple py_args) {
        if (py_args.size() != sizeof...(Args)) {
            throw std::runtime_error("Argument count mismatch!");
        }
        
        launchFromTupleImpl(py_args, std::index_sequence_for<Args...>{});
    }

    Fn getPtr() const { return fn; }

private:
    Fn fn;
    std::string symbol;

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

    template <std::size_t... Is>
    void launchFromTupleImpl(pybind11::tuple args, std::index_sequence<Is...>) {
        using TupleT = std::tuple<Args...>;
        
        launch(
            castPythonArg<std::tuple_element_t<Is, TupleT>>(args[Is])...
        );
    }
};

} // namespace pyggpu