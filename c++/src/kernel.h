#pragma once

#include <iostream>
#include <string>
#include "type_mapping.h"

namespace pyggpu{

class BaseKernel {
public:
    virtual ~BaseKernel() = default;
    virtual void launch() = 0;
    // virtual void* getPtr() const = 0;
};

template <typename Sig>
class Kernel;

template <typename... Args>
class Kernel<Signature<Args...>> : public BaseKernel {
public:
    using Fn = void(*)(Args...);
    Fn fn;

    Kernel(const std::string& symbol, void* fn_ptr)
    : symbol(symbol) {
        this->fn_ptr = reinterpret_cast<Fn>(fn_ptr);
        std::cout << "Kernel created with symbol: " << symbol << std::endl;
    }
    
    void launch() override{
        std::cout << "Launching kernel with symbol: " << symbol << std::endl;
        float ai[5] = {0};
        float aj[5] = {1, 2, 3, 4, 5};
        std::cout << "Launching kernel..." << std::endl;
        //reinterpret_cast<void(*)(float*, float*)>(fn_ptr)(ai, aj);
        fn_ptr(ai, aj);
        std::cout << "Kernel done." << std::endl;
        for(auto i = 0; i < 5; ++i) {
            std::cout << "ai[" << i << "] = " << ai[i] << std::endl;
        }
        for(auto i = 0; i < 5; ++i) {
            std::cout << "aj[" << i << "] = " << aj[i] << std::endl;
        }
    }

    Fn getPtr() const { return fn_ptr; }

private:
    Fn fn_ptr;
    std::string symbol;
};

} // namespace pyggpu