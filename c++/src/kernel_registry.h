#pragma once

#include <array>
#include <iostream>
#include <memory>
#include "type_list.h"
#include "kernel.h"

namespace pyggpu::kernel_registry {

// 1. Function Pointer Wrapper / Factory
template<typename Sig>
struct KernelEntry;

template<typename... Args>
struct KernelEntry<Signature<Args...>> {
    static std::unique_ptr<BaseKernel> make(const std::string& symbol, void* fn_ptr, KernelTarget target) {
        using Sig = Signature<Args...>;
        using Fn = void(*)(Args...);

        if (target == KernelTarget::CPU) {
            return std::make_unique<CPUKernel<Sig>>(symbol, reinterpret_cast<Fn>(fn_ptr));
        }
        else {
            throw std::runtime_error("Unsupported kernel target");
        }
    }
};

// 2. Registry Structures
struct RegistryEntry {
    SignatureID id;
    std::unique_ptr<BaseKernel> (*factory)(const std::string& symbol, void* fn_ptr, KernelTarget target);
};

template<typename AllSigs>
using RegistryTable = std::array<RegistryEntry, type_list::size<AllSigs>::value>;

// 3. Registry Factory Functions
template<typename Sig>
constexpr RegistryEntry makeEntry() {
    return {computeId<Sig>(), &KernelEntry<Sig>::make};
}

template<typename... Sigs>
constexpr RegistryTable<type_list::type_list<Sigs...>> makeRegistryImpl(type_list::type_list<Sigs...>) {
    return { makeEntry<Sigs>()... };
}

template<typename AllSigs>
constexpr auto makeRegistry() {
    return makeRegistryImpl(AllSigs{});
}

// 4. Debug / Utility Functions
template<typename... Sigs>
void printSignatures(type_list::type_list<Sigs...>) {
    ((std::cout << Sigs::asString() << "\n"), ...);
}

// 5. Metaprogramming Transformations
template<typename List>
struct toSignature;

template<typename... Ts>
struct toSignature<type_list::type_list<Ts...>> {
    using type = Signature<Ts...>;
};

template<typename List>
struct mapToSignature;

template<typename... Lists>
struct mapToSignature<type_list::type_list<Lists...>> {
    using type = type_list::type_list<typename toSignature<Lists>::type...>;
};

using AllSignaturesRaw = typename type_list::cartesian_product<
    AllowedTypes, AllowedTypes
>;

using AllSignatures = typename mapToSignature<AllSignaturesRaw>::type;

inline void dumpSignatures() {
    printSignatures(AllSignatures{});
}

inline constexpr auto registry_table = makeRegistry<AllSignatures>();

inline std::unique_ptr<BaseKernel> createKernel(const std::string& symbol, const SignatureID id, void* fn_ptr, KernelTarget target) {
    for (const auto& entry : registry_table) {
        if (entry.id == id) {
            return entry.factory(symbol, fn_ptr, target);
        }
    }
    
    throw std::runtime_error("Kernel signature ID " + std::to_string(id) + " not found!");
}

} // namespace pyggpu::kernel_registry