#include "kernel_registry.h"
#include "type_list.h"
#include "type_mapping.h"

namespace pyggpu {



//template<typename Sig>
//struct InstantiateKernel;
//
//template<typename... Ts>
//struct InstantiateKernel<Signature<Ts...>> {
//    template class Kernel<Signature<Ts...>>;
//    template class KernelEntry<Signature<Ts...>>;
//    template struct RegistryEntry<Signature<Ts...>>;
//};
//
//template<typename... Sigs>
//struct InstantiateAll;
//
//template<typename First, typename... Rest>
//struct InstantiateAll<First, Rest...> {
//    InstantiateKernel<First> _;
//    InstantiateAll<Rest...> __;
//};
//
//template<>
//struct InstantiateAll<> {};
//
//InstantiateAll<AllSignatures> instantiate_registry;
//
//const RegistryTable<AllSignatures> registry = make_registry<AllSignatures>();

}