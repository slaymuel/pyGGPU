#pragma once

#include <iostream>
#include <typeinfo>

enum class IRBaseType {
    F32,
    F64,
    I32,
    I64,
    F32Ptr,
    F64Ptr,
    I32Ptr,
    I64Ptr,
};

template <IRBaseType K>
struct MapType;

template <>
struct MapType<IRBaseType::F32Ptr> {
    using type = float*;
};

template <>
struct MapType<IRBaseType::F64Ptr> {
    using type = double*;
};

template <>
struct MapType<IRBaseType::I32> {
    using type = int32_t;
};

template <>
struct MapType<IRBaseType::I64> {
    using type = int64_t;
};

template<typename... Args>
struct Signature {
    static std::string asString() {
        return "void(" + ((std::string(typeid(Args).name()) + ", ") + ...) + ")";
    }
};

template <IRBaseType... Ks>
struct MakeSignature {
    using type = Signature<typename MapType<Ks>::type...>;
};
